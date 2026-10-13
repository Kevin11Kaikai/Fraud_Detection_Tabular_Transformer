"""Typed, training-only adaptation of the already frozen feature matrices."""
from __future__ import annotations

import gc
import time

import joblib
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import mutual_info_classif
from category_encoders import CatBoostEncoder

from .common import *
from ..kaggle_fraudsquad_reproduction_20261011.features import STRING_COLUMNS


def raw_missing(cat_names, n_train, n_valid, ids):
    """Restore missing versus unknown semantics lost in the tree code -1."""
    transaction = PROJECT / 'data/raw/train_transaction.csv'
    identity = PROJECT / 'data/raw/train_identity.csv'
    transaction_names = set(pd.read_csv(transaction, nrows=0).columns)
    identity_names = set(pd.read_csv(identity, nrows=0).columns)
    primitive = [n for n in cat_names if n in transaction_names or n in identity_names]
    tx = [n for n in primitive if n in transaction_names]
    frame = pd.read_csv(transaction, usecols=['TransactionID','TransactionDT']+tx,
                        nrows=n_train+n_valid).set_index('TransactionID')
    if not np.array_equal(frame.index.to_numpy(), ids):
        raise RuntimeError('Raw categorical row identity mismatch')
    result = {n: frame[n].isna().to_numpy() for n in tx}
    ident = [n for n in primitive if n in identity_names]
    if ident:
        pieces=[]
        for chunk in pd.read_csv(identity, usecols=['TransactionID']+ident, chunksize=50000):
            pieces.append(chunk.loc[chunk.TransactionID.isin(ids)].set_index('TransactionID'))
        data=pd.concat(pieces).reindex(ids)
        result.update({n:data[n].isna().to_numpy() for n in ident})
    return result, frame.TransactionDT.to_numpy()


class TypedAdapter:
    def fit(self, raw, names, categoricals, missing):
        self.names=names
        self.categorical=[n for n in names if n in categoricals]
        self.numeric=[n for n in names if n not in categoricals]
        self.positions={n:names.index(n) for n in names}
        a=raw[:,[self.positions[n] for n in self.numeric]].copy()
        a[a == -1]=np.nan
        with __import__('warnings').catch_warnings():
            __import__('warnings').simplefilter('ignore',RuntimeWarning)
            self.medians=np.nanmedian(a,axis=0)
        self.medians[~np.isfinite(self.medians)]=0.
        eligible=[j for j,n in enumerate(self.numeric) if np.isnan(a[:,j]).any() or n.startswith('id_') or 'uid' in n]
        # Group only identical training masks within the same information family.
        # Future rows use OR across the saved group's source fields. This retains
        # a missingness signal without copying the same V-block mask19 times.
        groups={}
        for j in eligible:
            n=self.numeric[j]
            family='identity' if n.startswith('id_') else ('uid' if 'uid' in n else 'common')
            key=(family,array_hash(np.packbits(np.isnan(a[:,j]))))
            groups.setdefault(key,[]).append(j)
        self.flag_groups=list(groups.values())
        self.flag_names=[family+'__missing_'+h[:12] for family,h in groups]
        self.flag_source_count=len(eligible)
        a=np.where(np.isfinite(a),a,self.medians)
        self.qt=QuantileTransformer(n_quantiles=1024,output_distribution='normal',subsample=100000,random_state=42).fit(a)
        self.vocabs={}
        for n in self.categorical:
            values=raw[:,self.positions[n]]
            values=values[np.isfinite(values)&(values!=-1)]
            counts=pd.Series(values).value_counts()
            entries=[(float(k),int(v)) for k,v in counts.items() if v>=2]
            entries.sort(key=lambda t:(-t[1],t[0]))
            self.vocabs[n]={k:j+2 for j,(k,v) in enumerate(entries[:50000])}
        self.uid_present=any('uid' in n for n in names)
        self.output_names=self.numeric+self.flag_names
        for n in self.categorical:self.output_names.extend([n+'__missing',n+'__unknown'])
        if self.uid_present:self.output_names.append('uid__unseen_training')
        return self

    def transform(self, raw, missing, uid_unseen=None):
        a=raw[:,[self.positions[n] for n in self.numeric]].copy()
        flag=~np.isfinite(a)|(a==-1)
        a=np.where(flag,self.medians,a)
        numeric=self.qt.transform(a).astype('float32')
        extra=[np.column_stack([np.any(flag[:,group],axis=1) for group in self.flag_groups]).astype('float32')]
        cats=[]
        for n in self.categorical:
            values=raw[:,self.positions[n]]
            # Composite strings are always formed, including missing primitives;
            # their -1 fitted code denotes an unknown composite, not missing.
            absent=np.asarray(missing.get(n,np.zeros(len(values),dtype=bool)))
            vocab=self.vocabs[n]
            codes=np.fromiter((1 if m else vocab.get(float(v),0) for v,m in zip(values,absent)),dtype=np.int64,count=len(values))
            cats.append(codes)
            extra.extend([absent[:,None].astype('float32'),(codes==0)[:,None].astype('float32')])
        if self.uid_present:extra.append(np.asarray(uid_unseen,dtype='float32')[:,None])
        return np.column_stack([numeric,*extra]).astype('float32'),np.column_stack(cats).astype('int64')


def historical_encode(cat,y,valcat,dt):
    cols=[str(j) for j in range(cat.shape[1])]
    frame=pd.DataFrame(cat.astype(str),columns=cols)
    vf=pd.DataFrame(valcat.astype(str),columns=cols)
    out=np.full(cat.shape,.5,np.float32)
    cuts=np.linspace(0,len(y),6,dtype=int)
    for j in range(1,5):cuts[j]=np.searchsorted(dt,dt[cuts[j]],side='left')
    for j in range(1,5):
        start,end=cuts[j:j+2]
        if dt[start-1]>=dt[start]:raise RuntimeError('Historical encoding timestamp boundary')
        enc=CatBoostEncoder(cols=cols,random_state=42,handle_unknown='value',handle_missing='value',a=1.)
        enc.fit(frame.iloc[:start],y[:start])
        out[start:end]=enc.transform(frame.iloc[start:end]).to_numpy(np.float32)
    enc=CatBoostEncoder(cols=cols,random_state=42,handle_unknown='value',handle_missing='value',a=1.).fit(frame,y)
    return out,enc.transform(vf).to_numpy(np.float32),enc,cuts


def excel_prepare(root,parts,schema):
    dest=root/'excel'
    if (dest/'receipt.json').exists():
        receipt=read(dest/'receipt.json')
        for name,h in receipt['files'].items():
            if sha(dest/name)!=h:raise RuntimeError('Excel adapter artifact changed')
        return receipt
    tick=time.perf_counter();dest.mkdir(exist_ok=True)
    tr,va=parts['train'],parts['validation']
    a,b,encoder,cuts=historical_encode(tr['x_cat'],tr['y'],va['x_cat'],tr['dt'])
    a=np.column_stack([tr['x_num'],a]);b=np.column_stack([va['x_num'],b])
    ix=np.random.default_rng(42).choice(len(a),20000,replace=False)
    mi=mutual_info_classif(a[ix],tr['y'][ix],random_state=42,discrete_features=False,n_neighbors=3)
    if not np.isfinite(mi).all() or mi.sum()<=0:raise RuntimeError('Invalid training MI weights')
    order=np.lexsort((np.arange(a.shape[1]),-mi));weights=(mi[order]/mi.sum()).astype('float32')
    for part,x in [('train',a),('validation',b)]:
        out=np.lib.format.open_memmap(dest/(part+'.npy'),mode='w+',dtype='float32',shape=x.shape)
        for off in range(0,len(x),32768):out[off:off+32768]=x[off:off+32768,order]
        out.flush();del out
    joblib.dump(encoder,dest/'categorical_encoder.joblib')
    np.save(dest/'mi_training_indices.npy',ix)
    receipt=dict(columns=[*schema['numeric_output_names'],*schema['categorical']],feature_order=order.tolist(),
                 mi_weights=weights.tolist(),chronological_block_boundaries=cuts.tolist(),training_only=True,
                 source_schema_sha256=sha(root/'schema.json'),elapsed_seconds=time.perf_counter()-tick,
                 files={p.name:sha(p) for p in dest.iterdir() if p.is_file()},finished_utc=now())
    write(dest/'receipt.json',receipt)
    print('EXCEL_PREPARED',root.name,'dims',a.shape[1],'seconds',round(receipt['elapsed_seconds'],1),flush=True)
    return receipt


def prepare():
    from .freeze import freeze
    protocol=freeze();tick=time.perf_counter()
    source=read(SOURCE/'derived/train_only/manifest.json');all_names=source['column_names']
    source_arrays={p:np.load(SOURCE/f'derived/train_only/{p}.npy',mmap_mode='r') for p in ('train','validation')}
    lab=labels();ntrain=len(lab['y_train']);nvalid=len(lab['y_validation'])
    categorical=set(STRING_COLUMNS)|{'card1_addr1','card1_addr1_P_emaildomain'}
    cat_names=[n for n in all_names if n in categorical]
    missing,dt=raw_missing(cat_names,ntrain,nvalid,np.r_[lab['ids_train'],lab['ids_validation']])
    with np.load(ABLATION/'strata_validation.npz') as z:
        if not np.array_equal(z['ids'],lab['ids_validation']):raise RuntimeError('Strata row identity')
        unseen=np.r_[np.zeros(ntrain,dtype=bool),~z['uid_seen']]
    families=read(ABLATION/'feature_groups.json')['variants']
    checks={}
    for variant in VARIANTS:
        root=HERE/'derived'/variant;root.mkdir(parents=True,exist_ok=True)
        if not (root/'schema.json').exists():
            names=families[variant];ix=[all_names.index(n) for n in names]
            raw=np.asarray(source_arrays['train'][:,ix],dtype='float32')
            proc=TypedAdapter().fit(raw,names,categorical,{k:v[:ntrain] for k,v in missing.items()})
            joblib.dump(proc,root/'adapter.joblib')
            for part,start,end in [('train',0,ntrain),('validation',ntrain,ntrain+nvalid)]:
                rawpart=np.asarray(source_arrays[part][:,ix],dtype='float32')
                xn,xc=proc.transform(rawpart,{k:v[start:end] for k,v in missing.items()},unseen[start:end])
                if not np.isfinite(xn).all() or not (xc>=0).all():raise RuntimeError('Invalid typed tensors')
                np.save(root/f'{part}_x_num.npy',xn);np.save(root/f'{part}_x_cat.npy',xc)
                for k in ['ids','y']:np.save(root/f'{part}_{k}.npy',lab[f'{k}_{part}'])
                np.save(root/f'{part}_dt.npy',dt[start:end])
                del rawpart,xn,xc
            schema=dict(variant=variant,source_columns=names,numeric_output_names=proc.output_names,categorical=proc.categorical,
                        cat_cardinalities=[len(proc.vocabs[n])+2 for n in proc.categorical],source_scope='train_only',
                        missing_mask_groups=[[proc.numeric[j] for j in group] for group in proc.flag_groups],
                        missing_source_columns=proc.flag_source_count,
                        missing_id=1,unknown_id=0,fit_rows=ntrain,test_accessed=False,
                        source_sha256=sha(SOURCE/'derived/train_only/manifest.json'),
                        preparation_code={str(p):sha(p) for p in [HERE/'prepare.py',HERE/'common.py']},
                        files={p.name:sha(p) for p in root.iterdir() if p.is_file()},finished_utc=now())
            write(root/'schema.json',schema)
            # Fixed tiny invariance fixture: fitted transform acts row-wise, not batch-wise.
            sample=np.asarray(source_arrays['validation'][:128,ix],dtype='float32')
            m={k:v[ntrain:ntrain+128] for k,v in missing.items()}
            a,b=proc.transform(sample,m,unseen[ntrain:ntrain+128])
            c,d=proc.transform(sample[:32],{k:v[:32] for k,v in m.items()},unseen[ntrain:ntrain+32])
            np.testing.assert_array_equal(a[:32],c);np.testing.assert_array_equal(b[:32],d)
            del raw,sample,proc;gc.collect()
        parts,schema=load(variant)
        for name,h in schema['preparation_code'].items():
            if sha(Path(name))!=h:raise RuntimeError('Prepared representation code changed')
        for name,h in schema['files'].items():
            if sha(root/name)!=h:raise RuntimeError('Prepared tensor fingerprint mismatch')
        checks[variant]=dict(source_fields=len(schema['source_columns']),numerical_inputs=len(schema['numeric_output_names']),
                             categorical_inputs=len(schema['categorical']),unknown_missing_distinct=True,
                             validation_batch_invariance=True,training_only_fit=True)
        print('FT_PREPARED',variant,checks[variant],flush=True)
        excel_prepare(root,parts,schema);del parts;gc.collect()
    write(HERE/'adaptation_verification.json',dict(status='PASS',checks=checks,elapsed_seconds=time.perf_counter()-tick,
         raw_training_only_statistic_policy=protocol['feature_policy'],final_test_accessed=False,finished_utc=now()))
    print('ADAPTATION_PASS',round(time.perf_counter()-tick,1),'seconds',flush=True)


if __name__=='__main__':
    # Persist class identity under the importable package, never __main__.
    from .prepare import prepare as entrypoint
    entrypoint()
