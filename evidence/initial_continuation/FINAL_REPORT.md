# XGBoost→ExcelFormer B1C 配对蒸馏报告

结果状态：**CONTINUATION_TARGET_ACHIEVED_DISTILLATION_GAIN_NOT_ESTABLISHED**。验证状态：**PASS**。

**单Transformer管线超过既定LightGBM参考；本轮没有观察到蒸馏相对匹配BCE续训的额外优势。**

## 从输入到输出

原始交易及identity协变量进入已冻结B1C预处理，生成335个token输入；ExcelFormer输出欺诈概率并排序。训练时，三个过去前缀XGBoost分别产生后续TRAIN块的软标签，随原Hid-Mix同权重混合；推理不需要XGBoost。教师的特征映射只拟合各自过去前缀，学生保留原TRAIN批量映射，因此整条管线不宣称流式预处理。

## 配对结果

| 模型 | AP ↑ | Recall@Top1% ↑ | Precision@Top1% ↑ | Recall@Top5% ↑ | Brier ↓ |
|---|---:|---:|---:|---:|---:|
| ExcelFormer B1C parent | 0.601916 | 27.02% | 92.78% | 62.62% | 0.019731 |
| bce_control | 0.614284 | 27.42% | 94.13% | 64.56% | 0.019484 |
| xgb_kd_lambda1 | 0.613028 | 27.15% | 93.23% | 64.40% | 0.020573 |
| 常规LightGBM（同rich_all输入） | 0.608338 | 26.46% | 90.86% | 65.29% | 0.020009 |
| 原强LightGBM（rich_all） | 0.633859 | 27.05% | 92.89% | 66.50% | 0.019143 |
| LightGBM B1C（Excel B1C同输入对照） | 0.687708 | 27.98% | 96.05% | 70.58% | 0.016913 |
| 冠军XGB公开组件（train-only） | 0.671557 | 28.53% | 97.97% | 67.23% | 0.019389 |

两条训练共同完成5轮；主要归因比较只在该共同范围内选择checkpoint。蒸馏减BCE的AP差值：-0.0012559647462662138。原checkpoint保留为epoch0候选，避免将退化模型交付为提升。

BCE相对原B1C的AP变化为+0.012368，Top1%抓获数变化为+12。蒸馏相对BCE的AP变化为-0.001256，Top1%抓获数变化为-8，Brier变化为+0.001089。蒸馏ROC-AUC较高，但预先固定的主指标是AP，没有改用AUC宣布获胜。

原B1C最优出现在原训练最后一轮，继续训练本身可能有益；BCE对照用于区分这部分收益。教师全TRAIN模型的AP0.671557不是学生性能保证，前缀教师也不冒用该指标。温度固定1，λ固定1，未搜索额外教师、架构或seed。

三个TRAIN后续块的教师AP分别为0.589873, 0.678231, 0.611154；平均预测风险分别为1.24%, 1.50%, 1.65%，真实欺诈率分别为3.52%, 4.09%, 3.93%。这些是不同TRAIN块的描述性OOF结果，不能并入主验证表排名，也没有用于选择教师轮数。教师风险概率偏低是观察到的信号限制；它是否造成学生损失仍由配对结果评估。详见teacher_diagnostics.json。

本轮对照识别的是完整masked distillation目标的净效果；可用教师掩码也会改变后75%TRAIN样本的有效损失权重，不能将结果进一步归因于教师知识或校准的某个独立机制。教师概率偏低、前缀时间漂移和规则表示迁移是后续假设，本轮没有追加试验验证这些机制。

## 比较边界与时间泛化

常规rich_all LightGBM AP0.608338是已声明的管线参考，B1C增加identity/UID表示，超过该参考不等于同输入架构优越。保留原强LightGBM AP0.633859和B1C同输入LightGBM AP0.687708，未削弱或重训基线。固定五个连续验证时间块和UID/identity分层结果见temporal_uid_metrics.csv；分层Top-K使用组内预算，不能当作全局调查预算收益。

BCE的五个时间块AP依次为0.739567, 0.645878, 0.625303, 0.536364, 0.513576，晚期下降仍明显。UID seen/unseen的AP分别为0.716807/0.570083；identity observed/missing分别为0.782685/0.426678。这些组的欺诈率和样本构成不同，不能将差异全部归因于某一字段。蒸馏在最后时间块AP为0.523726，高于BCE该块0.513576，但全验证AP仍较低；这是探索性异质性观察，没有据此改换主方法或评价角色。图见temporal_uid_diagnostics.png。

本数据是反复用于选择的开发validation；没有访问剩余test标签或官方竞赛test。单seed、短续训和validation选择不能建立统计显著性、独立未来泛化或部署保证。召回变化完整披露，不作为新增硬验收门槛。

## 资源与验证

本轮累计GPU租约内用时73.73分钟，上限120分钟；排队等待另计。租约配置和释放开销也计入whole-lease成本，原始窗口见budget.json。逐trial训练/等待/显存/参数量见ablation_results.csv与各result.json。教师CPU准备成本见teacher/features_receipt.json。历史预算保持原样，未将旧训练清零。

当前累计排队等待126.91分钟；两条模型均470738个参数，训练峰值reserved显存均约3.09GiB。所有GPU工作使用现有FIFO，未终止其他线程。

代码、原模型与输入哈希、同采样及Hid-Mix配对、保存预测复算、fresh-model checkpoint重载证据见verification.json；本轮为作者自查，不称独立第三方验证。未完成和失败记录保存在campaign_state.json、failure记录及budget.json。

复现入口见REPRODUCTION.md。后备LightGBM蒸馏没有执行。

经过共同训练范围比较后，AP最高的单Transformer保存在selected_model.pt，其来源、输入缓存、指标与哈希见selected_model_card.json。如果最终保留原checkpoint或选择BCE续训，不将其称为蒸馏成果。
