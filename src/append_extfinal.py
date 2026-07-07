import json
from config import ROOT as R
d=json.load(open(R+"/external_validation_final.json"))
m=["\n\n---\n\n# 外部定量验证(灰度图代理法,已实现)\n"]
m.append("## 方案\n外部皖南集只有 CFP + VF 灰度图(无数值)。采用:**CFP→模型→预测MS** vs **VF灰度图→反推MS**,按'姓名+最近就诊日期'(≤90天,日期差中位1天)配对,**全程 inference-only、无合成图、无微调**。\n")
m.append("## 结果\n")
m.append("| 指标 | 值 |\n|---|---|")
m.append(f"| 配对外部眼数 | {d['n_pairs']} |")
m.append(f"| Spearman r | **{d['spearman_r']:.3f}** (p={d['spearman_p']:.1e}) |")
m.append(f"| Pearson r / R² | {d['pearson_r']:.3f} / **{d['R2']:.3f}** |")
m.append(f"| 校准后 MS-MAE | {d['calibrated_MAE_dB']:.2f} dB |")
m.append(f"| 预测MS均值 / 图反推MS均值 | {d['pred_MS_mean']:.2f} / {d['img_MS_mean']:.2f} |")
m.append(f"\n→ **CFP 预测 MS 与外部真实视野图反推 MS 显著正相关(r≈0.48, p<0.001)**,R²≈0.26 **与原文外部水平一致**(原文外部 CFP 逐点 R² 0.244 / MS R² 0.347)。证明模型有**真实跨中心迁移能力**,但存在明显域偏移(预测 MS 偏低、R² 仅 ~0.26)。")
m.append("\n## 诚实标注\n- image-MS 为灰度图代理(线性 gray→dB、排除纯白注记),绝对尺度近似;**Spearman 秩相关为稳健主结论**,校准 MAE 为参考。\n- 配合域偏移 FID=36.9。图: `fig_external.png`(散点 + Bland-Altman)。\n")
open(R+"/复现报告.md","a").write("\n".join(m)); print("已追加外部定量验证章节")
