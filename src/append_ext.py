import json
from config import RESULTS, REPORTS
d=json.load(open(f"{RESULTS}/ext_domain_shift.json"))
m=["\n\n---\n\n# 外部数据集分析 (External Validation - AH Glaucoma 2025)\n"]
m.append("## 数据情况\n")
m.append("- 上传内容: 343 个 CFP 眼底图(zip)+ 119 张 VF **灰度报告图**;**无数值VF表、无PLR3标签、无临床信息表**。")
m.append("- 提取出 342 张外部 CFP 眼底图;VF 为灰度可视化图(0–30dB梯度),非逐点数值。\n")
m.append("## 可做 / 不可做\n")
m.append("- ❌ 回归外部 MAE/R²、分类外部 ROC-AUC: **无法计算**(缺数值VF与PLR3真值)。")
m.append("- ✅ 域偏移量化(本节)。\n")
m.append("## 域偏移 FID/KID (R4-8)\n")
m.append("| 对比 | FID↓ | KID↓ |")
m.append("|---|---:|---:|")
m.append(f"| 外部CFP vs GRAPE CFP | {d['FID_external_vs_GRAPE']:.2f} | {d['KID_external_vs_GRAPE_mean']:.4f}±{d['KID_std']:.4f} |")
m.append(f"| (参考)GRAPE vs StyleGAN合成 | 36.63 | 0.0378±0.0037 |")
m.append(f"\n→ 外部中心(安徽)CFP 与 GRAPE 的分布差异(FID {d['FID_external_vs_GRAPE']:.1f})与'GRAPE vs 合成图'相当,**量化了中心间域偏移**(n_GRAPE={d['n_GRAPE']}, n_ext={d['n_external']})。")
m.append("\n## 要跑完整外部验证还需提供\n- 外部每只眼的**数值 VF(60/61点 dB)** —— 现仅有灰度图;\n- 外部 **PLR3 进展标签**。\n提供后即可计算外部 MAE/RMSE/R² 和分类 ROC-AUC/PR-AUC。\n")
open(f"{REPORTS}/复现报告.md","a").write("\n".join(m))
print("已追加外部数据集分析章节")
