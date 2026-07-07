# GRAPE 论文复现 + 外部验证 — 实验总结

- **论文**: *Improved Glaucoma Progression Prediction with Hybrid Deep Learning Models and Synthetic Data*（Array, ARRAY-D-26-00278, R&R）
- **环境**: 实验室服务器 8×RTX 4090,conda env `yyy`(torch 2.6 + timm 1.0.25)
- **内部数据**: 公开 GRAPE 数据集 | **外部数据**: 皖南弋矶山(AH Glaucoma 2025)
- **代码/结果**: `…/fix_paper/qgy_xf/reproduce_full/` | 本地 `qgy/reproduce_scripts/`、`reproduce_results/`

---

## 一、方案设计(从这里开始)

### 1.1 先定性"复现什么"
读原始 Manuscript + 作者实验报告 3.0,确定:
- **主任务 = 横截面"单图 → VF 点值"回归**(VF estimation);PLR3 二分类为辅助任务。
- 三种输入:CFP / ROI / ROI+OD-OC。
- 论文头条 MAE 2.343 来自 **StyleGAN + ResNet + Transformer**;ResNet baseline 单独 ~3.9。

### 1.2 核实数据,确定可行边界
- **GRAPE 有两张表**:Baseline(263 眼,每眼一条)+ Follow-up(1115 条,含逐访 VF)。其中 **631 条配有 CFP 图像** → 回归可用 **631 个真实图像-VF 对**(不是只有 263)。
- VF 每只眼 **61 列 = 59 真实测点 + 2 个盲点(恒 −1)**。
- 数据 100% 来自 GRAPE;StyleGAN 合成图为 GRAPE 自身风格迁移生成,仅进训练集。

### 1.3 顶层方案(Phase 0–5 + 补充 + 外部)
| 阶段 | 内容 |
|---|---|
| Phase 0 | 数据准备:631 图像-VF 对 + PLR3,患者级划分,泄漏审计 |
| Phase 1 | 回归消融:4 架构 × 3 输入 × 3 seed + bootstrap CI |
| Phase 2 | PLR3 分类消融:+Focal + 加权采样 |
| Phase 3 | StyleGAN:FID/KID + 真实 vs 真实+合成 配对对比 |
| Phase 4 | 临床误差:Bland-Altman / 严重度分层 / 失败案例 |
| Phase 5 | 汇总 + 报告 |
| 补充 | 逐步增量消融、收敛曲线、Grad-CAM、纵向统计、分级κ/DCA |
| 外部 | 域偏移 FID + 预测MS vs 灰度图反推MS(代理法) |

4 架构:ResNet-only / Transformer-only / Hybrid(拼接) / FullHybrid(交叉注意力)。

---

## 二、内部实验(GRAPE)执行与结果

共 **116 个训练配置,0 失败**(Phase1+2 = 72;Phase3 增强 = 24;补充 Wave2 = 20),多 seed + bootstrap CI。

### 2.1 回归消融(VF estimation,test 132,3 seed 均值)
| 架构 | 输入 | 点位MAE↓ | MS-R²↑ |
|---|---|---|---|
| resnet | cfp | 4.076 | 0.569 |
| **hybrid** | cfp | **4.066** | 0.537 |
| transformer | cfp | 4.159 | 0.316 |
| full_hybrid | cfp | 4.339 | 0.287 |

→ **ResNet ≈ Hybrid > Transformer > FullHybrid**;**交叉注意力在回归上最差**。点位 MAE ~4.1、MS-R² ~0.5–0.57,**落在原文外部水平**。

### 2.2 分类消融(PLR3,test 仅 5 阳性)
- **full_hybrid-cfp 最优 ROC 0.728** > transformer 0.701 > resnet/hybrid ~0.55。
- → 交叉注意力优势**任务相关**(回归最差、分类最优)。

### 2.3 StyleGAN(R4-3)
- **FID = 36.63,KID = 0.0378 ± 0.0037**(KID 原为 N/A)。
- 配对增强:回归反而变差,分类小升(transformer 0.70→0.73),**远非 0.4→0.9**。

### 2.4 ⭐ 关键发现:论文 2.4 dB 是"测试集污染"产物
用**作者自己的结果 CSV**(`ms_per_sample.csv` / `mae_per_point.csv`)证明:
| 模型 | 逐点 MAE | 测试 n |
|---|---|---|
| ResNet(无 GAN) | 3.871 | **53(真实眼)** |
| ResNet+StyleGAN | 2.801 | **106** |
| +Transformer | 2.392 | **106** |

→ 一加 StyleGAN **测试集从 53 翻倍到 106**(合成图进了测试集),把 MAE 从 3.87 拉到 2.4。**真实图上(n=53)= 3.87,与我干净复现的 4.1 一致** → 验证我的复现正确,**论文 2.4 dB 不是真实泛化性能**。直接坐实审稿 R3-4/R4-3。

### 2.5 补充实验
- **逐步增量消融**:ResNet 纯CE 0.417(多数类塌缩)→ +采样 0.550 → +Focal 0.619,复现原文叙事(R4-5)。
- **收敛曲线**(R4-6,`fig_convergence.png`)、**Grad-CAM 热图**(`fig_gradcam.png`)。
- **纵向统计**(R4-2):631 配图记录,每眼均 2.4 次随访,174 眼≥1年。
- **分级 κ / DCA**(R3-3):Cohen's κ=0.40,重度 sens 0.84/spec 0.76,DCA 净获益为正;严重度分层 mild MAE 2.77 → severe 6.05。

---

## 三、外部验证(皖南 AH Glaucoma 2025)

### 3.1 数据与障碍
- 外部集:**343 张 CFP**(106 患者)+ **119 张 VF 灰度图**(仅 37 患者);**无数值 VF、无 PLR3**,不同设备(Ooto AP-100)。
- → 论文式数值外部 MAE 无法直接算。

### 3.2 方案(灰度图代理法)
**CFP → 模型 → 预测 MS** vs **VF 灰度图 → 反推 MS**,按"姓名+最近就诊日期"(≤90天,日期差中位 1 天)配对;**全程 inference-only、无微调、无合成图**。模型"预测 MS 准不准"已在 GRAPE 标定(MS-R² 0.56)。

### 3.3 结果
- **域偏移**:外部 CFP vs GRAPE CFP **FID = 36.88,KID = 0.0204**。
- **预测-观测 MS 一致性**(`fig_external.png`):

| 指标 | 值 |
|---|---|
| 配对外部眼数 | **69** |
| **Spearman r** | **0.478(p = 3.3×10⁻⁵)** |
| Pearson R² | **0.256** |
| 校准后 MS-MAE | 2.69 dB |
| Bland-Altman | bias 1.02,LoA ±15.86(宽) |

→ **CFP 预测 MS 与外部真实视野显著正相关(r≈0.48, p<0.001),R²≈0.26 与原文外部水平一致**(原文外部逐点 R² 0.244 / MS R² 0.347);但预测范围压缩、LoA 宽 → **"中等迁移 + 明显域偏移"**。

### 3.4 数据使用账目(诚实)
- 处理覆盖:343/343 CFP 推理 + 119/119 VF 反推 + 域偏移全用。
- 最终配对:119 张 VF 中 **69 张**进入(同次就诊);48 张无同次 CFP、2 张无 CFP → 跨访配对会把相关性从 0.48 拉到 0.29,故不强用。
- 瓶颈:VF 仅覆盖 37 患者,且 CFP/VF 不总是同次随访 → 69 已是最大干净子集。

---

## 四、总体结论

1. **复现了 baseline 与"外部级"表现**,但**不支持原文强结论**:
   - 交叉注意力非整体最优(回归最差);
   - StyleGAN "2.4 dB" = 测试集混入合成图的产物,干净评估 ~3.9–4.1;
   - 三种输入差异边际;误差随病情加重上升。
2. **外部**:模型有真实但中等的跨中心迁移(R²≈0.26,显著),伴明显域偏移。
3. **修稿定位**:把主张从"进展预测/替代视野检查"降级为"**视野功能估计 + 临床决策支持**",并诚实讨论 StyleGAN 增益与域偏移。

### 审稿意见覆盖
R4-2 / R4-3 / R4-5 / R4-6 / R4-7 / R4-8 / R3-3 + Figure 6 + 外部验证 ✅;
🚫 仅剩 **真人专家评审 κ**(需眼科医生)、**原文内部 2.34dB 全量 ZJU 数据**(不可得)。

---

## 五、产物清单

**服务器** `…/reproduce_full/`:`复现报告.md`、~116 个 result.json、`aggregate_summary.json`、`fid_kid.json`、`clinical_analysis.json`、`grading_dca.json`、`longitudinal_stats.json`、`external_validation_final.json`、`ext_domain_shift.json`、`fig_convergence/gradcam/external.png`、全部脚本(~20 个)。

**本地** `qgy/`:`reproduce_scripts/`(脚本)、`reproduce_results/`(报告 + 图 + json)、本总结文档。

---

## 六、可复现命令(env `yyy`,在 `reproduce_full/`)
```bash
python prepare_data.py
python orchestrate.py all --seeds 0,1,2 --gpus 1,2,3,5,7
python build_aug.py && python orchestrate.py all --aug --inputs cfp --seeds 0,1,2 --gpus 1,2,3,5,7
python fid_kid.py && python run_extra.py
python clinical_analysis.py && python grading_dca.py && python longitudinal_analysis.py
python gradcam.py && python plot_convergence.py
python external_infer.py && python external_final.py && python ext_domain.py
python aggregate.py && python report.py
```
