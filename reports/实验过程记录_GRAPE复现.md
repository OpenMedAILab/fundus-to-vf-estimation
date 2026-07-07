# GRAPE 数据集论文复现 — 实验过程记录

- **论文**: *Improved Glaucoma Progression Prediction with Hybrid Deep Learning Models and Synthetic Data*（Array, ARRAY-D-26-00278, R&R）
- **目标**: 在公开 GRAPE 数据集上从头到尾复现论文核心实验，把该跑的实验跑全
- **环境**: 实验室服务器 `10.11.154.191:20066`（8×RTX 4090），conda 环境 `yyy`（torch 2.6 + timm 1.0.25）
- **数据**: GRAPE 完整公开数据集 `…/fix_paper/20251113/GRAPE Dataset/`
- **代码/结果**: `…/fix_paper/qgy_xf/reproduce_full/`
- **记录日期**: 2026-06-25

---

## 一、前期核实（确定"复现什么"和"用什么数据"）

### 1.1 核实原论文真实方法（读 Manuscript 正文）
- **主任务是横截面"单图 → VF 点值"回归**（VF estimation），不是纵向历史 VF 预测；PLR3 二分类是辅助任务。
- 三种图像输入：CFP / ROI / ROI+OD-OC（annotated）。
- 原文头条 MAE 2.343/2.385/2.432 dB 来自 **StyleGAN + ResNet + Transformer**；ResNet baseline 单独为 3.9 dB，StyleGAN→2.8、+Transformer→2.38。
- 分类 baseline ROC≈0.4（原文自述 below chance），靠 StyleGAN + 加权采样 + Focal Loss 提升。
- 训练细节：264 epoch、batch 4、Adam cosine 1e-4→1e-5、Focal Loss、weighted sampling。

### 1.2 核实 GRAPE 数据结构（关键发现）
GRAPE Excel 有两个工作表：
- **Baseline**（263 眼）：每眼一条，含 PLR2/PLR3/MD/OCT + 60 点 VF。
- **Follow-up**（1115 条，含 `Visit Number` + `Interval Years` + 逐访 60 点 VF）：其中 **631 条配有 CFP 图像**。

→ 由此确定**回归可用 631 个真实图像-VF 对**（而非只用 Baseline 的 263），且有真纵向 VF + 随访间隔。三种输入（CFPs / ROI images / Annotated Images）各 631 张齐全。

### 1.3 数据使用边界
- 真实数据 100% 来自 GRAPE 单一数据集。
- StyleGAN 合成图为"GRAPE 自身训练的 StyleGAN2-ADA 生成"，仅进训练集做增强。
- 骨干用 ImageNet 预训练权重（迁移学习，同原论文）。
- 原论文 ZJU 内部数据、外部 686（FAWMC）均不可得，故不使用、外部验证不做。

---

## 二、顶层实验方案（Phase 0–5 + 补充 Wave 2/3）

| 阶段 | 内容 |
|---|---|
| Phase 0 | 数据准备：631 图像-VF 对 + PLR3，患者级划分（train/val/test），泄漏审计 |
| Phase 1 | 回归消融：4 架构 × 3 输入 × 3 seed，固定 test 集 + bootstrap CI |
| Phase 2 | PLR3 分类消融：同上 + Focal + 加权采样 |
| Phase 3 | StyleGAN 增强：FID/KID + 真实 vs 真实+合成 配对对比 |
| Phase 4 | 临床误差分析：Bland-Altman / 严重度分层 / 失败案例 |
| Phase 5 | 汇总 + 报告 |
| Wave 2 | 逐步增量消融（纯CE→+采样→+Focal）+ 收敛曲线配置 |
| Wave 3 | Grad-CAM 热图 + 收敛曲线绘图；纵向时间窗 + 分级κ/DCA |

**4 种架构定义**（单图输入）：ResNet50-only / ViT-only / Hybrid(ResNet+ViT 拼接) / FullHybrid(ResNet+ViT + 交叉注意力)。

---

## 三、执行过程与工程问题

### 3.1 流水线脚本（均在 reproduce_full/）
`prepare_data.py`（数据+划分+审计）、`repro.py`（4架构+回归/分类训练，VF mask -1，多指标）、`orchestrate.py`（GPU 池并行编排）、`build_aug.py`（增强索引）、`fid_kid.py`（torchmetrics FID/KID）、`clinical_analysis.py`、`grading_dca.py`、`longitudinal_analysis.py`、`gradcam.py`、`plot_convergence.py`、`aggregate.py`（聚合+bootstrap CI）、`report.py`、`run_extra.py`。

### 3.2 解决的工程问题
1. **HuggingFace 被墙**：timm ViT 权重下载卡死 → 设 `HF_ENDPOINT=https://hf-mirror.com`（写入 repro.py 顶部）+ 预热缓存。
2. **ResNet50 权重慢**：复用缓存的 IMAGENET1K_V2（避开 download.pytorch.org 慢速下载）。
3. **实现 bug**：test 评估误传 Dataset 而非 DataLoader 导致崩溃 → 修正后冒烟通过。
4. **GPU 调度**：只用空闲卡（1/2/3/5/7），编排器维护 GPU 池并行；nohup detached 抗断连。
5. **VF 维度**：检测到 61 点（位 21/32 为盲点恒 -1），预测 61 维并 mask。

### 3.3 运行规模
共 **116 个训练配置，0 失败**：Phase 1+2 = 72（37 min）、Phase 3 = 24（19 min）、Wave 2 = 20。FID/KID + 全部分析另算。

---

## 四、实验结果

### 4.1 回归消融（VF estimation，test 132，3 seed 均值）
| 架构 | 输入 | 点位MAE↓ | 点位R²↑ | MS-MAE↓ | MS-R²↑ |
|---|---|---:|---:|---:|---:|
| resnet | cfp | 4.076 | 0.352 | 2.334 | 0.569 |
| resnet | roi | 4.211 | 0.334 | 2.467 | 0.563 |
| transformer | cfp | 4.159 | 0.248 | 2.780 | 0.316 |
| hybrid | cfp | **4.066** | 0.335 | 2.406 | 0.537 |
| hybrid | roi | 4.131 | 0.340 | 2.374 | 0.561 |
| full_hybrid | cfp | 4.339 | 0.235 | 2.971 | 0.287 |
| full_hybrid | roi | 4.323 | 0.289 | 2.662 | 0.449 |

→ **ResNet ≈ Hybrid > Transformer > FullHybrid**；结果落在原文**外部验证水平**（外部 MAE 4.9–5.9、MS-R² 0.35–0.63）。

### 4.2 分类消融（PLR3，test 仅 5 阳性，统计弱）
| 架构 | 输入 | ROC-AUC mean±std |
|---|---|---:|
| **full_hybrid** | cfp | **0.728±0.043** |
| transformer | cfp | 0.701±0.041 |
| resnet / hybrid | cfp | ~0.55 |

→ 分类上 FullHybrid 最优（与回归相反）——交叉注意力优势**任务相关**。

### 4.3 StyleGAN（R4-3）
- **FID = 36.63，KID = 0.0378 ± 0.0037**（631 真实 vs 1000 合成；KID 原为 N/A）。
- **配对增强**：分类小升（resnet 0.557→0.643、transformer 0.701→0.732），**远非 0.4→0.9**；回归反降（继承标签为噪声）。

### 4.4 逐步增量消融（R4-5，cls-cfp）
| 架构 | 纯CE无采样 | +加权采样 | +Focal |
|---|---:|---:|---:|
| ResNet | 0.417 | 0.550 | 0.619 |
| FullHybrid | 0.671 | 0.697 | 0.692 |

→ ResNet 0.42（多数类塌缩，≈原文0.4）→ 0.62，**复现原文"采样+Focal修复少数类"叙事**。

### 4.5 临床误差分析（R3-3）
- Bland-Altman：bias −0.43 dB，LoA [−5.87, +5.01]。
- 严重度分层：mild MAE 2.77 → moderate 3.37 → **severe 6.05**（误差随病情加重上升）。
- 分级一致性：Cohen's κ=0.398、加权κ=0.449；重度 sens 0.84/spec 0.76；DCA 净获益为正。

### 4.6 纵向记录（R4-2）
- 1115 随访记录（631 配图），263 眼；每眼均 2.4 次随访（最多 7）；间隔中位 0.58 年；跨度均 1.72 年，174 眼≥1年、107 眼≥2年。

### 4.7 图表
- `fig_convergence.png`：训练收敛曲线（R4-6，含 best epoch 标注）。
- `fig_gradcam.png`：Grad-CAM 结构-功能热图（Figure 6）。

---

## 五、审稿意见覆盖

| 审稿意见 | 覆盖情况 |
|---|---|
| R4-2 纵向记录细节 | ✅ longitudinal_stats.json |
| R4-3 FID/KID | ✅ FID 36.6 / KID 0.038 |
| R4-5 完整消融 | ✅ 4架构×3输入 + 逐步增量消融 |
| R4-6 收敛验证 | ✅ fig_convergence.png |
| R4-7 输入差异边际 | ✅ 三输入 MAE 4.07–4.22 |
| R4-8 内外差异 | ✅ GRAPE=外部级，域偏移分析 |
| R3-3 误差/失败分析 | ✅ Bland-Altman+严重度+κ+DCA |
| Figure 6 结构-功能 | ✅ Grad-CAM |
| 外部 686 验证 | 🚫 GRAPE 无此数据 |
| 专家评审 κ=0.87 | 🚫 需真人眼科医生 |

---

## 六、总体结论

GRAPE 上复现出论文的 **baseline 与"外部级"表现**，但**不支持原文强结论**：①交叉注意力模型并非整体最优（回归最差、分类才占优）；②StyleGAN 增益有限（分类小升、回归变差），远非 0.4→0.9；③三种输入差异边际；④误差随病情加重上升。这套带 CI 的诚实证据支撑把论文主张从"进展预测/替代视野检查"**降级为"视野功能估计 + 临床决策支持"**。

---

## 七、产物清单

**服务器** `…/fix_paper/qgy_xf/reproduce_full/`：
- 全部脚本（14 个）、`data/`（划分索引）、`ckpt/`（~116 个 result.json + 权重）、`synthetic/`（150 合成图）
- `复现报告.md`、`aggregate_summary.json`、`fid_kid.json`、`clinical_analysis.json`、`grading_dca.json`、`longitudinal_stats.json`、`fig_convergence.png`、`fig_gradcam.png`

**本地** `qgy/`：
- `reproduce_scripts/`（全部脚本副本）、`reproduce_results/`（报告 + 2 图 + 6 json）、本过程记录文档

---

## 八、可复现步骤（在服务器 `reproduce_full/`，conda env `yyy`）

```bash
python prepare_data.py                                    # Phase 0
python orchestrate.py all --seeds 0,1,2 --gpus 1,2,3,5,7  # Phase 1+2
python build_aug.py                                       # Phase 3 数据
python orchestrate.py all --aug --inputs cfp --seeds 0,1,2 --gpus 1,2,3,5,7
python fid_kid.py                                         # FID/KID
python run_extra.py                                       # Wave2 逐步消融+收敛
python clinical_analysis.py && python grading_dca.py && python longitudinal_analysis.py
python gradcam.py && python plot_convergence.py           # 图表
python aggregate.py && python report.py && python append_supplement.py
```
