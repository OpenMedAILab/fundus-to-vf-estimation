# GRAPE 青光眼视野 / 分级复现与外部验证

基于 **GRAPE** 眼底彩照 (CFP) 数据集的青光眼分析复现工程，包含两大任务、四种网络结构、三种输入形态的完整训练/评估流水线，以及外部数据集（皖南）验证与论文配图生成脚本。

> ⚠️ **本仓库只含代码、配置、评估结果与配图**。原始数据集（GRAPE / external CFP）与模型权重（`ckpt/**/best.pth`，约 35 GB）**未纳入版本库**，见下方「未包含内容」。

## 任务与模型

- **任务**
  - `reg` — 视野 (VF) 估计，61 个测试点，缺失点用 `-1` 掩膜。
  - `cls` — PLR3 青光眼进展二分类（Focal loss + 加权采样处理类别不平衡）。
- **网络结构** (`--arch`)：`resnet` · `transformer` · `hybrid` · `full_hybrid`（cross-attention 融合）。
- **输入形态** (`--input`)：`cfp`（原图）· `roi`（视盘 ROI）· `annotated`（标注图）。
- **随机种子**：`0,1,2`（三次重复取均值 ± 标准差）。

数据按**患者**划分，无泄漏（train 92 / val 23 / test 29 例，见 `data/summary.json`）。

## 目录结构

```
repro.py               # 核心训练脚本（自包含：数据集/模型/训练/评估）
orchestrate.py         # 多 GPU 并行编排 (task×arch×input×seed)
warm.py / build_aug.py # 预热下载权重 / 构建增广(合成)样本索引
run_extra.py           # 补充实验批处理

aggregate.py           # 汇总各 run 的 result.json -> aggregate_summary.json
report.py              # 生成结果报表
append_*.py            # 追加外部/补充实验结果到汇总

# 分析与配图
clinical_analysis.py   grading_dca.py   longitudinal_analysis.py
gradcam.py             fid_kid.py       plot_convergence.py   make_supp_figures.py

# 外部验证（皖南医院数据，inference-only）
external_infer.py  external_final.py  external_pair_v2.py  external_read_ms.py
external_pseudocolor.py  external_ssim.py  ext_domain.py  wannan_ext_val.py

data/                  # 数据集划分索引 (cls/reg 的 train/val/test json) + summary
synthetic/             # 150 张合成/增广样本 (供 --aug 与 FID/KID 评估)
vf_layout.npy          # 61 点视野排布

ckpt/<run>/result.json # 每个 run 的评估指标（权重 best.pth 已排除）
logs/                  # 训练日志
*.png / supp_fig/ / wannan_extval/   # 论文正文及附录配图
数据结构说明.md 及各 *.md            # 数据结构说明与复现/实验报告（中文）
```

## 快速开始

```bash
pip install -r requirements.txt

# 配置数据路径（所有脚本经 config.py 统一读取；不设则默认到仓库目录下的同名文件夹）
export GRAPE_ROOT="/path/to/GRAPE Dataset/GRAPE Dataset"   # 含 CFPs/ ROI images/ Annotated Images/ 及 xlsx
export EXTERNAL_ROOT="/path/to/external"                    # 含 CFP/ 与 VF/（外部验证，可选）
# 可选：export GEN_DIR=... (StyleGAN 生成图)  export SIMHEI_TTF=... (中文字体)

# 单个 run
python repro.py --task reg --arch full_hybrid --input cfp --seed 0 [--aug]

# 全量并行（按 GPU 分发全部配置）
python orchestrate.py all --seeds 0,1,2 --gpus 0,1,2,3

# 汇总结果
python aggregate.py && python report.py
```

> 所有路径集中在 **`config.py`**：仓库自身位置自动作为 `ROOT`，数据集等外部路径通过环境变量覆盖（各有默认值）。无需再手改任何脚本。

## 未包含内容（需自行准备）

| 内容 | 说明 |
|------|------|
| `GRAPE Dataset/` | 原始眼底彩照数据集，请从官方渠道申请获取 |
| `external/` (CFP) | 外部验证原始影像（含受试者信息，不可公开分发） |
| `ckpt/**/best.pth` | 训练得到的模型权重（约 35 GB），仅保留了 `result.json` 指标 |

## ⚠️ 重要提示

1. **路径配置**：所有脚本的数据/输出路径已统一到 `config.py`，由环境变量覆盖（见「快速开始」），
   不再有指向特定服务器的硬编码绝对路径。
2. **患者隐私**：外部验证结果文件（`external_ms_pairing.json`、`external_pred_ms.json`、
   `wannan_extval/wannan_extval_pairs.json`）中的真实受试者姓名已**脱敏为匿名 ID**（`P001`…），
   姓名↔ID 映射表 `deid_mapping.json` 仅保留在本地、未纳入版本库。
