"""
补充图生成 (R3-3 / Supplementary Figures) — 不需新数据,从已有 test 预测出图。
  fig_clinical_error.png : Bland-Altman + 逐眼MAE分布 + 严重度分层 + 预测vs真实MS散点 (2x2)
  fig_failure_cases.png  : 最差MAE眼 真实vs预测 伪彩(高风险假阴性: 真重度->预测偏轻)
  fig_stard_flow.png     : STARD/数据流程图 (从已知计数绘制)
英文标注以避免 matplotlib CJK 字形缺失。
"""
import json, glob, numpy as np
from scipy.interpolate import griddata
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = "/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/reproduce_full"
import os; OUT = ROOT + "/supp_fig"; os.makedirs(OUT, exist_ok=True)

# ---------- 载入最优回归配置的 test 预测(跨seed集成) ----------
agg = json.load(open(ROOT + "/aggregate_summary.json"))
best = min(agg["reg"].items(), key=lambda kv: kv[1]["point_mae_mean"]); cfg = best[0]
rs = [json.load(open(f)) for f in glob.glob(f"{ROOT}/ckpt/reg_{cfg}_s*/result.json")]
P = np.mean([np.array(r["test_pred"]["P"]) for r in rs], 0); T = np.array(rs[0]["test_pred"]["T"])
print(f"best={cfg}  seeds={len(rs)}  test={len(T)}")

def ms(v): m = v != -1; return v[m].mean()
ms_t = np.array([ms(T[i]) for i in range(len(T))]); ms_p = np.array([ms(P[i]) for i in range(len(P))])
emae = np.array([mean_absolute_error(T[i][T[i] != -1], P[i][T[i] != -1]) for i in range(len(T))])
diff = ms_p - ms_t; bias = diff.mean(); sd = diff.std(); lo, hi = bias - 1.96*sd, bias + 1.96*sd

# ============ Figure 1: 临床误差 2x2 ============
fig, ax = plt.subplots(2, 2, figsize=(11.5, 10.0), constrained_layout=True)
m_ms = (ms_p + ms_t) / 2
ax[0,0].scatter(m_ms, diff, s=26, alpha=.7, edgecolors='k', linewidths=.3)
ax[0,0].axhline(bias, color='b', lw=1.2); ax[0,0].axhline(hi, color='r', ls='--'); ax[0,0].axhline(lo, color='r', ls='--')
ax[0,0].set_title(f"(a) Bland-Altman of MS\nbias={bias:.2f} dB, 95% LoA [{lo:.2f}, {hi:.2f}]")
ax[0,0].set_xlabel("Mean of predicted & true MS (dB)"); ax[0,0].set_ylabel("Predicted − True MS (dB)")

ax[0,1].hist(emae, bins=20, color='#4c72b0', edgecolor='k')
ax[0,1].axvline(emae.mean(), color='r', ls='--', label=f"mean={emae.mean():.2f} dB")
ax[0,1].set_title("(b) Per-eye pointwise MAE distribution"); ax[0,1].set_xlabel("Pointwise MAE (dB)"); ax[0,1].set_ylabel("# eyes"); ax[0,1].legend()

q1, q2 = np.percentile(ms_t, [33, 66])
grp = np.where(ms_t < q1, "severe", np.where(ms_t < q2, "moderate", "mild"))
gs = ["mild", "moderate", "severe"]; vals = [emae[grp == g].mean() for g in gs]; ns = [int((grp == g).sum()) for g in gs]
bars = ax[1,0].bar(gs, vals, color=['#55a868', '#dd8452', '#c44e52'])
for i, (v, nn) in enumerate(zip(vals, ns)): ax[1,0].text(i, v, f"{v:.2f}\n(n={nn})", ha='center', va='bottom', fontsize=9)
ax[1,0].set_title("(c) Pointwise MAE by severity (true-MS tertiles)"); ax[1,0].set_ylabel("Pointwise MAE (dB)")

ax[1,1].scatter(ms_t, ms_p, s=26, alpha=.7, edgecolors='k', linewidths=.3)
lim = [min(ms_t.min(), ms_p.min()) - 1, max(ms_t.max(), ms_p.max()) + 1]; ax[1,1].plot(lim, lim, 'k--', alpha=.5)
R2 = pearsonr(ms_t, ms_p)[0] ** 2
ax[1,1].set_title(f"(d) Predicted vs True MS (test n={len(T)})\nPearson R²={R2:.3f}")
ax[1,1].set_xlabel("True MS (dB)"); ax[1,1].set_ylabel("Predicted MS (dB)")
fig.suptitle(f"Internal clinical error analysis (R3-3) — best config: {cfg}", fontsize=13)
plt.savefig(OUT + "/fig_clinical_error.png", dpi=120, bbox_inches="tight"); plt.close()

# ============ Figure 2: 失败案例(最差MAE眼) ============
LAY = np.load(ROOT + "/vf_layout.npy"); KEEP = [i for i in range(61) if i not in (21, 32)]
Ng = 160; xs = np.linspace(-1, 1, Ng); ys = np.linspace(1, -1, Ng); GX, GY = np.meshgrid(xs, ys); DISC = GX**2 + GY**2 <= 1
def surf(v59):
    ok = v59 != -1
    z = griddata(LAY[ok], v59[ok], (GX, GY), method="cubic")
    zn = griddata(LAY[ok], v59[ok], (GX, GY), method="nearest")
    z[np.isnan(z)] = zn[np.isnan(z)]; return np.clip(z, 0, 33)
worst = np.argsort(-emae)[:6]
fig, axes = plt.subplots(3, 4, figsize=(13, 9.5))
for r_, idx in enumerate(worst):
    row, col = divmod(r_, 2)
    tv = T[idx][KEEP]; pv = P[idx][KEEP]
    fn = "  [high-risk FN: true severe, pred milder]" if (ms_t[idx] < q1 and ms_p[idx] > ms_t[idx] + 4) else ""
    for j, (v, lab) in enumerate([(tv, "True"), (pv, "Pred")]):
        a = axes[row, col*2 + j]
        im = a.imshow(np.where(DISC, surf(v), np.nan), cmap="jet", vmin=0, vmax=33); a.axis("off")
        ttl = f"#{idx} {lab} VF\ntrueMS={ms_t[idx]:.1f}, MAE={emae[idx]:.1f}" if j == 0 else f"#{idx} {lab} VF\npredMS={ms_p[idx]:.1f}{fn}"
        a.set_title(ttl, fontsize=8)
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.02, pad=.01, label="Sensitivity (dB)")
fig.suptitle("Failure cases: worst-MAE internal test eyes (True vs Predicted VF) — model floors for severe fields", fontsize=12)
plt.savefig(OUT + "/fig_failure_cases.png", dpi=115, bbox_inches="tight"); plt.close()

# ============ Figure 3: STARD / 数据流程图 ============
fig, ax = plt.subplots(figsize=(11.5, 8.5)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 12)
def box(x, y, w, h, t, fc="#eaf2fb"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec="k", lw=1.2))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=8.5)
def arr(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.1, color="k"))
box(3.0, 11.0, 4.0, 0.9, "GRAPE dataset (public, single center)", fc="#d6e4f0")
box(0.4, 9.3, 4.2, 1.0, "Baseline: 263 eyes\n(1 record/eye)")
box(5.4, 9.3, 4.2, 1.0, "Follow-up: 1115 records\n(longitudinal VF)")
arr(4.4, 11.0, 2.5, 10.3); arr(5.6, 11.0, 7.5, 10.3)
box(5.1, 7.7, 4.8, 1.0, "631 records WITH CFP\n= 631 image–VF pairs (regression)")
arr(7.5, 9.3, 7.5, 8.7)
box(5.1, 5.9, 4.8, 1.1, "Patient-level split (no leakage)\nTrain 410 / Val 89 / Test 132")
arr(7.5, 7.7, 7.5, 7.0)
box(5.1, 4.3, 4.8, 1.0, "+ StyleGAN synthetic\n(TRAIN only, never in test)", fc="#fdf0e6")
arr(6.3, 5.9, 6.3, 5.3)
box(0.4, 7.7, 4.2, 1.0, "PLR3 progression label\n168 / 42 / 53 (classification)")
arr(2.5, 9.3, 2.5, 8.7)
box(0.4, 5.9, 4.2, 1.0, "4 architectures × 3 inputs\n× 3 seeds + bootstrap CI", fc="#eaf2fb")
arr(2.5, 7.7, 2.5, 6.9)
box(2.6, 2.7, 4.8, 1.0, "External: Wannan AH 2025\n343 CFP / 119 VF (image-only, no numeric VF)", fc="#e9f3ea")
arr(5.0, 4.3, 5.0, 3.7)
box(0.3, 0.7, 4.5, 1.2, "Route A — MS proxy\n69 paired eyes\nSpearman 0.478 (p<0.001)", fc="#e9f3ea")
box(5.2, 0.7, 4.5, 1.2, "Route B — pseudocolor SSIM\n21 paired visits\nstructure null (p=0.44)", fc="#e9f3ea")
arr(4.2, 2.7, 2.6, 1.9); arr(5.8, 2.7, 7.4, 1.9)
ax.set_title("Data flow / STARD-style diagram (internal GRAPE + external Wannan)", fontsize=12)
plt.savefig(OUT + "/fig_stard_flow.png", dpi=120, bbox_inches="tight"); plt.close()

print("saved ->", OUT)
print("bias=%.2f LoA[%.2f,%.2f]  severity MAE:"%(bias,lo,hi), {g:round(v,2) for g,v in zip(gs,vals)})
print("worst eyes idx:", worst.tolist())
