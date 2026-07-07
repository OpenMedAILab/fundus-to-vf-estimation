"""
皖南(Wannan)外部验证 — 独立新实现 (2026-06, 不复用既有 external_*.py)
================================================================
CFP --Hybrid--> 59点VF(dB) --> 伪彩 sensitivity 曲面
真实 "Level" 灰度图 --凡例LUT较准--> dB 曲面 (单位=27°圆盘)
对比:
  (1) 严重度 MS 一致性  : 预测盘内均dB vs 真实盘内均dB (Spearman/Pearson)
  (2) 结构 SSIM         : matched vs mismatched (镜像不变, 局部窗口, 盘内)
  (3) 59点逐点          : 每眼 pred59 vs real59 的 Spearman / MAE
  (4) 上下半野非对称    : (sup-inf)dB pred vs real (对 OD/OS 镜像不变)
全程 inference-only, 无微调, 无合成图。
凡例较准(实测): gray->dB 非线性; 黑(0)=X=绝对暗点≈0dB; 0dB=gray96; >30=gray254; 白格线/标注=255.
几何(实测,934x934): 圆心=(W/2,H/2); 30°边界=0.41W(穿过"30"刻度框); 单位27°=0.37W.
"""
import os, glob, re, json, sys
import numpy as np, torch
from datetime import datetime
from PIL import Image
from scipy.interpolate import griddata
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from scipy.ndimage import binary_dilation
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/reproduce_full"
EXT  = "/remote-home/guijiangsheng/yyy/yang/fix_paper/external"
OUT  = ROOT + "/wannan_extval"; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)
from repro import Hybrid, N_VF
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 凡例较准 (左侧色标实测) ----------
LEG_GRAY = np.array([0, 96, 117, 141, 172, 193, 210, 220, 228, 236, 242, 248, 254], float)
LEG_DB   = np.array([0,  0,   3,   6,   9,  12,  15,  18,  21,  24,  27,  30,  33], float)
def gray2db(g): return np.interp(g, LEG_GRAY, LEG_DB)

# ---------- 正准单位圆盘 (单位=27°), row0=上=上半野 ----------
N = 200
xs = np.linspace(-1, 1, N); ys = np.linspace(1, -1, N)
GX, GY = np.meshgrid(xs, ys); RR = GX**2 + GY**2
DISC = RR <= 1.0
DISC_ER = RR <= 0.90**2          # 腐蚀盘(SSIM/std 平均, 避边界)
CEN = RR <= 0.30**2              # 中央(辨别 30°/50° 页)
SUP = DISC & (GY > 0.10); INF = DISC & (GY < -0.10)

# ---------- VF 点布局 (59点, 单位≈27°) ----------
LAY = np.load(ROOT + "/vf_layout.npy")            # (59,2)
KEEP = [i for i in range(61) if i not in (21, 32)]
# 59点在正准网格中的(row,col)
LcolF = (LAY[:, 0] + 1) / 2 * (N - 1)
LrowF = (1 - LAY[:, 1]) / 2 * (N - 1)
Lcol = np.clip(LcolF.astype(int), 0, N - 1); Lrow = np.clip(LrowF.astype(int), 0, N - 1)

CX_FR, CY_FR = 0.5, 0.5          # 圆心(图像比例)
R30_FR, RUNIT_FR = 0.41, 0.37    # 30°边界 / 27°单位

def extract_real(path):
    """真实Level灰度图 -> (dB曲面[单位27°盘], cdata中央有数据占比, struct结构std)"""
    a = np.array(Image.open(path).convert("L")).astype(float); H, W = a.shape
    cx, cy = CX_FR * W, CY_FR * H; Runit = RUNIT_FR * W
    px = cx + GX * Runit; py = cy - GY * Runit
    ix = np.clip(px.astype(int), 0, W - 1); iy = np.clip(py.astype(int), 0, H - 1)
    g = a[iy, ix]
    white = a >= 253                                  # 白格线/标注框
    whiteD = binary_dilation(white, iterations=4)     # 膨胀吞掉框内黑字
    gnod = whiteD[iy, ix]                             # 无数据(网格采样)
    db = gray2db(g); db[gnod] = np.nan
    m = DISC & np.isnan(db); ok = DISC & ~np.isnan(db)
    if m.any() and ok.any():
        vi = np.array(np.where(ok)).T; vv = db[ok]; hi = np.array(np.where(m)).T
        db[m] = griddata(vi, vv, hi, method="nearest")
    db = np.clip(np.nan_to_num(db, nan=0.0), 0, 33)
    db[~DISC] = db[DISC].mean()
    cdata = float((CEN & ~gnod).sum()) / max(1, CEN.sum())
    struct = float(db[DISC_ER].std())
    return db, cdata, struct

def pred_surface(vf59):
    z = griddata(LAY, vf59, (GX, GY), method="cubic")
    zn = griddata(LAY, vf59, (GX, GY), method="nearest")
    z[np.isnan(z)] = zn[np.isnan(z)]; z = np.clip(z, 0, 33)
    z[~DISC] = z[DISC].mean(); return z

def disc_ssim(pred, real):
    best = -1.0
    for rr in (real, real[:, ::-1]):                  # OD/OS 镜像不变 -> 取较优
        _, S = ssim(pred, rr, data_range=33.0, gaussian_weights=True, sigma=1.5,
                    use_sample_covariance=False, full=True)
        best = max(best, float(S[DISC_ER].mean()))
    return best

def hemi(surf): return float(surf[SUP].mean() - surf[INF].mean())

def parse(s):
    n = re.match(r'^\D+', s); n = n.group(0) if n else s
    d = re.search(r'(\d{8})', s)
    try: dt = datetime.strptime(d.group(1), "%Y%m%d") if d else None
    except Exception: dt = None
    return n, dt

# ---------- 模型 ----------
m = Hybrid(N_VF).to(dev)
m.load_state_dict(torch.load(ROOT + "/ckpt/reg_hybrid_cfp_s0/best.pth", map_location=dev)); m.eval()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# ---------- CFP 按(姓名,日期)聚合 -> 访次级预测VF ----------
cfp_imgs = sorted(glob.glob(EXT + "/CFP/*/*.JPG") + glob.glob(EXT + "/CFP/*/*.jpg"))
cfp_visits = {}
for p in cfp_imgs:
    n, d = parse(os.path.basename(os.path.dirname(p)))
    if d is None: continue
    cfp_visits.setdefault((n, d), []).append(p)
cfp_pred = {}
with torch.no_grad():
    for k, paths in cfp_visits.items():
        vs = []
        for p in paths:
            try: vs.append(m(tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev))[0].cpu().numpy()[KEEP])
            except Exception: pass
        if vs: cfp_pred[k] = (np.clip(np.mean(vs, 0), 0, 33), paths[0])
print(f"[CFP] 访次={len(cfp_visits)} 成功预测={len(cfp_pred)}")

# ---------- VF 按(姓名,日期)聚合 -> 选 30° 中央页 ----------
vf_files = sorted(glob.glob(EXT + "/VF/*.jpg") + glob.glob(EXT + "/VF/*.JPG"))
vf_visits = {}
for p in vf_files:
    n, d = parse(os.path.basename(p))
    if d is None: continue
    vf_visits.setdefault((n, d), []).append(p)
vf_sel = {}
for k, paths in vf_visits.items():
    cand = []
    for p in paths:
        try:
            db, cdata, struct = extract_real(p)
            cand.append((cdata, struct, db, p))
        except Exception as e:
            print("  extract fail", os.path.basename(p), e)
    cand = [c for c in cand if c[0] > 0.5 and c[1] > 1.5]   # 排除50°外周页(cdata低)+全黑/空白页(struct低)
    if not cand: continue
    cand.sort(key=lambda c: c[1], reverse=True)            # 取结构最丰富的中央页
    cdata, struct, db, p = cand[0]
    vf_sel[k] = (db, p, cdata, struct)
print(f"[VF] 访次={len(vf_visits)} 选出30°中央页={len(vf_sel)}")

# ---------- 同次配对 (姓名 + 最近日期 ≤90天) ----------
pairs = []
for (vn, vd) in vf_sel:
    cc = [(abs((vd - cd).days), (cn, cd)) for (cn, cd) in cfp_pred if cn == vn]
    if not cc: continue
    gap, ck = min(cc, key=lambda x: x[0])
    if gap <= 90: pairs.append((ck, (vn, vd), gap))
print(f"[PAIR] 配对访次={len(pairs)} 日期差中位={int(np.median([g for *_,g in pairs])) if pairs else -1}天")

# ---------- 指标 ----------
rows = []
for ck, vk, gap in pairs:
    vf59 = cfp_pred[ck][0]
    psurf = pred_surface(vf59)
    rdb = vf_sel[vk][0]
    real59 = rdb[Lrow, Lcol]
    # 逐点 spearman (镜像不变: 取 {x, -x} 较优)
    real59_mir = rdb[Lrow, (N - 1) - Lcol]
    sp1 = spearmanr(vf59, real59).correlation
    sp2 = spearmanr(vf59, real59_mir).correlation
    pt_sp = np.nanmax([sp1, sp2])
    pt_mae = float(np.mean(np.abs(vf59 - (real59 if sp1 >= sp2 else real59_mir))))
    rows.append(dict(
        name=vk[0], date=vk[1].strftime("%Y%m%d"), gap=gap,
        cfp=cfp_pred[ck][1], vf=vf_sel[vk][1],
        ssim=disc_ssim(psurf, rdb),
        pred_MS=float(vf59.mean()), real_MS=float(real59.mean()),          # 临床MS = 59测点均(主)
        pred_MS_disc=float(psurf[DISC].mean()), real_MS_disc=float(rdb[DISC].mean()),  # 盘面均(对照)
        hemi_pred=hemi(psurf), hemi_real=hemi(rdb),
        pt_spearman=float(pt_sp), pt_mae=pt_mae,
        _psurf=psurf, _rdb=rdb, _vf59=vf59))
n = len(rows); print("[METRIC] 用于对比 n =", n)

matched = np.array([r["ssim"] for r in rows])
rng = np.random.RandomState(0); mis = []
for i in range(n):
    for _ in range(5):
        j = rng.randint(n)
        if j != i: mis.append(disc_ssim(rows[i]["_psurf"], rows[j]["_rdb"]))
mis = np.array(mis)
u, p_ssim = mannwhitneyu(matched, mis, alternative="greater") if n > 2 else (np.nan, np.nan)

pMS = np.array([r["pred_MS"] for r in rows]); rMS = np.array([r["real_MS"] for r in rows])
sr_ms, sp_ms = spearmanr(pMS, rMS); pr_ms, pp_ms = pearsonr(pMS, rMS)
ms_mae = float(np.mean(np.abs(pMS - rMS)))
pMSd = np.array([r["pred_MS_disc"] for r in rows]); rMSd = np.array([r["real_MS_disc"] for r in rows])
sr_msd, sp_msd = spearmanr(pMSd, rMSd)
# 校准后(线性回归去系统偏差) MS-MAE
A = np.vstack([pMS, np.ones_like(pMS)]).T
coef, *_ = np.linalg.lstsq(A, rMS, rcond=None); ms_mae_cal = float(np.mean(np.abs(A @ coef - rMS)))

hp = np.array([r["hemi_pred"] for r in rows]); hr = np.array([r["hemi_real"] for r in rows])
sr_h, sp_h = spearmanr(hp, hr)
pt_sps = np.array([r["pt_spearman"] for r in rows])

res = dict(
    n_pairs=n, median_gap_days=int(np.median([r["gap"] for r in rows])) if n else -1,
    severity_MS=dict(spearman_r=float(sr_ms), spearman_p=float(sp_ms),
                     pearson_r=float(pr_ms), pearson_p=float(pp_ms),
                     R2=float(pr_ms**2), MS_MAE=ms_mae, MS_MAE_calibrated=ms_mae_cal,
                     spearman_r_discmean=float(sr_msd), spearman_p_discmean=float(sp_msd)),
    structure_SSIM=dict(matched_mean=float(matched.mean()), matched_std=float(matched.std()),
                        mismatched_mean=float(mis.mean()), mismatched_std=float(mis.std()),
                        mannwhitney_p_matched_gt_mismatched=float(p_ssim)),
    pointwise=dict(per_eye_spearman_mean=float(np.nanmean(pt_sps)),
                   per_eye_spearman_median=float(np.nanmedian(pt_sps)),
                   frac_eyes_p_pos=float(np.mean(pt_sps > 0))),
    hemifield_asym=dict(spearman_r=float(sr_h), spearman_p=float(sp_h)),
    note="独立新实现; 凡例LUT较准; grid/box掩膜; 27°正准盘; OD/OS镜像不变; inference-only")
json.dump(res, open(OUT + "/wannan_extval.json", "w"), indent=2, ensure_ascii=False)
json.dump([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
          open(OUT + "/wannan_extval_pairs.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(res, ensure_ascii=False, indent=1))

# ============ 图0(调试): 59点叠加 + 提取曲面, 验证几何/较准 ============
dbg = rows[int(np.argmax(matched))]
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
rimg = np.array(Image.open(dbg["vf"]).convert("L"))
ax[0].imshow(rimg, cmap="gray"); ax[0].set_title("real grayscale + 59pts"); ax[0].axis("off")
H0, W0 = rimg.shape
ax[0].scatter(CX_FR*W0 + LAY[:,0]*RUNIT_FR*W0, CY_FR*H0 - LAY[:,1]*RUNIT_FR*H0, s=10, c="r")
ax[1].imshow(np.where(DISC, dbg["_rdb"], np.nan), cmap="jet", vmin=0, vmax=33); ax[1].set_title("real -> dB surface"); ax[1].axis("off")
ax[2].imshow(np.where(DISC, dbg["_psurf"], np.nan), cmap="jet", vmin=0, vmax=33); ax[2].set_title("predicted dB surface"); ax[2].axis("off")
plt.savefig(OUT + "/fig_debug_extract.png", dpi=110, bbox_inches="tight"); plt.close()

# ============ 图1: MS 散点 ============
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(pMS, rMS, s=40, edgecolors="k")
lim = [min(pMS.min(), rMS.min()) - 1, max(pMS.max(), rMS.max()) + 1]
ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("Predicted MS from CFP (dB)"); ax.set_ylabel("Real VF MS (dB)")
ax.set_title(f"Severity transfer (n={n})\nSpearman r={sr_ms:.3f} p={sp_ms:.1e}  R2={pr_ms**2:.3f}")
plt.savefig(OUT + "/fig_ms_scatter.png", dpi=120, bbox_inches="tight"); plt.close()

# ============ 图2: 伪彩预测 gallery ============
gk = list(cfp_pred)[:12]
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for ax, k in zip(axes.ravel(), gk):
    z = pred_surface(cfp_pred[k][0])
    im = ax.imshow(np.where(DISC, z, np.nan), cmap="jet", vmin=0, vmax=33)
    ax.scatter((LAY[:,0]+1)/2*(N-1), (1-LAY[:,1])/2*(N-1), c=cfp_pred[k][0], cmap="jet", vmin=0, vmax=33, s=14, edgecolors="k", linewidths=0.3)
    ax.axis("off")
for ax in axes.ravel()[len(gk):]: ax.axis("off")
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label="Predicted sensitivity (dB)")
fig.suptitle("Predicted pseudo-color VF from external CFP (12 visits)", fontsize=12)
plt.savefig(OUT + "/fig_pred_gallery.png", dpi=110, bbox_inches="tight"); plt.close()

# ============ 图3: 配对三联(CFP|预测|真实灰度|真实dB), 按SSIM高->低 ============
idx = np.argsort(-matched)
pick = list(idx[:3]) + [idx[n//2]] + list(idx[-2:]) if n >= 6 else list(idx)
fig, axes = plt.subplots(len(pick), 4, figsize=(13, 3.0*len(pick)))
if len(pick) == 1: axes = axes[None, :]
for r, i in enumerate(pick):
    R = rows[i]
    try: axes[r,0].imshow(Image.open(R["cfp"]).convert("RGB"))
    except Exception: pass
    axes[r,0].axis("off"); axes[r,0].set_title(f"CFP ({R['name']})", fontsize=9)
    z = R["_psurf"]; axes[r,1].imshow(np.where(DISC, z, np.nan), cmap="jet", vmin=0, vmax=33)
    axes[r,1].axis("off"); axes[r,1].set_title(f"Predicted (MS={R['pred_MS']:.1f})", fontsize=9)
    try: axes[r,2].imshow(Image.open(R["vf"]).convert("L"), cmap="gray")
    except Exception: pass
    axes[r,2].axis("off"); axes[r,2].set_title("Real VF 'Level'", fontsize=9)
    im = axes[r,3].imshow(np.where(DISC, R["_rdb"], np.nan), cmap="jet", vmin=0, vmax=33)
    axes[r,3].axis("off"); axes[r,3].set_title(f"Real->dB (MS={R['real_MS']:.1f}, SSIM={R['ssim']:.2f})", fontsize=9)
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01, label="Sensitivity (dB)")
fig.suptitle(f"External: predicted-from-CFP vs real perimetry (n={n})", fontsize=12)
plt.savefig(OUT + "/fig_pairs.png", dpi=115, bbox_inches="tight"); plt.close()
print("saved ->", OUT)
