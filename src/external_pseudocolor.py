"""
外部"用预测结果生成伪彩图"验证 (改进版 SSIM)
=================================================
思路: CFP --模型--> 预测59点VF --> 渲染伪彩/灰度视野曲面;
      真实外部VF灰度"Level"图 --> 同一单位圆盘曲面;
      结构相似度对比 (matched vs mismatched)。

针对首版全局-SSIM 零结果(matched≈mismatched, p=0.47)的三处修复:
  1. 页面筛选: 每次随访有 30°/50° 多页, 仅选"30°中央页"(模型视野≈中央30°),
     排除 50°外周页(中央留白)与全黑退化页。
  2. OD/OS 朝向: 无眼别标签 -> SSIM 取 {原图, 水平镜像} 的较优(消除左右眼镜像约定)。
  3. 局部窗口 SSIM (skimage, gaussian, data_range=33) 仅在圆盘内平均,
     不再用单一全局窗口(只测均值/方差)。
  另加: 上下半野(sup/inf)dB差 的 matched 相关 —— 对 OD/OS 镜像不变, 临床上半野非对称是青光眼关键征。
全程 inference-only, 无微调, 无合成图。
"""
import os, glob, re, json, sys, numpy as np, torch
from datetime import datetime
from scipy.interpolate import griddata
from scipy.stats import spearmanr, mannwhitneyu
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from torchvision import transforms
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from config import SRC, CKPT, RESULTS, FIGURES, VF_LAYOUT, EXTERNAL_ROOT as EXT
sys.path.insert(0, SRC)
from repro import Hybrid, N_VF
dev="cuda" if torch.cuda.is_available() else "cpu"

# ---------- 视野点布局 (单位圆盘, +y=上方=上半野) ----------
lay=np.load(VF_LAYOUT)            # (59,2) 已归一到单位圆
order=np.lexsort((lay[:,0],-lay[:,1])); LP=lay[order]
KEEP=[i for i in range(61) if i not in (21,32)]   # 模型61维 -> 去2盲点 -> 59, 对齐 LP

N=256
xs=np.linspace(-1,1,N); ys=np.linspace(1,-1,N)    # row0=上(+y=superior)
GX,GY=np.meshgrid(xs,ys)
DISC=(GX**2+GY**2)<=1.0
DISC_ER=(GX**2+GY**2)<=0.85**2                     # 腐蚀盘(SSIM 平均, 避边界)
SUP=DISC&(GY>0.12); INF=DISC&(GY<-0.12)

def pred_surface(vf59):
    """59点预测 -> dB曲面(单位盘, 圆外填均值)"""
    z=griddata(LP,vf59,(GX,GY),method="cubic")
    zn=griddata(LP,vf59,(GX,GY),method="nearest")
    z[np.isnan(z)]=zn[np.isnan(z)]
    z=np.clip(z,0,33)
    out=z.copy(); out[~DISC]=z[DISC].mean()
    return out

def real_field(path):
    """真实灰度Level图 -> dB曲面(单位盘) + 中央数据占比(辨别30°/50°页) + 有效占比"""
    a=np.array(Image.open(path).convert("L")).astype(float); H,W=a.shape
    cy,cx=H/2.0,W/2.0; R=0.42*W
    Xpix=(cx+GX*R).astype(int); Ypix=(cy-GY*R).astype(int)
    inb=(Xpix>=0)&(Xpix<W)&(Ypix>=0)&(Ypix<H)
    g=np.full((N,N),255.0)
    g[inb]=a[Ypix[inb],Xpix[inb]]
    white=g>248                                    # 纯白=背景/坐标标注/>30dB(二义)->无数据
    db=np.where(white,np.nan,g/255.0*33.0)         # 灰度->dB(白=高敏, 黑=0=缺损)
    # 中央30%半径内"有数据"占比: 50°外周页中央留白 -> 低; 30°中央页 -> 高
    central=DISC&((GX**2+GY**2)<=0.30**2)
    cdata=float((~white & central).sum())/max(1,central.sum())
    valid=DISC&~white
    vfrac=float(valid.sum())/max(1,DISC.sum())
    # 圆盘内缺失(白)用最近邻填(注: 黑=0dB 属有效, 不填)
    out=db.copy()
    m=DISC&np.isnan(out)
    if m.any() and (DISC&~np.isnan(out)).any():
        vi=np.array(np.where(DISC&~np.isnan(out))).T; vv=out[DISC&~np.isnan(out)]
        hi=np.array(np.where(m)).T
        out[m]=griddata(vi,vv,hi,method="nearest")
    out[~DISC]=np.nanmean(out[DISC]) if DISC.any() else 0
    out=np.nan_to_num(out,nan=0.0)
    dbstd=float(np.nanstd(db[valid])) if valid.sum()>50 else 0.0
    return out,cdata,vfrac,dbstd

def disc_ssim(a,b):
    """局部窗口SSIM, 仅圆盘内平均; OD/OS未知 -> 取{原, 水平镜像}较优"""
    best=-1.0
    for bb in (b, b[:,::-1]):
        _,S=ssim(a,bb,data_range=33.0,gaussian_weights=True,sigma=1.5,
                 use_sample_covariance=False,full=True)
        v=float(S[DISC_ER].mean())
        if v>best: best=v
    return best

def hemi_asym(surf):
    return float(surf[SUP].mean()-surf[INF].mean())

def parse(s):
    n=re.match(r'^\D+',s); n=n.group(0) if n else s
    d=re.search(r'(\d{8})',s)
    try: dt=datetime.strptime(d.group(1),"%Y%m%d") if d else None
    except: dt=None
    return n,dt

# ---------- 模型 ----------
m=Hybrid(N_VF).to(dev)
m.load_state_dict(torch.load(f"{CKPT}/reg_hybrid_cfp_s0/best.pth",map_location=dev)); m.eval()
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# ---------- CFP 按(姓名,日期)聚合 -> 访次级预测VF ----------
cfp_imgs=sorted(glob.glob(EXT+"/CFP/*/*.JPG")+glob.glob(EXT+"/CFP/*/*.jpg"))
cfp_visits={}                                       # (name,date) -> [img paths]
for p in cfp_imgs:
    n,d=parse(os.path.basename(os.path.dirname(p)))
    if d is None: continue
    cfp_visits.setdefault((n,d),[]).append(p)
cfp_pred={}                                          # (name,date) -> (vf59, one img path)
with torch.no_grad():
    for k,paths in cfp_visits.items():
        vs=[]
        for p in paths:
            try: vs.append(m(tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev))[0].cpu().numpy()[KEEP])
            except Exception: pass
        if vs: cfp_pred[k]=(np.mean(vs,0), paths[0])
print(f"CFP访次: {len(cfp_visits)}, 成功预测: {len(cfp_pred)}")

# ---------- VF 按(姓名,日期)聚合 -> 选"30°中央页" ----------
vf_files=sorted(glob.glob(EXT+"/VF/*.jpg"))
vf_visits={}
for p in vf_files:
    n,d=parse(os.path.basename(p))
    if d is None: continue
    vf_visits.setdefault((n,d),[]).append(p)
vf_sel={}                                            # (name,date) -> (surf, path, cdata)
for k,paths in vf_visits.items():
    cand=[]
    for p in paths:
        try:
            surf,cdata,vfrac,dbstd=real_field(p)
            cand.append((cdata,dbstd,surf,p,vfrac))
        except Exception: pass
    if not cand: continue
    # 选中央数据占比最高(剔除50°外周页); 同分按dB结构(std)更大者(剔除全黑退化页)
    cand.sort(key=lambda c:(round(c[0],2),c[1]),reverse=True)
    cdata,dbstd,surf,p,vfrac=cand[0]
    if cdata<0.5: continue                            # 仍是外周/无中央数据 -> 跳过
    vf_sel[k]=(surf,p,cdata)
print(f"VF访次: {len(vf_visits)}, 选出30°中央页: {len(vf_sel)}")

# ---------- 同次就诊配对 (姓名 + 最近日期 <=90天) ----------
pairs=[]   # (cfp_key, vf_key, gap)
for (vn,vd) in vf_sel:
    cands=[(abs((vd-cd).days),(cn,cd)) for (cn,cd) in cfp_pred if cn==vn]
    if not cands: continue
    gap,ck=min(cands,key=lambda x:x[0])
    if gap<=90: pairs.append((ck,(vn,vd),gap))
print(f"配对访次(VF唯一): {len(pairs)}  (日期差中位 {int(np.median([g for *_,g in pairs])) if pairs else -1} 天)")

# ---------- 计算曲面/指标 ----------
P=[]; Rsurf=[]; meta=[]
for ck,vk,gap in pairs:
    psurf=pred_surface(cfp_pred[ck][0])
    rsurf=vf_sel[vk][1]; rsurf=vf_sel[vk][0]
    P.append(psurf); Rsurf.append(rsurf)
    meta.append({"cfp":cfp_pred[ck][1],"vf":vf_sel[vk][1],"gap":gap})
n=len(P); print("用于结构对比:",n)

matched=np.array([disc_ssim(P[i],Rsurf[i]) for i in range(n)])
rng=np.random.RandomState(0); mis=[]
for i in range(n):
    for _ in range(5):
        j=rng.randint(n)
        if j!=i: mis.append(disc_ssim(P[i],Rsurf[j]))
mis=np.array(mis)
u,pu=mannwhitneyu(matched,mis,alternative="greater")

# 上下半野不对称(OD/OS镜像不变) matched相关
pa=np.array([hemi_asym(P[i]) for i in range(n)])
ra=np.array([hemi_asym(Rsurf[i]) for i in range(n)])
sr_a,sp_a=spearmanr(pa,ra)
# 平均dB(空间MS)一致性(对照: 应≈MS代理)
pm=np.array([P[i][DISC].mean() for i in range(n)])
rm=np.array([Rsurf[i][DISC].mean() for i in range(n)])
sr_m,sp_m=spearmanr(pm,rm)

res={"n_struct_pairs":int(n),
     "matched_SSIM_mean":float(matched.mean()),"matched_SSIM_std":float(matched.std()),
     "mismatched_SSIM_mean":float(mis.mean()),"mismatched_SSIM_std":float(mis.std()),
     "mannwhitney_p_matched>mismatched":float(pu),
     "hemifield_asym_spearman_r":float(sr_a),"hemifield_asym_spearman_p":float(sp_a),
     "spatialMS_spearman_r":float(sr_m),"spatialMS_spearman_p":float(sp_m),
     "note":"30°中央页筛选; OD/OS取镜像较优; 局部窗口SSIM(skimage,圆盘内); VF访次级去重; inference-only"}
json.dump(res,open(f"{RESULTS}/external_pseudocolor.json","w"),indent=2,ensure_ascii=False)
print(json.dumps(res,ensure_ascii=False,indent=1))

# ============ 图1: 伪彩预测 + 真实对比 (代表性配对) ============
def draw_pred(ax,vf59,title):
    z=pred_surface(vf59); zz=np.where(DISC,z,np.nan)
    im=ax.imshow(zz,cmap="jet",vmin=0,vmax=33,extent=[-1,1,-1,1])
    ax.scatter(LP[:,0],LP[:,1],c=vf59,cmap="jet",vmin=0,vmax=33,s=42,edgecolors="k",linewidths=0.4)
    th=np.linspace(0,2*np.pi,200); ax.plot(np.cos(th),np.sin(th),"k",lw=0.8)
    ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title,fontsize=9); return im
def draw_real_db(ax,surf,title):
    zz=np.where(DISC,surf,np.nan)
    im=ax.imshow(zz,cmap="jet",vmin=0,vmax=33,extent=[-1,1,-1,1])
    th=np.linspace(0,2*np.pi,200); ax.plot(np.cos(th),np.sin(th),"k",lw=0.8)
    ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title,fontsize=9); return im

idx=np.argsort(-matched)                              # 由高到低
pick=list(idx[:3])+[idx[n//2]]+list(idx[-2:]) if n>=6 else list(range(n))
rows=len(pick)
fig,axes=plt.subplots(rows,4,figsize=(13,3.0*rows))
if rows==1: axes=axes[None,:]
for r,i in enumerate(pick):
    # 1 CFP
    try:
        cimg=Image.open(meta[i]["cfp"]).convert("RGB"); axes[r,0].imshow(cimg)
    except Exception: pass
    axes[r,0].axis("off"); axes[r,0].set_title(f"External CFP (pair#{i})",fontsize=9)
    # 2 预测伪彩
    draw_pred(axes[r,1],cfp_pred[pairs[i][0]][0],f"Predicted VF (pseudo-color, dB)\nspatial-MS={pm[i]:.1f}")
    # 3 真实灰度图
    try:
        rimg=Image.open(meta[i]["vf"]).convert("L"); axes[r,2].imshow(rimg,cmap="gray")
    except Exception: pass
    axes[r,2].axis("off"); axes[r,2].set_title("Real VF 'Level' (grayscale)",fontsize=9)
    # 4 真实dB曲面(同伪彩色标)
    im=draw_real_db(axes[r,3],Rsurf[i],f"Real VF -> dB surface\nSSIM={matched[i]:.2f}")
fig.colorbar(im,ax=axes.ravel().tolist(),fraction=0.012,pad=0.01,label="Sensitivity (dB)")
fig.suptitle(f"External pseudo-color VF: predicted-from-CFP vs real perimetry  (n={n} visits)",fontsize=12)
plt.savefig(f"{FIGURES}/fig_external_pseudocolor.png",dpi=115,bbox_inches="tight"); plt.close()

# ============ 图2: 一批外部CFP的预测伪彩图(展示生成器) ============
gkeys=list(cfp_pred)[:12]
fig,axes=plt.subplots(3,4,figsize=(12,9))
for ax,k in zip(axes.ravel(),gkeys):
    im=draw_pred(ax,cfp_pred[k][0],"")
for ax in axes.ravel()[len(gkeys):]: ax.axis("off")
fig.colorbar(im,ax=axes.ravel().tolist(),fraction=0.02,pad=0.01,label="Predicted sensitivity (dB)")
fig.suptitle("Predicted pseudo-color VF maps from external CFP (12 visits)",fontsize=12)
plt.savefig(f"{FIGURES}/fig_pred_pseudocolor_gallery.png",dpi=115,bbox_inches="tight"); plt.close()
print("saved: external_pseudocolor.json, fig_external_pseudocolor.png, fig_pred_pseudocolor_gallery.png")
