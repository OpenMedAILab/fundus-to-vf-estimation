"""外部验证定稿: TOL=90配对, 出散点+Bland-Altman图, 存最终指标"""
import os, glob, re, json, numpy as np
from PIL import Image
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import RESULTS, FIGURES, EXT_VF as VF
TOL=90
def parse(s):
    n=re.match(r'^\D+',s); n=n.group(0) if n else s
    d=re.search(r'(\d{8})',s)
    try: dt=datetime.strptime(d.group(1),"%Y%m%d") if d else None
    except: dt=None
    return n,dt
def image_ms(path):
    a=np.array(Image.open(path).convert("L")).astype(float); H,W=a.shape
    yy,xx=np.ogrid[:H,:W]; cy,cx=H/2,W/2
    mask=((xx-cx)**2+(yy-cy)**2<=(0.42*W)**2)&(xx>=W*0.08)&~((xx>W*0.72)&(yy>H*0.78))&(a<=248)
    return a[mask].mean()/255.0*33.0 if mask.sum()>500 else None
pred=json.load(open(f"{RESULTS}/external_pred_ms.json"))
cfp=[(parse(r["case"])[0],parse(r["case"])[1],r["pred_MS"]) for r in pred if "pred_MS" in r and parse(r["case"])[1]]
vfs=[]
for p in sorted(glob.glob(VF+"/*.jpg")):
    n,d=parse(os.path.basename(p)); ms=image_ms(p)
    if d and ms is not None: vfs.append((n,d,ms))
pairs=[]
for n,d,ims in vfs:
    cand=[(abs((d-cd).days),pm) for cn,cd,pm in cfp if cn==n]
    if cand:
        gap,pm=min(cand,key=lambda x:x[0])
        if gap<=TOL: pairs.append((pm,ims))
P=np.array([x[0] for x in pairs]); I=np.array([x[1] for x in pairs])
sr,sp=spearmanr(P,I); pr,pp=pearsonr(P,I)
a,b=np.polyfit(I,P,1); Pcal=a*I+b; mae=np.abs(P-Pcal).mean(); r2=pr**2
res={"n_pairs":len(pairs),"spearman_r":float(sr),"spearman_p":float(sp),"pearson_r":float(pr),
     "pearson_p":float(pp),"R2":float(r2),"calibrated_MAE_dB":float(mae),
     "pred_MS_mean":float(P.mean()),"img_MS_mean":float(I.mean()),"tol_days":TOL,
     "note":"image-MS为灰度图代理(线性gray->dB,排除纯白);inference-only;同次就诊配对"}
json.dump(res,open(f"{RESULTS}/external_validation_final.json","w"),indent=2,ensure_ascii=False)
# 图
fig,ax=plt.subplots(1,2,figsize=(13,5.2))
ax[0].scatter(I,P,alpha=0.6,c="#1f77b4"); xs=np.array([I.min(),I.max()])
ax[0].plot(xs,a*xs+b,"r--",label=f"fit (Pearson r={pr:.2f}, R2={r2:.2f})")
ax[0].set_xlabel("Image-derived MS (from VF grayscale map)"); ax[0].set_ylabel("Predicted MS (from CFP)")
ax[0].set_title(f"External: Predicted vs Observed MS (n={len(pairs)}, Spearman={sr:.2f}, p={sp:.1e})"); ax[0].legend()
m=(Pcal+I)/2; diff=Pcal-I; bias=diff.mean(); sd=diff.std()
ax[1].scatter(m,diff,alpha=0.6,c="#2ca02c"); ax[1].axhline(bias,color="k"); ax[1].axhline(bias+1.96*sd,color="grey",ls="--"); ax[1].axhline(bias-1.96*sd,color="grey",ls="--")
ax[1].set_xlabel("Mean MS"); ax[1].set_ylabel("Diff (calibrated pred - image)"); ax[1].set_title(f"Bland-Altman (bias={bias:.2f}, LoA±{1.96*sd:.2f})")
plt.tight_layout(); plt.savefig(f"{FIGURES}/fig_external.png",dpi=120,bbox_inches="tight")
print(json.dumps(res,ensure_ascii=False,indent=1)); print("saved fig_external.png + external_validation_final.json")
