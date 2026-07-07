"""外部配对 v2: 按姓名+最近日期匹配 VF图<->CFP, 重算 预测MS vs 图反推MS"""
import os, glob, re, json, numpy as np
from PIL import Image
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
from config import ROOT, EXT_VF as VF

def parse(s):
    name=re.match(r'^\D+',s); name=name.group(0) if name else s
    d=re.search(r'(\d{8})',s); 
    try: date=datetime.strptime(d.group(1),"%Y%m%d") if d else None
    except: date=None
    return name, date

def image_ms(path):
    a=np.array(Image.open(path).convert("L")).astype(float); H,W=a.shape
    yy,xx=np.ogrid[:H,:W]; cy,cx=H/2,W/2
    disc=(xx-cx)**2+(yy-cy)**2<=(0.42*W)**2; legend=xx<W*0.08
    inset=(xx>W*0.72)&(yy>H*0.78); white=a>248
    mask=disc&~legend&~inset&~white
    return a[mask].mean()/255.0*33.0 if mask.sum()>500 else None

# 预测MS: 每张CFP -> (name,date,pred_MS)
pred=json.load(open(f"{ROOT}/external_pred_ms.json"))
cfp=[]
for r in pred:
    if "pred_MS" not in r: continue
    n,d=parse(r["case"]); 
    if d: cfp.append((n,d,r["pred_MS"]))

# VF图 -> (name,date,img_MS)
vfs=[]
for p in sorted(glob.glob(VF+"/*.jpg")):
    n,d=parse(os.path.basename(p)); ms=image_ms(p)
    if d and ms is not None: vfs.append((n,d,ms))

# 每张VF匹配同名最近日期CFP
for TOL in [7,30,90,99999]:
    pairs=[]
    for n,d,ims in vfs:
        cand=[(abs((d-cd).days),pm) for cn,cd,pm in cfp if cn==n]
        if not cand: continue
        gap,pm=min(cand,key=lambda x:x[0])
        if gap<=TOL: pairs.append((pm,ims,gap))
    if len(pairs)<5: 
        print(f"TOL={TOL}天: 配对{len(pairs)} (太少)"); continue
    P=np.array([x[0] for x in pairs]); I=np.array([x[1] for x in pairs])
    sr,sp=spearmanr(P,I); pr,pp=pearsonr(P,I)
    a,b=np.polyfit(I,P,1); mae=np.abs(P-(a*I+b)).mean()
    gaps=[x[2] for x in pairs]
    print(f"TOL={TOL:5d}天: 配对{len(pairs)} | Spearman={sr:.3f}(p={sp:.3g}) Pearson={pr:.3f} | 校准后MAE={mae:.2f}dB | 日期差中位{int(np.median(gaps))}天")
