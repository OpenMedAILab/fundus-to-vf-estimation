"""外部验证 Step B+C: 从灰度VF图反推 image-MS, 与预测MS患者级配对+相关"""
import os, glob, re, json, numpy as np
from PIL import Image
from scipy.stats import spearmanr, pearsonr
ROOT="/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/reproduce_full"
VF="/remote-home/guijiangsheng/yyy/yang/fix_paper/external/VF"

def key_of(name):  # 患者名+日期 (去掉尾部 -编号 和扩展名)
    b=re.sub(r'\.(jpg|jpeg|png)$','',name,flags=re.I)
    return re.sub(r'-\d+$','',b)

def image_ms(path):
    a=np.array(Image.open(path).convert("L")).astype(float); H,W=a.shape
    yy,xx=np.ogrid[:H,:W]; cy,cx=H/2,W/2
    disc=(xx-cx)**2+(yy-cy)**2 <= (0.42*W)**2      # 视野盘
    legend=xx< W*0.08                               # 左图例条
    inset=(xx>W*0.72)&(yy>H*0.78)                   # 右下盲点框
    white=a>248                                     # 纯白=背景/标注/>30
    mask=disc & ~legend & ~inset & ~white
    if mask.sum()<500: return None
    g=a[mask].mean()                                # 平均灰度
    return g/255.0*33.0                             # 线性映射 gray->dB (0..33)

# 反推所有外部VF图
img=[]
for p in sorted(glob.glob(VF+"/*.jpg")):
    ms=image_ms(p)
    if ms is not None: img.append({"key":key_of(os.path.basename(p)),"img_MS":ms})
# 患者级聚合 image-MS
ikey={}
for r in img: ikey.setdefault(r["key"],[]).append(r["img_MS"])
ikey={k:np.mean(v) for k,v in ikey.items()}

# 读预测MS, 患者级聚合
pred=json.load(open(f"{ROOT}/external_pred_ms.json"))
pkey={}
for r in pred:
    if "pred_MS" in r: pkey.setdefault(key_of(r["case"]),[]).append(r["pred_MS"])
pkey={k:np.mean(v) for k,v in pkey.items()}

# 内连接
common=sorted(set(ikey)&set(pkey))
P=np.array([pkey[k] for k in common]); I=np.array([ikey[k] for k in common])
print(f"VF图反推: {len(img)}张 -> {len(ikey)}患者visit | 预测MS: {len(pkey)}患者visit | 配对成功: {len(common)}")
if len(common)>=5:
    sr,sp=spearmanr(P,I); pr,pp=pearsonr(P,I)
    print(f"预测MS vs 图反推MS:  Spearman r={sr:.3f}(p={sp:.3g})  Pearson r={pr:.3f}(p={pp:.3g})")
    print(f"  预测MS: mean={P.mean():.2f} std={P.std():.2f} | 图MS: mean={I.mean():.2f} std={I.std():.2f}")
    # 线性拟合后 MAE
    a,b=np.polyfit(I,P,1); pred_from_img=a*I+b
    print(f"  线性校准后 预测MS vs (校准)图MS  MAE={np.abs(P-pred_from_img).mean():.2f} dB")
json.dump({"common":common,"pred_MS":P.tolist(),"img_MS":I.tolist()},open(f"{ROOT}/external_ms_pairing.json","w"),ensure_ascii=False,indent=1)
print("saved external_ms_pairing.json")
