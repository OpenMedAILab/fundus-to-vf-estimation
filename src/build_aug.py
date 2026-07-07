"""Phase 3 数据: 复用已生成的合成CFP, 构建增强训练索引 (合成图只进训练集, 泄漏安全)
- 分类: 合成图标 PLR3=1 (少数类)
- 回归: 合成图 VF 继承自训练集中最严重(低MS)真实眼 (label inheritance, 文档化近似)
"""
import json, os, glob, random, numpy as np, shutil
random.seed(42); np.random.seed(42)
from config import ROOT, GEN_DIR as GEN
SYN=ROOT+"/synthetic"; os.makedirs(SYN, exist_ok=True)
DATA=ROOT+"/data"
N_SYN=150

# 合成图: 用 seed*.png
gens=sorted(glob.glob(GEN+"/seed*.png"))[:N_SYN]
print("可用合成图:",len(glob.glob(GEN+"/seed*.png")),"使用:",len(gens))
syn_names=[]
for i,g in enumerate(gens):
    name=f"syn_{i:04d}.png"; dst=os.path.join(SYN,name)
    if not os.path.exists(dst): shutil.copy(g,dst)
    syn_names.append(name)

# ---- 分类增强 ----
cls=json.load(open(f"{DATA}/cls_train.json"))
syn_cls=[{"subject":-1,"eye":"SYN","cfp_filename":n,"plr3":1,"plr2":1,"md":None,"is_synthetic":True} for n in syn_names]
json.dump(cls+syn_cls, open(f"{DATA}/cls_train_aug.json","w"), indent=1)
print(f"cls_train_aug: {len(cls)}+{len(syn_cls)} = {len(cls)+len(syn_cls)} (PLR3阳: {sum(1 for x in cls if x['plr3']==1)}+{len(syn_cls)})")

# ---- 回归增强: VF 继承自最严重真实眼 ----
reg=json.load(open(f"{DATA}/reg_train.json"))
def ms(vf): 
    v=np.array([x for x in vf if x is not None and x!=-1]); return v.mean() if len(v) else 99
reg_sorted=sorted(reg, key=lambda r: ms(r["vf"]))  # 低MS=严重
severe=reg_sorted[:max(20,len(reg)//5)]  # 最严重20%
syn_reg=[]
for n in syn_names:
    src=random.choice(severe)
    syn_reg.append({"subject":-1,"eye":"SYN","visit":None,"interval_years":None,
                    "cfp_filename":n,"vf":src["vf"],"plr3":1,"plr2":1,"md":None,"is_synthetic":True})
json.dump(reg+syn_reg, open(f"{DATA}/reg_train_aug.json","w"), indent=1)
print(f"reg_train_aug: {len(reg)}+{len(syn_reg)} = {len(reg)+len(syn_reg)} (VF继承自最严重{len(severe)}眼)")
print("synthetic dir:", len(os.listdir(SYN)), "张")
