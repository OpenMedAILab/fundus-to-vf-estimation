"""Phase 3 指标: FID + KID (torchmetrics), real GRAPE CFP vs StyleGAN生成图"""
import os, glob, json, numpy as np, torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

REAL="/remote-home/guijiangsheng/yyy/yang/fix_paper/20251113/GRAPE Dataset/GRAPE Dataset/CFPs"
GEN="/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/cloud_upload_package/experiments/generated"
OUT="/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/reproduce_full/fid_kid.json"
dev="cuda" if torch.cuda.is_available() else "cpu"
tf=transforms.Compose([transforms.Resize((256,256)), transforms.PILToTensor()])  # uint8

def load(paths):
    out=[]
    for p in paths:
        try: out.append(tf(Image.open(p).convert("RGB")))
        except: pass
    return torch.stack(out)

real=load(sorted(glob.glob(REAL+"/*.jpg")))
fake=load(sorted(glob.glob(GEN+"/seed*.png")))
print("real",real.shape,"fake",fake.shape, flush=True)

fid=FrechetInceptionDistance(feature=2048, normalize=False).to(dev)
kid=KernelInceptionDistance(subset_size=100, normalize=False).to(dev)
def feed(metric, imgs, real_flag):
    for i in range(0,len(imgs),64):
        metric.update(imgs[i:i+64].to(dev), real=real_flag)
for m in (fid,kid):
    feed(m, real, True); feed(m, fake, False)
fid_v=float(fid.compute())
kid_m,kid_s=kid.compute()
res={"FID":fid_v,"KID_mean":float(kid_m),"KID_std":float(kid_s),"n_real":len(real),"n_fake":len(fake)}
json.dump(res, open(OUT,"w"), indent=2)
print(json.dumps(res), flush=True)
