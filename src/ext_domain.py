"""外部域偏移: 外部CFP vs GRAPE CFP 的 FID/KID (R4-8 domain shift)"""
import glob, json, os, torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from config import RESULTS, GRAPE_CFP as GRAPE, EXT_CFP_IMGS as EXT
dev="cuda" if torch.cuda.is_available() else "cpu"
tf=transforms.Compose([transforms.Resize((256,256)),transforms.PILToTensor()])
def load(paths):
    out=[]
    for p in paths:
        try: out.append(tf(Image.open(p).convert("RGB")))
        except: pass
    return torch.stack(out)
g=load(sorted(glob.glob(GRAPE+"/*.jpg")))
e=load(sorted(glob.glob(EXT+"/*.jpg"))+sorted(glob.glob(EXT+"/*.JPG")))
print("GRAPE",g.shape,"External",e.shape,flush=True)
fid=FrechetInceptionDistance(feature=2048,normalize=False).to(dev)
kid=KernelInceptionDistance(subset_size=100,normalize=False).to(dev)
for m in (fid,kid):
    for i in range(0,len(g),64): m.update(g[i:i+64].to(dev),real=True)
    for i in range(0,len(e),64): m.update(e[i:i+64].to(dev),real=False)
res={"FID_external_vs_GRAPE":float(fid.compute()),
     "KID_external_vs_GRAPE_mean":float(kid.compute()[0]),"KID_std":float(kid.compute()[1]),
     "n_GRAPE":len(g),"n_external":len(e)}
json.dump(res,open(f"{RESULTS}/ext_domain_shift.json","w"),indent=2)
print(json.dumps(res),flush=True)
