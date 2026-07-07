"""外部相似度验证: 预测VF渲染图 vs 真实视野图 SSIM, 匹配 vs 错配检验"""
import os, glob, re, json, sys, numpy as np, torch
from datetime import datetime
from scipy.interpolate import griddata
from PIL import Image
from torchvision import transforms
sys.path.insert(0,"/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/reproduce_full")
from repro import Hybrid, N_VF
ROOT="/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/reproduce_full"
EXT="/remote-home/guijiangsheng/yyy/yang/fix_paper/external"; dev="cuda" if torch.cuda.is_available() else "cpu"
lay=np.load(ROOT+"/vf_layout.npy"); order=np.lexsort((lay[:,0],-lay[:,1])); LP=lay[order]
gx,gy=np.mgrid[-1:1:200j,-1:1:200j]; disc=(gx**2+gy**2)<=1.0
def render(v):
    z=griddata(LP,v,(gx,gy),method="cubic"); zn=griddata(LP,v,(gx,gy),method="nearest"); z[np.isnan(z)]=zn[np.isnan(z)]; return z
def real_surface(path):  # 真实灰度图 -> dB曲面(200x200 disc)
    a=np.array(Image.open(path).convert("L")).astype(float); H,W=a.shape
    cy,cx=H/2,W/2; R=0.42*W
    out=np.full((200,200),np.nan)
    for i in range(200):
        for j in range(200):
            if not disc[i,j]: continue
            X=cx+gx[i,j]*R; Y=cy-gy[i,j]*R
            if 0<=int(Y)<H and 0<=int(X)<W:
                g=a[int(Y),int(X)]
                if g<=248: out[i,j]=g/255*33
    # 填洞
    m=np.isnan(out)
    if m.any() and (~m&disc).any():
        vi=np.array(np.where(~m&disc)).T; vv=out[~m&disc]; hi=np.array(np.where(m&disc)).T
        out[m&disc]=griddata(vi,vv,hi,method="nearest")
    return out
def gssim(a,b):
    a=a[disc]; b=b[disc]; ma,mb=np.nanmean(a),np.nanmean(b); va,vb=np.nanvar(a),np.nanvar(b)
    cov=np.nanmean((a-ma)*(b-mb)); L=35.0;c1=(0.01*L)**2;c2=(0.03*L)**2
    return ((2*ma*mb+c1)*(2*cov+c2))/((ma**2+mb**2+c1)*(va+vb+c2))
def parse(s):
    n=re.match(r'^\D+',s); n=n.group(0) if n else s; d=re.search(r'(\d{8})',s)
    try: dt=datetime.strptime(d.group(1),"%Y%m%d") if d else None
    except: dt=None
    return n,dt
# 配对 (姓名+最近日期<=90)
cfp_imgs=glob.glob(EXT+"/CFP/*/*.JPG")+glob.glob(EXT+"/CFP/*/*.jpg")
cfp=[(parse(os.path.basename(os.path.dirname(p)))[0],parse(os.path.basename(os.path.dirname(p)))[1],p) for p in cfp_imgs]
vfs=[(parse(os.path.basename(p))[0],parse(os.path.basename(p))[1],p) for p in glob.glob(EXT+"/VF/*.jpg")]
pairs=[]
for n,d,vp in vfs:
    if d is None: continue
    cand=[(abs((d-cd).days),cp) for cn,cd,cp in cfp if cn==n and cd]
    if cand:
        g,cp=min(cand); 
        if g<=90: pairs.append((cp,vp))
print("配对眼数:",len(pairs))
# 模型预测 -> 渲染图; 真实图 -> 曲面
m=Hybrid(N_VF).to(dev); m.load_state_dict(torch.load(ROOT+"/ckpt/reg_hybrid_cfp_s0/best.pth",map_location=dev)); m.eval()
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
keep=[i for i in range(61) if i not in(21,32)]
pmaps=[]; rmaps=[]
with torch.no_grad():
    for cp,vp in pairs:
        try:
            vf=m(tf(Image.open(cp).convert("RGB")).unsqueeze(0).to(dev))[0].cpu().numpy()[keep]
            pmaps.append(render(vf)); rmaps.append(real_surface(vp))
        except Exception as e: pass
n=len(pmaps); print("成功渲染:",n)
matched=np.array([gssim(pmaps[i],rmaps[i]) for i in range(n)])
rng=np.random.RandomState(0); mis=[]
for i in range(n):
    for _ in range(5):
        j=rng.randint(n)
        if j!=i: mis.append(gssim(pmaps[i],rmaps[j]))
mis=np.array(mis)
from scipy.stats import mannwhitneyu
u,pu=mannwhitneyu(matched,mis,alternative="greater")
res={"n_pairs":n,"matched_SSIM_mean":float(matched.mean()),"matched_SSIM_std":float(matched.std()),
     "mismatched_SSIM_mean":float(mis.mean()),"mannwhitney_p_matched>mismatched":float(pu)}
json.dump(res,open(ROOT+"/external_ssim.json","w"),indent=2)
print(json.dumps(res,indent=1))
