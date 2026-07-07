"""Wave3: Grad-CAM 热图 (结构-功能, R: Figure6) — ResNet-ROI回归模型对预测MS的关注区域"""
import json, os, numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
from config import setup_cjk_font; setup_cjk_font(); import matplotlib.pyplot as plt
from PIL import Image; from torchvision import transforms
import sys, config; sys.path.insert(0, config.SRC)
from repro import ResNetOnly, N_VF, GRAPE, BASE
from config import CKPT, DATA, FIGURES
dev="cuda" if torch.cuda.is_available() else "cpu"
ck=f"{CKPT}/reg_resnet_roi_s0/best.pth"
m=ResNetOnly(N_VF).to(dev); m.load_state_dict(torch.load(ck,map_location=dev)); m.eval()
layer4=m.b[7]   # 最后卷积块
feat={}; grad={}
layer4.register_forward_hook(lambda mod,i,o: feat.__setitem__("v",o))
layer4.register_full_backward_hook(lambda mod,gi,go: grad.__setitem__("v",go[0]))
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
recs=json.load(open(f"{DATA}/reg_test.json"))[:6]
fig,axs=plt.subplots(2,6,figsize=(20,7))
for j,r in enumerate(recs):
    p=os.path.join(BASE,"ROI images",r["cfp_filename"])
    img=Image.open(p).convert("RGB").resize((224,224)); x=tf(img).unsqueeze(0).to(dev)
    m.zero_grad(); out=m(x); out.mean().backward()   # 对预测MS
    fmap=feat["v"][0]; g=grad["v"][0]; w=g.mean((1,2))
    cam=F.relu((w[:,None,None]*fmap).sum(0)); cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
    cam=F.interpolate(cam[None,None],size=(224,224),mode="bilinear")[0,0].cpu().detach().numpy()
    axs[0,j].imshow(img); axs[0,j].set_title(r["cfp_filename"],fontsize=8); axs[0,j].axis("off")
    axs[1,j].imshow(img); axs[1,j].imshow(cam,cmap="jet",alpha=0.5); axs[1,j].axis("off")
axs[0,0].set_ylabel("ROI"); axs[1,0].set_ylabel("Grad-CAM")
plt.suptitle("Grad-CAM: ResNet-ROI VF estimation — structure-function attention",fontsize=12)
plt.tight_layout(); plt.savefig(f"{FIGURES}/fig_gradcam.png",dpi=120,bbox_inches="tight")
print("saved fig_gradcam.png")
