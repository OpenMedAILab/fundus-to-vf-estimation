"""外部验证 Step A: 训好的GRAPE模型在外部CFP上推理 -> 预测59点VF + 预测MS"""
import os, glob, json, sys, numpy as np, torch
sys.path.insert(0,"/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/reproduce_full")
from repro import Hybrid, ResNetOnly, N_VF
from PIL import Image
from torchvision import transforms
ROOT="/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/reproduce_full"
EXT="/remote-home/guijiangsheng/yyy/yang/fix_paper/external/CFP"
dev="cuda" if torch.cuda.is_available() else "cpu"
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
# 用 hybrid_cfp (点位MAE最优) 的 seed0 权重
m=Hybrid(N_VF).to(dev); m.load_state_dict(torch.load(f"{ROOT}/ckpt/reg_hybrid_cfp_s0/best.pth",map_location=dev)); m.eval()
out=[]
cfps=sorted(glob.glob(EXT+"/*/*.JPG")+glob.glob(EXT+"/*/*.jpg"))
with torch.no_grad():
    for p in cfps:
        case=os.path.basename(os.path.dirname(p))   # 患者名+日期+编号
        try:
            x=tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            vf=m(x)[0].cpu().numpy()
            ms=float(np.mean(vf))   # 59维全有效(模型输出无-1)
            out.append({"case":case,"file":os.path.basename(p),"pred_MS":ms})
        except Exception as e:
            out.append({"case":case,"file":os.path.basename(p),"error":str(e)})
pred_ms=[o["pred_MS"] for o in out if "pred_MS" in o]
json.dump(out,open(f"{ROOT}/external_pred_ms.json","w"),indent=1,ensure_ascii=False)
print(f"外部CFP推理完成: {len(out)} 张, 成功 {len(pred_ms)}")
print(f"预测MS: mean={np.mean(pred_ms):.2f} std={np.std(pred_ms):.2f} min={np.min(pred_ms):.2f} max={np.max(pred_ms):.2f}")
# 对比GRAPE测试集预测MS分布
g=[json.load(open(f)) for f in glob.glob(f"{ROOT}/ckpt/reg_hybrid_cfp_s0/result.json")][0]
P=np.array(g["test_pred"]["P"]); T=np.array(g["test_pred"]["T"])
gms=[P[i][T[i]!=-1].mean() for i in range(len(P))]
print(f"GRAPE测试预测MS: mean={np.mean(gms):.2f} std={np.std(gms):.2f}")
