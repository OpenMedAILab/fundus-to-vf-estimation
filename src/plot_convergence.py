"""Wave3: 收敛曲线 (R4-6) — train_loss + val_metric vs epoch"""
import json, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
from config import setup_cjk_font; setup_cjk_font(); import matplotlib.pyplot as plt
from config import CKPT, FIGURES
cfgs=[("reg_hybrid_cfp_conv_s0","回归 Hybrid-CFP (val MAE↓)"),
      ("reg_full_hybrid_cfp_conv_s0","回归 FullHybrid-CFP (val MAE↓)"),
      ("cls_resnet_cfp_focal_samp_s0","分类 ResNet (val PR-AUC↑)"),
      ("cls_full_hybrid_cfp_focal_samp_s0","分类 FullHybrid (val PR-AUC↑)")]
fig,axs=plt.subplots(1,len(cfgs),figsize=(5*len(cfgs),4))
for ax,(tag,title) in zip(axs,cfgs):
    fs=glob.glob(f"{CKPT}/{tag}/result.json")
    if not fs: ax.set_title(title+" (无)"); continue
    h=json.load(open(fs[0])).get("history",[])
    if not h: ax.set_title(title+" (无history)"); continue
    ep=[x["epoch"] for x in h]; tl=[x["train_loss"] for x in h]; vs=[x["val_score"] for x in h]
    be=json.load(open(fs[0]))["best_epoch"]
    ax2=ax.twinx()
    ax.plot(ep,tl,"b-",label="train loss"); ax2.plot(ep,vs,"r-",label="val metric")
    ax.axvline(be,color="g",ls="--",alpha=.6,label=f"best ep {be}")
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss",color="b"); ax2.set_ylabel("val metric",color="r")
    ax.set_title(title,fontsize=10); ax.legend(loc="upper right",fontsize=7)
plt.suptitle("Figure 3 (复现): 训练收敛曲线",fontsize=12); plt.tight_layout()
plt.savefig(f"{FIGURES}/fig_convergence.png",dpi=120,bbox_inches="tight")
print("saved fig_convergence.png")
