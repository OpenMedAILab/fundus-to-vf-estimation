"""Phase 4: 临床误差分析 (R3-3) — Bland-Altman / ICC / 严重度分层 / 失败案例
基于最优回归配置(按点位MAE)的跨seed集成 test 预测。"""
import json, glob, numpy as np
from sklearn.metrics import mean_absolute_error
from config import ROOT
CK=ROOT+"/ckpt"

# 选最优回归配置
agg=json.load(open(ROOT+"/aggregate_summary.json"))
best=min(agg["reg"].items(), key=lambda kv: kv[1]["point_mae_mean"])
cfg=best[0]; print("最优回归配置:",cfg, "point_mae=%.3f"%best[1]["point_mae_mean"])
arch,inp = cfg.rsplit("_",1) if cfg.count("_")==1 else ("_".join(cfg.split("_")[:-1]), cfg.split("_")[-1])
# 收集该配置所有seed
rs=[json.load(open(f)) for f in glob.glob(f"{CK}/reg_{cfg}_s*/result.json")]
P=np.mean([np.array(r["test_pred"]["P"]) for r in rs],0); T=np.array(rs[0]["test_pred"]["T"])
print(f"集成 {len(rs)} seeds, test样本 {len(T)}")

# 每眼 MS (有效点均值)
def ms(v): m=v!=-1; return v[m].mean()
ms_t=np.array([ms(T[i]) for i in range(len(T))]); ms_p=np.array([ms(P[i]) for i in range(len(P))])
# 每眼点位MAE
emae=np.array([mean_absolute_error(T[i][T[i]!=-1], P[i][T[i]!=-1]) for i in range(len(T))])

# Bland-Altman (MS)
diff=ms_p-ms_t; bias=diff.mean(); sd=diff.std(); loa=(bias-1.96*sd, bias+1.96*sd)
# ICC(2,1) 近似
def icc(a,b):
    m=np.c_[a,b]; n=len(m); gm=m.mean()
    msr=n*((m.mean(1)-gm)**2).sum()/(n-1)
    msc=2*((m.mean(0)-gm)**2).sum()/1
    mse=((m-m.mean(1,keepdims=True)-m.mean(0,keepdims=True)+gm)**2).sum()/(n-1)
    return float((msr-mse)/(msr+mse+2/n*(msc-mse)))
ICC=icc(ms_t,ms_p)

# 严重度分层 (按真实MS三分位: 重<低分位, 中, 轻>高分位)
q1,q2=np.percentile(ms_t,[33,66])
grp=np.where(ms_t<q1,"severe",np.where(ms_t<q2,"moderate","mild"))
sev={g:{"n":int((grp==g).sum()),"point_mae":float(emae[grp==g].mean()),"ms_mae":float(np.abs(diff)[grp==g].mean())} for g in ["mild","moderate","severe"]}

# 失败案例 (点位MAE最差10)
worst=sorted([{"idx":int(i),"point_mae":float(emae[i]),"true_ms":float(ms_t[i]),"pred_ms":float(ms_p[i])} for i in range(len(T))], key=lambda x:-x["point_mae"])[:10]

out={"best_config":cfg,"n_seeds":len(rs),"n_test":int(len(T)),
     "bland_altman":{"bias":float(bias),"sd":float(sd),"loa_lower":float(loa[0]),"loa_upper":float(loa[1]),"icc":ICC},
     "severity_stratified":sev,"worst_cases":worst,
     "overall":{"point_mae":float(emae.mean()),"ms_mae":float(np.abs(diff).mean())}}
json.dump(out, open(ROOT+"/clinical_analysis.json","w"), indent=2)
print(json.dumps(out["bland_altman"],indent=1)); print("严重度:",json.dumps(sev,ensure_ascii=False))
print("saved clinical_analysis.json")
