"""R3-3扩展: 预测VF vs 真实VF 严重度分级一致性(κ/混淆矩阵/敏感度特异度) + 决策曲线(DCA)"""
import json, glob, numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from config import ROOT
agg=json.load(open(ROOT+"/aggregate_summary.json"))
best=min(agg["reg"].items(), key=lambda kv: kv[1]["point_mae_mean"])[0]
rs=[json.load(open(f)) for f in glob.glob(f"{ROOT}/ckpt/reg_{best}_s*/result.json")]
P=np.mean([np.array(r["test_pred"]["P"]) for r in rs],0); T=np.array(rs[0]["test_pred"]["T"])
def ms(v): m=v!=-1; return v[m].mean()
ms_t=np.array([ms(T[i]) for i in range(len(T))]); ms_p=np.array([ms(P[i]) for i in range(len(P))])
# 严重度分级 (按真实MS分布: 临床近似 — 高=轻, 低=重)
def grade(x): return np.where(x>=26,"mild",np.where(x>=22,"moderate","severe"))
gt,gp=grade(ms_t),grade(ms_p)
labels=["mild","moderate","severe"]
cm=confusion_matrix(gt,gp,labels=labels).tolist()
kappa=float(cohen_kappa_score(gt,gp,labels=labels))
kappa_w=float(cohen_kappa_score(gt,gp,labels=labels,weights="linear"))
# 二分类"重度" (severe vs not) 的敏感度/特异度
sev_t=(gt=="severe").astype(int); sev_p=(gp=="severe").astype(int)
tn,fp,fn,tp=confusion_matrix(sev_t,sev_p,labels=[0,1]).ravel()
sens=tp/(tp+fn) if tp+fn else 0; spec=tn/(tn+fp) if tn+fp else 0
# DCA: "预测重度则转诊", 真实重度为label, 用预测MS阈值扫净获益
risk=1-(ms_p-ms_p.min())/(ms_p.max()-ms_p.min()+1e-9)  # 越重风险越高
n=len(sev_t); prev=sev_t.mean()
dca=[]
for pt in np.arange(0.05,0.6,0.05):
    pred=(risk>=pt).astype(int)
    tp_=((pred==1)&(sev_t==1)).sum(); fp_=((pred==1)&(sev_t==0)).sum()
    nb=tp_/n - fp_/n*(pt/(1-pt))
    nb_all=prev - (1-prev)*(pt/(1-pt))
    dca.append({"pt":round(float(pt),2),"net_benefit_model":float(nb),"net_benefit_treat_all":float(nb_all)})
out={"best_config":best,"grading_cutoffs_MS":{"mild":">=26","moderate":"22-26","severe":"<22"},
     "confusion_matrix(rows=true,cols=pred; mild/mod/severe)":cm,
     "cohen_kappa":kappa,"weighted_kappa_linear":kappa_w,
     "severe_vs_rest":{"sens":float(sens),"spec":float(spec),"cm_tn_fp_fn_tp":[int(tn),int(fp),int(fn),int(tp)]},
     "decision_curve":dca}
json.dump(out,open(ROOT+"/grading_dca.json","w"),indent=2,ensure_ascii=False)
print("κ=%.3f  weighted-κ=%.3f  severe sens=%.2f spec=%.2f"%(kappa,kappa_w,sens,spec))
print("混淆矩阵(mild/mod/severe):",cm)
print("DCA(部分):",[(d["pt"],round(d["net_benefit_model"],3)) for d in dca[:5]])
