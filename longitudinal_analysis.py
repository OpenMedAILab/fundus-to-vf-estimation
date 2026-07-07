"""R4-2: 纵向记录定义 — 随访次数/时间间隔分布/年度结构 (基于 GRAPE Follow-up)"""
import pandas as pd, numpy as np, json, collections
XLS="/remote-home/guijiangsheng/yyy/yang/fix_paper/20251113/GRAPE Dataset/GRAPE Dataset/VF and clinical information.xlsx"
fu=pd.read_excel(XLS, sheet_name="Follow-up", header=[0,1]); fu.columns=["|".join([str(a),str(b)]) for a,b in fu.columns]
c=list(fu.columns); subj,lat,vis,interval,cfp=c[0],c[1],c[2],c[3],[x for x in c if "Corresponding" in x][0]
fu=fu[fu[subj].notna()].copy()
fu["img"]=fu[cfp].astype(str).str.endswith(".jpg")
fu["eye"]=fu[subj].astype(int).astype(str)+"_"+fu[lat].astype(str)
img=fu[fu["img"]]
# 随访次数(有图)
vpe=img.groupby("eye")[vis].count()
# 时间间隔(相邻随访)
ivals=pd.to_numeric(img[interval],errors="coerce").dropna()
out={
 "total_followup_records":int(len(fu)),
 "records_with_image":int(fu["img"].sum()),
 "unique_eyes_with_image":int(img["eye"].nunique()),
 "visits_per_eye(image)":{"mean":float(vpe.mean()),"min":int(vpe.min()),"max":int(vpe.max()),
    "dist":{int(k):int(v) for k,v in sorted(collections.Counter(vpe.values).items())}},
 "interval_years":{"mean":float(ivals.mean()),"median":float(ivals.median()),"min":float(ivals.min()),
    "max":float(ivals.max()),"p25":float(ivals.quantile(.25)),"p75":float(ivals.quantile(.75))},
 "followup_span_years_per_eye":{}
}
span=img.groupby("eye").apply(lambda g: pd.to_numeric(g[interval],errors="coerce").max())
out["followup_span_years_per_eye"]={"mean":float(span.mean()),"max":float(span.max()),
    "eyes_ge1yr":int((span>=1).sum()),"eyes_ge2yr":int((span>=2).sum())}
json.dump(out, open("/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/reproduce_full/longitudinal_stats.json","w"),indent=2,ensure_ascii=False)
print(json.dumps(out,ensure_ascii=False,indent=2))
