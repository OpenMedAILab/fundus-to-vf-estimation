"""Phase 5: 聚合所有 result.json -> 均值±std + bootstrap 95% CI -> 消融表 (reg & cls)"""
import json, glob, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, r2_score

ROOT = "/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/reproduce_full"
CK = ROOT + "/ckpt"
ARCH_ORDER = ["resnet", "transformer", "hybrid", "full_hybrid"]

def boot(fn, *arr, n=2000, seed=0):
    rng = np.random.RandomState(seed); N = len(arr[0]); out = []
    for _ in range(n):
        idx = rng.randint(0, N, N)
        try: out.append(fn(*[a[idx] for a in arr]))
        except Exception: pass
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (None, None)

res = [json.load(open(f)) for f in glob.glob(CK + "/*/result.json")]
print(f"收集 {len(res)} 个结果\n")

def group(task, aug):
    g = {}
    for r in res:
        if r["task"] != task or bool(r.get("aug")) != aug: continue
        g.setdefault((r["arch"], r["input"]), []).append(r)
    return g

# ---------- 回归表 ----------
def reg_table(aug=False):
    g = group("reg", aug)
    rows = []
    for arch in ARCH_ORDER:
        for inp in ["cfp", "roi", "annotated"]:
            rs = g.get((arch, inp))
            if not rs: continue
            mae = [r["test"]["point_mae"] for r in rs]; r2 = [r["test"]["point_r2"] for r in rs]
            msmae = [r["test"]["ms_mae"] for r in rs]; msr2 = [r["test"]["ms_r2"] for r in rs]
            # 集成: 跨seed平均预测, bootstrap CI on point_mae
            P = np.mean([np.array(r["test_pred"]["P"]) for r in rs], 0); T = np.array(rs[0]["test_pred"]["T"])
            mask = T != -1
            # 逐样本展平做 bootstrap (按样本重采样)
            def pmae(ii):
                Pi, Ti = P[ii], T[ii]; m = Ti != -1; return mean_absolute_error(Ti[m], Pi[m])
            ci = boot(lambda ii: pmae(ii), np.arange(len(T)))
            ens_mae = mean_absolute_error(T[mask], P[mask])
            rows.append((arch, inp, np.mean(mae), np.std(mae), ens_mae, ci, np.mean(r2), np.mean(msmae), np.mean(msr2), len(rs)))
    return rows

# ---------- 分类表 ----------
def cls_table(aug=False):
    g = group("cls", aug)
    rows = []
    for arch in ARCH_ORDER:
        for inp in ["cfp", "roi", "annotated"]:
            rs = g.get((arch, inp))
            if not rs: continue
            roc = [r["test"]["roc_auc"] for r in rs if r["test"].get("roc_auc") is not None]
            pr = [r["test"]["pr_auc"] for r in rs if r["test"].get("pr_auc") is not None]
            P = np.mean([np.array(r["test_pred"]["p"]) for r in rs], 0); Y = np.array(rs[0]["test_pred"]["y"])
            ci = boot(lambda y, p: roc_auc_score(y, p) if y.sum() not in (0, len(y)) else 0.5, Y, P)
            rows.append((arch, inp, np.mean(roc) if roc else None, np.std(roc) if roc else 0,
                         float(roc_auc_score(Y, P)) if Y.sum() not in (0, len(Y)) else None, ci,
                         np.mean(pr) if pr else None, len(rs)))
    return rows

out = {"reg": {}, "cls": {}}
print("="*90); print("回归 (VF estimation) — 点位 MAE↓ / R²↑ / MS-MAE↓ / MS-R²↑")
print("="*90)
print(f"{'arch':13s}{'input':11s}{'pointMAE mean±std':>20s}{'ens[95%CI]':>20s}{'pR2':>8s}{'msMAE':>8s}{'msR2':>8s}")
for r in reg_table(False):
    arch, inp, m, s, em, ci, r2, msm, msr2, n = r
    print(f"{arch:13s}{inp:11s}{m:8.3f}±{s:.3f}     {em:.3f}[{ci[0]:.2f},{ci[1]:.2f}] {r2:7.3f}{msm:8.3f}{msr2:8.3f}")
    out["reg"][f"{arch}_{inp}"] = {"point_mae_mean": m, "point_mae_std": s, "ens_mae": em, "ci95": ci,
                                   "point_r2": r2, "ms_mae": msm, "ms_r2": msr2, "n_seeds": n}

print("\n" + "="*90); print("分类 (PLR3) — ROC-AUC / PR-AUC (test阳性少, 仅供参考)")
print("="*90)
print(f"{'arch':13s}{'input':11s}{'ROC mean±std':>18s}{'ens[95%CI]':>20s}{'PR':>8s}")
for r in cls_table(False):
    arch, inp, m, s, em, ci, pr, n = r
    ms = f"{m:.3f}±{s:.3f}" if m is not None else "NA"
    es = f"{em:.3f}[{ci[0]:.2f},{ci[1]:.2f}]" if em is not None else "NA"
    print(f"{arch:13s}{inp:11s}{ms:>18s}{es:>20s}{(pr if pr else 0):8.3f}")
    out["cls"][f"{arch}_{inp}"] = {"roc_mean": m, "roc_std": s, "ens_roc": em, "ci95": ci, "pr_mean": pr, "n_seeds": n}

json.dump(out, open(ROOT + "/aggregate_summary.json", "w"), indent=2)
print("\nsaved:", ROOT + "/aggregate_summary.json")
