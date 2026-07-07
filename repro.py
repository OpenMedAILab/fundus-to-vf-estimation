"""
GRAPE 论文复现 — 核心训练脚本 (自包含)
任务: reg (VF估计, 61点, mask -1) | cls (PLR3二分类, Focal+采样)
架构: resnet | transformer | hybrid | full_hybrid (cross-attention)
输入: cfp | roi | annotated
用法: python repro.py --task reg --arch full_hybrid --input cfp --seed 0 [--aug]
"""
import argparse, json, os, time
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT","30")
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix)
import timm
from PIL import Image

BASE = "/remote-home/guijiangsheng/yyy/yang/fix_paper/20251113/GRAPE Dataset/GRAPE Dataset"
DATA = "/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/reproduce_full/data"
OUTROOT = "/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/reproduce_full/ckpt"
AUGDIR = "/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/reproduce_full/synthetic"
N_VF = 61

# ---------------- Dataset ----------------
class GRAPE(Dataset):
    DIRS = {"cfp": "CFPs", "roi": "ROI images", "annotated": "Annotated Images"}
    def __init__(self, index_path, task, inp, train=False):
        self.recs = json.load(open(index_path))
        self.dir = os.path.join(BASE, self.DIRS[inp]); self.task = task
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
               transforms.ColorJitter(0.1, 0.1, 0.1)] if train else []
        self.tf = transforms.Compose([transforms.Resize((224, 224)), *aug, transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __len__(self): return len(self.recs)
    def labels(self): return [int(r["plr3"]) if r.get("plr3") is not None else 0 for r in self.recs]
    def __getitem__(self, i):
        r = self.recs[i]
        d = AUGDIR if r.get("is_synthetic") else self.dir
        img = self.tf(Image.open(os.path.join(d, r["cfp_filename"])).convert("RGB"))
        if self.task == "reg":
            vf = [v if v is not None else -1.0 for v in r["vf"]]
            return img, torch.tensor(vf, dtype=torch.float32)
        return img, torch.tensor(int(r["plr3"]) if r.get("plr3") is not None else 0, dtype=torch.long)

# ---------------- Models ----------------
def _resnet_feat():  # -> (B,2048,1,1) backbone
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    return nn.Sequential(*list(net.children())[:-1])
def _resnet_map():   # -> (B,2048,7,7)
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    return nn.Sequential(*list(net.children())[:-2])

class ResNetOnly(nn.Module):
    def __init__(self, out):
        super().__init__(); self.b = _resnet_feat()
        self.h = nn.Sequential(nn.Flatten(), nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, out))
    def forward(self, x): return self.h(self.b(x))

class TransformerOnly(nn.Module):
    def __init__(self, out):
        super().__init__(); self.v = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.h = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, out))
    def forward(self, x): return self.h(self.v(x))

class Hybrid(nn.Module):  # ResNet+Transformer 简单拼接 (无 cross-attn)
    def __init__(self, out):
        super().__init__(); self.c = _resnet_feat()
        self.v = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.pc = nn.Linear(2048, 256); self.pv = nn.Linear(768, 256)
        self.h = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, out))
    def forward(self, x):
        c = self.pc(self.c(x).flatten(1)); v = self.pv(self.v(x))
        return self.h(torch.cat([c, v], 1))

class FullHybrid(nn.Module):  # ResNet+Transformer + cross-attention (论文最终模型)
    def __init__(self, out, heads=8):
        super().__init__(); self.c = _resnet_map()
        self.v = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.pc = nn.Linear(2048, 256); self.pv = nn.Linear(768, 256)
        self.cross = nn.MultiheadAttention(256, heads, dropout=0.3, batch_first=True)
        self.h = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, out))
    def forward(self, x):
        ctok = self.pc(self.c(x).flatten(2).transpose(1, 2))          # (B,49,256)
        vtok = self.pv(self.v.forward_features(x)[:, 1:, :])          # (B,196,256)
        att, _ = self.cross(ctok, vtok, vtok)                        # cnn query attends to vit
        return self.h(att.mean(1))

ARCH = {"resnet": ResNetOnly, "transformer": TransformerOnly, "hybrid": Hybrid, "full_hybrid": FullHybrid}

class Focal(nn.Module):
    def __init__(self, a=0.75, g=2.0): super().__init__(); self.a, self.g = a, g
    def forward(self, logit, t):
        ce = F.cross_entropy(logit, t, reduction="none"); pt = torch.exp(-ce)
        a = torch.where(t == 1, self.a, 1 - self.a); return (a * (1 - pt) ** self.g * ce).mean()

# ---------------- Metrics ----------------
def reg_metrics(P, T):  # P,T: (N,61) ; mask T==-1
    m = T != -1
    mae = float(mean_absolute_error(T[m], P[m])); rmse = float(np.sqrt(mean_squared_error(T[m], P[m])))
    r2 = float(r2_score(T[m], P[m]))
    # MS = 每样本有效点均值
    ms_t = np.array([T[i][T[i] != -1].mean() for i in range(len(T))])
    ms_p = np.array([P[i][T[i] != -1].mean() for i in range(len(P))])
    return {"point_mae": mae, "point_rmse": rmse, "point_r2": r2,
            "ms_mae": float(mean_absolute_error(ms_t, ms_p)), "ms_r2": float(r2_score(ms_t, ms_p))}

def cls_metrics(y, p, thr):
    if y.sum() in (0, len(y)): return {"roc_auc": None, "pr_auc": None}
    pred = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "sens": tp / (tp + fn) if tp + fn else 0.0, "spec": tn / (tn + fp) if tn + fp else 0.0,
            "bal_acc": float(balanced_accuracy_score(y, pred)), "cm": [int(tn), int(fp), int(fn), int(tp)], "thr": float(thr)}

# ---------------- Train ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["reg", "cls"], required=True)
    ap.add_argument("--arch", choices=list(ARCH), required=True)
    ap.add_argument("--input", choices=["cfp", "roi", "annotated"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", action="store_true", help="use StyleGAN-augmented train index")
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--loss", choices=["ce","focal"], default="focal")
    ap.add_argument("--no_sampler", action="store_true")
    ap.add_argument("--variant", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{a.task}_{a.arch}_{a.input}{'_aug' if a.aug else ''}{('_'+a.variant) if a.variant else ''}_s{a.seed}"
    ck = os.path.join(OUTROOT, tag); os.makedirs(ck, exist_ok=True)

    tr_idx = f"{DATA}/{a.task}_train{'_aug' if a.aug else ''}.json"
    tr = GRAPE(tr_idx, a.task, a.input, train=True)
    va = GRAPE(f"{DATA}/{a.task}_val.json", a.task, a.input)
    te = GRAPE(f"{DATA}/{a.task}_test.json", a.task, a.input)

    if a.task == "cls" and not a.no_sampler:
        lab = np.array(tr.labels()); w = np.where(lab == 1, max((lab == 0).sum() / max((lab == 1).sum(), 1), 1), 1.0)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), True)
        trl = DataLoader(tr, a.bs, sampler=sampler, num_workers=6, drop_last=True)
    else:
        trl = DataLoader(tr, a.bs, shuffle=True, num_workers=6, drop_last=True)
    val = DataLoader(va, a.bs, shuffle=False, num_workers=4); tel = DataLoader(te, a.bs, shuffle=False, num_workers=4)

    out_dim = N_VF if a.task == "reg" else 2
    model = ARCH[a.arch](out_dim).to(dev)
    opt = optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = (Focal() if a.loss=="focal" else nn.CrossEntropyLoss()) if a.task == "cls" else None

    def run_eval(loader):
        model.eval()
        with torch.no_grad():
            if a.task == "reg":
                Ps, Ts = [], []
                for x, y in loader:
                    Ps.append(model(x.to(dev)).cpu().numpy()); Ts.append(y.numpy())
                return np.concatenate(Ps), np.concatenate(Ts)
            else:
                ys, ps = [], []
                for x, y in loader:
                    ps += torch.softmax(model(x.to(dev)), 1)[:, 1].cpu().tolist(); ys += y.tolist()
                return np.array(ys), np.array(ps)

    best, best_ep, wait = (1e9 if a.task == "reg" else -1), 0, 0
    hist=[]
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); ep_loss=0.0
        for x, y in trl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            out = model(x)
            if a.task == "reg":
                m = y != -1; loss = F.mse_loss(out[m], y[m])
            else:
                loss = crit(out, y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            ep_loss += loss.item()*x.size(0)
        sch.step()
        if a.task == "reg":
            P, T = run_eval(val); score = reg_metrics(P, T)["point_mae"]; better = score < best
        else:
            y, p = run_eval(val); mm = cls_metrics(y, p, 0.5); score = mm.get("pr_auc") or -1; better = score > best
        hist.append({"epoch":ep,"train_loss":ep_loss/max(len(tr),1),"val_score":float(score)})
        if better:
            best, best_ep, wait = score, ep, 0; torch.save(model.state_dict(), f"{ck}/best.pth")
        else:
            wait += 1
            if wait >= a.patience: break

    model.load_state_dict(torch.load(f"{ck}/best.pth", map_location=dev))
    res = {"tag": tag, "task": a.task, "arch": a.arch, "input": a.input, "aug": a.aug, "seed": a.seed,
           "variant": a.variant, "loss": a.loss, "no_sampler": a.no_sampler,
           "best_epoch": best_ep, "epochs_run": ep, "minutes": (time.time() - t0) / 60, "history": hist}
    if a.task == "reg":
        P, T = run_eval(tel); res["test"] = reg_metrics(P, T)
        res["test_pred"] = {"P": P.tolist(), "T": T.tolist()}
    else:
        vy, vp = run_eval(val); ty, tp = run_eval(tel)
        # 阈值在 val 上选 (F1 最优)
        best_thr, bf1 = 0.5, -1
        for thr in np.unique(vp):
            pr = (vp >= thr).astype(int); tp_, fp_, fn_ = ((vy == 1) & (pr == 1)).sum(), ((vy == 0) & (pr == 1)).sum(), ((vy == 1) & (pr == 0)).sum()
            f1 = 2 * tp_ / (2 * tp_ + fp_ + fn_) if (2 * tp_ + fp_ + fn_) else 0
            if f1 > bf1: bf1, best_thr = f1, float(thr)
        res["test"] = cls_metrics(ty, tp, best_thr); res["test_pred"] = {"y": ty.tolist(), "p": tp.tolist()}
    json.dump(res, open(f"{ck}/result.json", "w"), indent=1)
    print(f"[{tag}] best@{best_ep} test={json.dumps(res['test'])}")

if __name__ == "__main__":
    main()
