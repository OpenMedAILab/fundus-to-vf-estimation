"""GPU 并行编排: 把 (task×arch×input×seed) 配置分发到空闲 GPU。
用法: python orchestrate.py <reg|cls|all> [--aug] [--seeds 0,1,2] [--gpus 1,2,3,5,7]"""
import subprocess, time, os, sys, argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("which", choices=["reg", "cls", "all"])
ap.add_argument("--aug", action="store_true")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--gpus", default="1,2,3,5,7")
ap.add_argument("--archs", default="resnet,transformer,hybrid,full_hybrid")
ap.add_argument("--inputs", default="cfp,roi,annotated")
ap.add_argument("--epochs", type=int, default=80)
a = ap.parse_args()
from config import SRC, LOGS
os.makedirs(LOGS, exist_ok=True)
GPUS = [int(g) for g in a.gpus.split(",")]
seeds = [int(s) for s in a.seeds.split(",")]
tasks = ["reg", "cls"] if a.which == "all" else [a.which]
CONFIGS = [dict(task=t, arch=ar, input=i, seed=s)
           for t in tasks for ar in a.archs.split(",") for i in a.inputs.split(",") for s in seeds]
print(f"配置总数: {len(CONFIGS)} | GPU: {GPUS} | aug={a.aug}")

def launch(cfg, gpu):
    tag = f"{cfg['task']}_{cfg['arch']}_{cfg['input']}{'_aug' if a.aug else ''}_s{cfg['seed']}"
    cmd = ["python", f"{SRC}/repro.py", "--task", cfg["task"], "--arch", cfg["arch"],
           "--input", cfg["input"], "--seed", str(cfg["seed"]), "--epochs", str(a.epochs)]
    if a.aug: cmd.append("--aug")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log = open(f"{LOGS}/{tag}.log", "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)

queue = list(CONFIGS); running = {}; free = list(GPUS); done = 0; t0 = time.time()
while queue or running:
    while queue and free:
        g = free.pop(0); cfg = queue.pop(0); running[g] = (launch(cfg, g), cfg)
        print(f"launch GPU{g}: {cfg['task']}_{cfg['arch']}_{cfg['input']}_s{cfg['seed']}", flush=True)
    time.sleep(8)
    for g, (p, cfg) in list(running.items()):
        if p.poll() is not None:
            done += 1; del running[g]; free.append(g)
            print(f"[{done}/{len(CONFIGS)}] done GPU{g}: {cfg['task']}_{cfg['arch']}_{cfg['input']}_s{cfg['seed']} (rc={p.returncode}) | {(time.time()-t0)/60:.1f}min", flush=True)
print(f"ALL DONE: {len(CONFIGS)} configs in {(time.time()-t0)/60:.1f} min")
