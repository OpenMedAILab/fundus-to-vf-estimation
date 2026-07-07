"""Wave2: 分类逐步消融 + 回归收敛配置, 跨GPU并行"""
import subprocess, time, os
ROOT="/remote-home/guijiangsheng/yyy/yang/fix_paper/qgy_xf/reproduce_full"
GPUS=[1,2,3,5,7]; os.makedirs(f"{ROOT}/logs",exist_ok=True)
C=[]
# 分类逐步消融: 纯CE无采样 -> CE+采样 -> Focal+采样
for arch in ["resnet","full_hybrid"]:
    for seed in [0,1,2]:
        C.append(dict(task="cls",arch=arch,input="cfp",seed=seed,variant="ce_nosamp",extra=["--loss","ce","--no_sampler"]))
        C.append(dict(task="cls",arch=arch,input="cfp",seed=seed,variant="ce_samp",extra=["--loss","ce"]))
        C.append(dict(task="cls",arch=arch,input="cfp",seed=seed,variant="focal_samp",extra=["--loss","focal"]))
# 回归收敛曲线 (带history)
for arch in ["hybrid","full_hybrid"]:
    C.append(dict(task="reg",arch=arch,input="cfp",seed=0,variant="conv",extra=[]))
print("配置数:",len(C))
def launch(c,g):
    tag=f"{c['task']}_{c['arch']}_{c['input']}_{c['variant']}_s{c['seed']}"
    cmd=["python",f"{ROOT}/repro.py","--task",c["task"],"--arch",c["arch"],"--input",c["input"],
         "--seed",str(c["seed"]),"--variant",c["variant"],"--epochs","80"]+c["extra"]
    log=open(f"{ROOT}/logs/{tag}.log","w")
    return subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g)),stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
q=list(C);run={};free=list(GPUS);done=0;t0=time.time()
while q or run:
    while q and free:
        g=free.pop(0);c=q.pop(0);run[g]=(launch(c,g),c)
        print("launch GPU%d: %s_%s_%s_s%d"%(g,c["task"],c["arch"],c["variant"],c["seed"]),flush=True)
    time.sleep(8)
    for g,(p,c) in list(run.items()):
        if p.poll() is not None:
            done+=1;del run[g];free.append(g)
            print("[%d/%d] done %s_%s_%s_s%d rc=%d"%(done,len(C),c["task"],c["arch"],c["variant"],c["seed"],p.returncode),flush=True)
print("EXTRA DONE %d in %.1fmin"%(len(C),(time.time()-t0)/60))
