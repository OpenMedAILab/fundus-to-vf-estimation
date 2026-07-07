import os, time
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT","30")
import timm, torch
from torchvision.models import resnet50, ResNet50_Weights
t=time.time()
m=timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
print("ViT OK in %.0fs"%(time.time()-t), flush=True)
r=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
print("ResNet50 V2 OK", flush=True)
