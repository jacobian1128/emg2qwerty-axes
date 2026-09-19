import sys, torch, yaml
from torch.utils.flop_counter import FlopCounterMode
from omegaconf import OmegaConf
from hydra.utils import instantiate
ckpt_path, cfg_path = sys.argv[1], sys.argv[2]
cfg = OmegaConf.load(cfg_path)
m = instantiate(cfg.module, optimizer=None, lr_scheduler=None, decoder=None, _recursive_=False) if "decoder" in str(cfg.module) or True else None
sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
missing, unexpected = m.load_state_dict(sd, strict=False)
print("params", sum(p.numel() for p in m.parameters())/1e6, "missing", len(missing), "unexpected", len(unexpected))
m.eval()
T = 60000 // 16  # 30 s at 2 kHz, hop 16
x = torch.randn(T, 1, 2, 16, 33)
with torch.no_grad():
    fc = FlopCounterMode(display=False)
    with fc:
        m.forward(x)
    print("GFLOPs/30s", fc.get_total_flops()/1e9)
