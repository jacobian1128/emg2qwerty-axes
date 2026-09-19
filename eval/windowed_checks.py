"""Verification of the windowed (tail-stream) encoder reported in the paper (Results, composition).

  python eval/windowed_checks.py <weights-only or full ckpt> [--device cuda]

Prints, for TailStreamConformerModule with the fixed per-session positional constant:
  A) exactness: a window spanning the whole input reproduces the single pass
     (max/mean abs diff of log-probs, argmax agreement)
  B) batch invariance: windowed outputs for batch sizes 32 / 7 / 1 agree
  C) FLOPs of one 624-frame window (FlopCounterMode), scaled to 30 s of input
  D) fixed-PE vs cycling-PE: greedy output difference when windows take the batch-position
     constants of training instead of the session's constant (log-prob diff only; the
     per-participant CER comparison is run with eval_8users.sh, env TAIL_PE=cycle)
"""
import argparse, os, sys, time
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conformer"))
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from emg2qwerty import conformer as C  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = a.device
m = C.TailStreamConformerModule.load_from_checkpoint(a.ckpt, map_location="cpu").to(dev).eval()

def single_pass(x, L):
    h = m.flatten(m.mlp(m.norm(x))); h, _ = m.conformer(h, L); return m.output_layer(h)

def cmp(tag, y1, y2):
    d = (y1 - y2).abs()
    print(f"{tag}: max={float(d.max()):.2e} mean={float(d.mean()):.2e} argmax_agree={float((y1.argmax(-1)==y2.argmax(-1)).float().mean()):.4f}")

# A) exactness -------------------------------------------------------------
torch.manual_seed(1); T = 800
x = torch.randn(T, 1, 2, 16, 33, device=dev); L = torch.tensor([T], device=dev)
os.environ.update(TAIL_SEG="1000", TAIL_DELAY="1000", TAIL_SEG_BATCH="32")
with torch.no_grad():
    yw = m(x, L); ys = single_pass(x, L)
print("fixed-PE installed:", isinstance(m.conformer.positional_encoding, C._FixedIndexPE))
cmp("A) whole-input window vs single pass", yw, ys)

# B) batch invariance --------------------------------------------------------
torch.manual_seed(2); T = 3000
x = torch.randn(T, 1, 2, 16, 33, device=dev); L = torch.tensor([T], device=dev)
os.environ.update(TAIL_SEG="624", TAIL_DELAY="62")
outs = {}
for sb in ["32", "7", "1"]:
    os.environ["TAIL_SEG_BATCH"] = sb
    with torch.no_grad(): outs[sb] = m(x, L)
for p, q in [("32", "1"), ("7", "1"), ("32", "7")]:
    cmp(f"B) batch {p} vs {q}", outs[p], outs[q])

# C) FLOPs -------------------------------------------------------------------
xw = torch.randn(624, 1, 2, 16, 33, device=dev); Lw = torch.tensor([624], device=dev)
with torch.no_grad(), FlopCounterMode(display=False) as fc:
    single_pass(xw, Lw)
fl = fc.get_total_flops(); per30 = fl * (30 / 0.096)
print(f"C) one 624-frame window: {fl/1e9:.2f} GFLOPs; x{30/0.096:.1f} windows per 30 s = {per30/1e12:.2f} TFLOPs")

# D) fixed vs cycling PE ---------------------------------------------------------
os.environ["TAIL_SEG_BATCH"] = "32"
with torch.no_grad(): y_fixed = m(x, L)
pe = m.conformer.positional_encoding
class _Cycling(C.PositionalEncoding):  # original behaviour: pe[i] for batch element i
    def forward(self, z): return self.dropout(z + self.pe[: z.shape[0]])
cyc = _Cycling(pe.pe.shape[2], dropout=pe.dropout.p, max_len=pe.pe.shape[0]).to(dev); cyc.pe = pe.pe
m.conformer.positional_encoding = cyc
C.TailStreamConformerModule.forward = C._tail_forward_orig  # bypass the auto-install
with torch.no_grad(): y_cyc = m(x, L)
cmp("D) fixed PE vs cycling PE (windowed)", y_fixed, y_cyc)
