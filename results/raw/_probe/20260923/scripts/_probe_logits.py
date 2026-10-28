"""决定性判据：分块与整段的**最后一位 logits 差多少**。

若差值在 fp16 舍入量级（~1e-2 绝对，或 top1/top2 几乎并列），
则 token 分岔是数值不可分辨，不是实现错误。
若差值巨大（>1），则是真错。
"""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "benchmark"); sys.path.insert(0, "scripts")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ascend_opt import Request
from real_backend import TransformerDriver, SyntheticPromptSource, ForwardItem

MODEL = "Qwen/Qwen2.5-7B-Instruct"; dev = "musa:0"
tok = AutoTokenizer.from_pretrained(MODEL)
m16 = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
excl = [v for n in ("eos_token_id","pad_token_id","bos_token_id","unk_token_id")
        if isinstance((v := getattr(tok, n, None)), int)]
src = SyntheticPromptSource(vocab_size=int(m16.config.vocab_size), seed=1234, excluded_ids=excl)
req = Request(req_id="v4", prompt_len=40, gen_len=1)

class Grab:
    def __init__(self, inner): self._inner = inner; self.last = None
    def __call__(self, *a, **kw):
        out = self._inner(*a, **kw)
        self.last = out.logits.detach().to("cpu", torch.float32)
        return out
    def __getattr__(self, n): return getattr(self._inner, n)

def last_logits(chunks):
    g = Grab(m16)
    d = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                          model=g, tokenizer=tok, prompt_source=src)
    d.load()
    for ci, n in enumerate(chunks):
        d.forward([ForwardItem(req=req, n_new_tokens=n, need_token=(ci==len(chunks)-1))])
    d.retire(["v4"])
    return g.last[0, -1, :]      # 最后一个位置的分布

whole = last_logits([40])
print("整段 top1:", int(whole.argmax()), " top1 值:", float(whole.max()))

for chunks in ([20,20], [16,16,8], [8,8,8,8,8]):
    lg = last_logits(chunks)
    d = (lg - whole).abs()
    top1 = int(lg.argmax())
    # top1/top2 的差距（在整段的分布上）
    v, idx = torch.topk(whole, 2)
    print(f"chunks={str(chunks):<16} top1={top1:<7} max|Δ|={float(d.max()):.4f} "
          f"mean|Δ|={float(d.mean()):.6f} | 整段 top1={int(idx[0])} val={float(v[0]):.4f} "
          f"top2={int(idx[1])} val={float(v[1]):.4f} 差={float(v[0]-v[1]):.4f}")
