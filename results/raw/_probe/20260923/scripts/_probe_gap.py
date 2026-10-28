"""在 token 序列分岔的那一步，量 top1 与 top2 的差距。

若分岔步的 top1-top2 差距与数值扰动同量级（~0.03），
则分岔是"近平局被舍入翻面"，不是实现错误。
"""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "benchmark"); sys.path.insert(0, "scripts")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ascend_opt import Request
from real_backend import TransformerDriver, SyntheticPromptSource, ForwardItem
from validate_fp16_path import reference_greedy

MODEL = "Qwen/Qwen2.5-7B-Instruct"; dev = "musa:0"; GEN = 16
tok = AutoTokenizer.from_pretrained(MODEL)
m16 = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
excl = [v for n in ("eos_token_id","pad_token_id","bos_token_id","unk_token_id")
        if isinstance((v := getattr(tok, n, None)), int)]
src = SyntheticPromptSource(vocab_size=int(m16.config.vocab_size), seed=1234, excluded_ids=excl)

# --- 参考：逐 token，每步记 top1/top2 的 gap ---
req = Request(req_id="v4", prompt_len=40, gen_len=GEN)
ids = src.prompt_ids(req)
past = None
cur = torch.tensor([ids], dtype=torch.long, device=dev)
print("参考逐 token 的 top1/top2 gap（只看前几步分岔前）：")
with torch.no_grad():
    for step in range(GEN):
        out = m16(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        lg = out.logits[0, -1, :].float()
        v, i = torch.topk(lg, 2)
        print(f"  step{step}: top1={int(i[0])} top2={int(i[1])} gap={float(v[0]-v[1]):.4f}")
        cur = torch.tensor([[int(i[0])]], dtype=torch.long, device=dev)

# --- 分块路径：同样的逐步 gap ---
print("\n[16,16,8] 分块路径的 top1/top2 gap：")
class Grab:
    def __init__(self, inner): self._inner=inner; self.last=None
    def __call__(self,*a,**kw):
        o=self._inner(*a,**kw); self.last=o.logits.detach().float(); return o
    def __getattr__(self,n): return getattr(self._inner,n)
g = Grab(m16)
d = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                      model=g, tokenizer=tok, prompt_source=src)
d.load()
for ci,n in enumerate([16,16,8]):
    d.forward([ForwardItem(req=req, n_new_tokens=n, need_token=(ci==2))])
lg = g.last[0,-1,:].float(); v,i = torch.topk(lg,2)
print(f"  prefill末: top1={int(i[0])} top2={int(i[1])} gap={float(v[0]-v[1]):.4f}")
d.retire(["v4"])
