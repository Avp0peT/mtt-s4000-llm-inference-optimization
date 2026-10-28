"""诊断新版 V3 判据为何给出 13 量级差异。"""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "benchmark"); sys.path.insert(0, "scripts")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ascend_opt import Request
from real_backend import TransformerDriver, SyntheticPromptSource, ForwardItem

MODEL = "Qwen/Qwen2.5-7B-Instruct"; dev = "musa:0"
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
excl = [v for n in ("eos_token_id","pad_token_id","bos_token_id","unk_token_id")
        if isinstance((v := getattr(tok, n, None)), int)]
src = SyntheticPromptSource(vocab_size=int(m.config.vocab_size), seed=1234, excluded_ids=excl)

L = [16, 24, 33, 48]
reqs = [Request(req_id=f"v3_{i}", prompt_len=n, gen_len=16) for i, n in enumerate(L)]

class Grab:
    def __init__(self, inner): self._inner=inner; self.last=None; self.calls=[]
    def __call__(self,*a,**kw):
        ids=kw.get("input_ids"); am=kw.get("attention_mask"); pid=kw.get("position_ids")
        pk=kw.get("past_key_values")
        try: sl=pk.get_seq_length()
        except Exception: sl=None
        self.calls.append((list(ids.shape) if hasattr(ids,'shape') else None,
                           list(am.shape) if hasattr(am,'shape') else None,
                           (list(pid[0].tolist())[:3], list(pid[0].tolist())[-3:]) if pid is not None else None,
                           sl))
        o=self._inner(*a,**kw); self.last=o.logits.detach().float().cpu(); return o
    def __getattr__(self,n): return getattr(self._inner,n)

for r in reqs:
    ids = src.prompt_ids(r)
    # 参考
    rg = Grab(m)
    with torch.no_grad():
        rg(input_ids=torch.tensor([ids],dtype=torch.long,device=dev), use_cache=True)
    ref_lg = rg.last[0,-1,:]

    # 驱动：只有该行单独（不用 padding），看是否与参考一致
    g1 = Grab(m)
    d1 = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                           model=g1, tokenizer=tok, prompt_source=src)
    d1.load()
    d1.forward([ForwardItem(req=r, n_new_tokens=r.prompt_len, need_token=False)])
    l1 = g1.last[0,-1,:] if g1.last is not None else None
    d1.retire([r.req_id])

    # 驱动：与下一行同批（有 padding）
    other = reqs[(L.index(r.prompt_len)+1) % len(reqs)]
    g2 = Grab(m)
    d2 = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                           model=g2, tokenizer=tok, prompt_source=src)
    d2.load()
    d2.forward([ForwardItem(req=r, n_new_tokens=r.prompt_len, need_token=False),
                ForwardItem(req=other, n_new_tokens=other.prompt_len, need_token=False)])
    l2 = g2.last[0,-1,:] if g2.last is not None else None
    d2.retire([r.req_id, other.req_id])

    print(f"\n{r.req_id} len={r.prompt_len} 的 forward 调用：")
    for c in rg.calls: print("   ref :", c)
    for c in g1.calls: print("   drv1:", c)
    for c in g2.calls: print("   drv2:", c)
    print(f"   |drv1-ref| max = {float((l1-ref_lg).abs().max()):.4f}" if l1 is not None else "   drv1 None")
    print(f"   |drv2-ref| max = {float((l2-ref_lg).abs().max()):.4f}" if l2 is not None else "   drv2 None")
