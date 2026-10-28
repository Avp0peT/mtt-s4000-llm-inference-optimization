"""跟踪分块 prefill 时每次 forward 的 KV 长度与 position_ids。

对照 [20,20]（过）与 [16,16,8]（挂），看哪一步开始不一致。
用 _LogitsRecorder 的方式拦 model.forward，记录 kwargs 形状。
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

class Spy:
    def __init__(self, inner): self._inner = inner; self.log = []
    def __call__(self, *a, **kw):
        ids = kw.get("input_ids"); am = kw.get("attention_mask"); pid = kw.get("position_ids")
        pk = kw.get("past_key_values")
        try: sl = pk.get_seq_length()
        except Exception: sl = None
        self.log.append({
            "ids": list(ids.shape) if hasattr(ids,'shape') else None,
            "mask": list(am.shape) if hasattr(am,'shape') else None,
            "mask_sum": int(am.sum().item()) if hasattr(am,'sum') else None,
            "pos": list(pid[0].tolist()) if pid is not None and hasattr(pid,'tolist') else None,
            "past_len": sl,
        })
        return self._inner(*a, **kw)
    def __getattr__(self, n): return getattr(self._inner, n)

for chunks in ([20,20], [16,16,8]):
    print(f"\n===== chunks={chunks} =====")
    spy = Spy(m16)
    d = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                          model=spy, tokenizer=tok, prompt_source=src)
    d.load()
    for ci, n in enumerate(chunks):
        last = ci == len(chunks)-1
        d.forward([ForwardItem(req=req, n_new_tokens=n, need_token=last)])
    d.retire(["v4"])
    for i, r in enumerate(spy.log):
        print(f"  call{i}: ids={r['ids']} mask={r['mask']} mask_sum={r['mask_sum']} "
              f"pos={r['pos']} past_len={r['past_len']}")
