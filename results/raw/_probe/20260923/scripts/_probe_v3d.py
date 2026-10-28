"""v3_0 的失败是确定性的（5/5）。现在看：是不是"第一个 token 之后位置算错"。

关键观察：drv 比 ref 多出一个 token，且从第 5 位起错位。
注意 ref[0:4] == drv[0:4]，从下标 4 开始分叉。
看 prompt 长度与 KV 长度的关系：v3_0 prompt_len=16，批内 pad 到 48。
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
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
excl = [v for n in ("eos_token_id","pad_token_id","bos_token_id","unk_token_id")
        if isinstance((v := getattr(tok, n, None)), int)]
src = SyntheticPromptSource(vocab_size=int(m.config.vocab_size), seed=1234, excluded_ids=excl)

L = [16, 24, 33, 48]
reqs = [Request(req_id=f"v3_{i}", prompt_len=n, gen_len=GEN) for i, n in enumerate(L)]
ids0 = src.prompt_ids(reqs[0])
print("v3_0 prompt ids:", ids0)
print("v3_0 decoded   :", repr(tok.decode(ids0)))
ref0 = reference_greedy(m, tok, ids0, GEN, dev)
print("ref0:", ref0, "->", repr(tok.decode(ref0)))

# 手工驱动 v3_0：只 prefill，逐步看 KV 长度与 position
d = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                      model=m, tokenizer=tok, prompt_source=src)
d.load()
print("\n-- 单跑 v3_0，逐次 forward 观察 --")
out = d.forward([ForwardItem(req=reqs[0], n_new_tokens=16, need_token=True)])
print("  prefill 后 token:", [s.token_id for s in out.slots])
for i in range(3):
    o = d.forward([ForwardItem(req=reqs[0], n_new_tokens=1, need_token=True)])
    print(f"  decode{i} token:", [s.token_id for s in o.slots])
d.retire(["v3_0"])

# 批内跑 v3_0：与长行同批
print("\n-- 批内 v3_0（pad 到 48）--")
d2 = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                       model=m, tokenizer=tok, prompt_source=src)
d2.load()
out2 = d2.forward([ForwardItem(req=r, n_new_tokens=16, need_token=True) for r in reqs])
print("  prefill 后各行 token:", {s.req_id: s.token_id for s in out2.slots})
for i in range(3):
    o = d2.forward([ForwardItem(req=r, n_new_tokens=1, need_token=True) for r in reqs])
    print(f"  decode{i}:", {s.req_id: s.token_id for s in o.slots})
d2.retire([r.req_id for r in reqs])
