"""隔离 V3 失败原因：把 v3_0 单跑 vs 放进变长批里跑，看是不是 batching 引入的。

三种口径：
  A. 参考逐 token 单跑（无 padding）
  B. 驱动单跑 v3_0（无 padding）
  C. 驱动把 v3_0 与长 prompt 放同一批（有 padding）
若 A==B 而 B!=C，则问题在 padding/批内位置；若 A!=B 则与 padding 无关。
"""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "benchmark"); sys.path.insert(0, "scripts")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ascend_opt import Request
from real_backend import TransformerDriver, ForwardItem, SyntheticPromptSource
from validate_fp16_path import reference_greedy, run_driver_greedy

MODEL = "Qwen/Qwen2.5-7B-Instruct"
dev = "musa:0"
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
excl = [v for n in ("eos_token_id","pad_token_id","bos_token_id","unk_token_id")
        if isinstance((v := getattr(tok, n, None)), int)]
src = SyntheticPromptSource(vocab_size=int(m.config.vocab_size), seed=1234, excluded_ids=excl)
GEN = 16

r0 = Request(req_id="v3_0", prompt_len=16, gen_len=GEN)
ids0 = src.prompt_ids(r0)

A = reference_greedy(m, tok, ids0, GEN, dev)
print("A ref single      :", A)

# B: 驱动单跑
d_b = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                        model=m, tokenizer=tok, prompt_source=src)
d_b.load()
B = run_driver_greedy(d_b, [r0], src, GEN)["v3_0"]
d_b.retire(["v3_0"])
print("B driver single   :", B)

# C: 驱动与长 prompt 同批
long = Request(req_id="v3_3", prompt_len=48, gen_len=GEN)
d_c = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                        model=m, tokenizer=tok, prompt_source=src)
d_c.load()
Cres = run_driver_greedy(d_c, [r0, long], src, GEN)
C = Cres["v3_0"]; Cl = Cres["v3_3"]
d_c.retire(["v3_0", "v3_3"])
print("C driver in batch :", C)
print("C long row        :", Cl)
print()
print("A==B (单跑一致)     :", A == B)
print("B==C (批不影响)     :", B == C)
print("A==C (批也不影响)   :", A == C)
# 长行是否受影响
ref_long = reference_greedy(m, tok, src.prompt_ids(long), GEN, dev)
print("long ref          :", ref_long)
print("long A==C         :", ref_long == Cl)
