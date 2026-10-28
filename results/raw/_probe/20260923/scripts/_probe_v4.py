"""V4: 分块 prefill 与整段 prefill 是否自洽。

上版探针把 API 用错了（末块 need_token=False 后无法 decode），
驱动以明确报错拒绝 —— 这本身是好事。这里按正确用法重写：
末块 need_token=True，拿到第 1 个 token 后再逐 token decode。
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

req = Request(req_id="v4", prompt_len=40, gen_len=GEN)
ids = src.prompt_ids(req)
ref16 = reference_greedy(m16, tok, ids, GEN, dev)
print("ref fp16  :", ref16, repr(tok.decode(ref16)))

def run_whole():
    d = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                          model=m16, tokenizer=tok, prompt_source=src)
    d.load()
    out = []
    o = d.forward([ForwardItem(req=req, n_new_tokens=40, need_token=True)])
    out += [s.token_id for s in o.slots if s.token_id is not None]
    for _ in range(GEN-1):
        o = d.forward([ForwardItem(req=req, n_new_tokens=1, need_token=True)])
        out += [s.token_id for s in o.slots if s.token_id is not None]
    d.retire(["v4"])
    return out

def run_chunk(chunks):
    d = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                          model=m16, tokenizer=tok, prompt_source=src)
    d.load()
    out = []
    for ci, n in enumerate(chunks):
        last = ci == len(chunks)-1
        o = d.forward([ForwardItem(req=req, n_new_tokens=n, need_token=last)])
        out += [s.token_id for s in o.slots if s.token_id is not None]
    for _ in range(GEN-1):
        o = d.forward([ForwardItem(req=req, n_new_tokens=1, need_token=True)])
        out += [s.token_id for s in o.slots if s.token_id is not None]
    d.retire(["v4"])
    return out

whole = run_whole()
print("drv whole :", whole)
# 多种分块：全部必须与整段一致（这是分块正确性的真正判据）
for chunks in ([16,16,8], [8,8,8,8,8], [20,20], [1]*40, [39,1], [16,24]):
    got = run_chunk(chunks)
    print(f"chunks={str(chunks):<18} whole==chunk:{got == whole}  ref==chunk:{got == ref16}")
