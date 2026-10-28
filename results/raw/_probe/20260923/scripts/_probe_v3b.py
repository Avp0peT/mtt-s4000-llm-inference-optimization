"""V3 的批规模是变量：2 行过、4 行挂。逐步加行，定位从哪一档开始坏。"""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "benchmark"); sys.path.insert(0, "scripts")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ascend_opt import Request
from real_backend import TransformerDriver, ForwardItem, SyntheticPromptSource
from validate_fp16_path import reference_greedy, run_driver_greedy

MODEL = "Qwen/Qwen2.5-7B-Instruct"; dev = "musa:0"; GEN = 16
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
excl = [v for n in ("eos_token_id","pad_token_id","bos_token_id","unk_token_id")
        if isinstance((v := getattr(tok, n, None)), int)]
src = SyntheticPromptSource(vocab_size=int(m.config.vocab_size), seed=1234, excluded_ids=excl)

LENGTHS = [16, 24, 33, 48]
reqs = [Request(req_id=f"v3_{i}", prompt_len=n, gen_len=GEN) for i, n in enumerate(LENGTHS)]
refs = {r.req_id: reference_greedy(m, tok, src.prompt_ids(r), GEN, dev) for r in reqs}

# 逐步加行：1、2、3、4 行
for k in range(1, 5):
    sub = reqs[:k]
    d = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                          model=m, tokenizer=tok, prompt_source=src)
    d.load()
    got = run_driver_greedy(d, sub, src, GEN)
    d.retire([r.req_id for r in sub])
    marks = {r.req_id: (got[r.req_id] == refs[r.req_id]) for r in sub}
    print(f"k={k} lens={str(LENGTHS[:k]):<18} rows_ok={marks}")
    if not all(marks.values()):
        for r in sub:
            if not marks[r.req_id]:
                print(f"   MISMATCH {r.req_id} (len={r.prompt_len})")
                print(f"     ref={refs[r.req_id]}")
                print(f"     drv={got[r.req_id]}")
