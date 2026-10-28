"""同一配置重复跑，看 V3 失败是否稳定复现 —— 区分"确定性缺陷"与"抖动"。"""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "benchmark"); sys.path.insert(0, "scripts")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ascend_opt import Request
from real_backend import TransformerDriver, SyntheticPromptSource
from validate_fp16_path import reference_greedy, run_driver_greedy

MODEL = "Qwen/Qwen2.5-7B-Instruct"; dev = "musa:0"; GEN = 16
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
excl = [v for n in ("eos_token_id","pad_token_id","bos_token_id","unk_token_id")
        if isinstance((v := getattr(tok, n, None)), int)]
src = SyntheticPromptSource(vocab_size=int(m.config.vocab_size), seed=1234, excluded_ids=excl)

L = [16, 24, 33, 48]
reqs = [Request(req_id=f"v3_{i}", prompt_len=n, gen_len=GEN) for i, n in enumerate(L)]
refs = {r.req_id: reference_greedy(m, tok, src.prompt_ids(r), GEN, dev) for r in reqs}

# 先验证参考实现自身是否可重复
r0 = reference_greedy(m, tok, src.prompt_ids(reqs[0]), GEN, dev)
r0b = reference_greedy(m, tok, src.prompt_ids(reqs[0]), GEN, dev)
print("参考实现重复一致:", r0 == r0b)

N = 5
bad_counts = {}
print(f"\n重复 {N} 次 4 行批：")
for it in range(N):
    d = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                          model=m, tokenizer=tok, prompt_source=src)
    d.load()
    got = run_driver_greedy(d, reqs, src, GEN)
    d.retire([r.req_id for r in reqs])
    marks = {r.req_id: got[r.req_id] == refs[r.req_id] for r in reqs}
    for k, v in marks.items():
        if not v: bad_counts[k] = bad_counts.get(k, 0) + 1
    print(f"  run{it}: {marks}")
print("\n各行的失败次数:", bad_counts or "全部通过")
