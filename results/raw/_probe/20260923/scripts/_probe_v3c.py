"""定位 V3 的精确触发条件：16+48 过、16+24 挂 —— 差在哪？

扫所有两两组合，并打印每行的 prompt 最后一个 token（看是不是 prompt 内容/长度
的某种性质导致，而不是"批"本身）。
"""
import sys, itertools
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
reqs = {n: Request(req_id=f"L{n}", prompt_len=n, gen_len=GEN) for n in L}
refs = {n: reference_greedy(m, tok, src.prompt_ids(reqs[n]), GEN, dev) for n in L}

print("== 单跑自证（每行单独跑必须与参考一致）==")
for n in L:
    d = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                          model=m, tokenizer=tok, prompt_source=src)
    d.load()
    g = run_driver_greedy(d, [reqs[n]], src, GEN)[f"L{n}"]
    d.retire([f"L{n}"])
    print(f"  len={n:<3} single_ok={g == refs[n]}")

print("\n== 两两组合 ==")
for a, b in itertools.combinations(L, 2):
    d = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                          model=m, tokenizer=tok, prompt_source=src)
    d.load()
    got = run_driver_greedy(d, [reqs[a], reqs[b]], src, GEN)
    d.retire([f"L{a}", f"L{b}"])
    ok_a = got[f"L{a}"] == refs[a]; ok_b = got[f"L{b}"] == refs[b]
    print(f"  ({a:>2},{b:>2}) ok_a={ok_a} ok_b={ok_b}   {'<-- 短行坏' if (not ok_a and a < b) else ''}")

print("\n== 各行 prompt 的尾部 token（判断是否与内容有关）==")
for n in L:
    ids = src.prompt_ids(reqs[n])
    print(f"  len={n:<3} head={ids[:4]} tail={ids[-4:]} last={ids[-1]}")
