"""决定性判据：这个"失败"是驱动错，还是**参考实现本身在长生成下不稳定**？

v3_0 的输出是"数数"（\ndef test_001():...assert 1 == 1）。
参考与驱动前 4 个 token 完全一致，之后错开一位 —— 典型是**某一步 logits 接近平局**，
不同的归约顺序就翻开不同结果。
用 fp32 参考重跑同一 prompt，看参考自己的 token 序列是否与 fp16 参考一致。
"""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "benchmark"); sys.path.insert(0, "scripts")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ascend_opt import Request
from real_backend import TransformerDriver, SyntheticPromptSource
from validate_fp16_path import reference_greedy, run_driver_greedy

MODEL = "Qwen/Qwen2.5-7B-Instruct"; dev = "musa:0"; GEN = 16
tok = AutoTokenizer.from_pretrained(MODEL)
excl = [v for n in ("eos_token_id","pad_token_id","bos_token_id","unk_token_id")
        if isinstance((v := getattr(tok, n, None)), int)]

m16 = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
src = SyntheticPromptSource(vocab_size=int(m16.config.vocab_size), seed=1234, excluded_ids=excl)
req0 = Request(req_id="v3_0", prompt_len=16, gen_len=GEN)

ref16 = reference_greedy(m16, tok, src.prompt_ids(req0), GEN, dev)
print("ref fp16 :", ref16)

d16 = TransformerDriver(model_name=MODEL, device=dev, precision="fp16",
                        model=m16, tokenizer=tok, prompt_source=src)
d16.load()
drv16 = run_driver_greedy(d16, [req0], src, GEN)["v3_0"]
d16.retire(["v3_0"])
print("drv fp16 :", drv16)
print("单跑一致 :", ref16 == drv16)

# 参考重复性：连跑 3 次
print("\n参考 fp16 连跑 3 次:")
for i in range(3):
    print("  ", reference_greedy(m16, tok, src.prompt_ids(req0), GEN, dev))

# fp32 参考：若与 fp16 参考不同，说明该 prompt 的分岔点本身对精度敏感
m32 = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev).eval()
ref32 = reference_greedy(m32, tok, src.prompt_ids(req0), GEN, dev)
print("\nref fp32 :", ref32)
print("fp16 vs fp32 参考一致:", ref16 == ref32)
