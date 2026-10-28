"""Qwen 单请求生成验证 v2（在 MTT S4000 / MUSA 上）。

回答三个问题，每个都要有**可复核的数字或标记**，不靠"跑通了"这种说法：
  1. 计算真的发生在设备上吗？—— 参数所在 device、显存增量、首次前向确实在 musa 上执行。
  2. 输出正确吗？—— ①贪心解码可复现（seed 0 跑 N 次结果逐 token 相同）；②可选的强校验：
     同样 fp32 权重下 MUSA 与 CPU 的**首个前向 logits** 最大绝对偏差（只看第一步，避免生成误差累积）。
  3. 显存占用多少？—— 各阶段 mem_get_info 差值与设备侧峰值分配。

v2 相对 v1 的改动（v1 的产出见 results/qwen_0.5b.json，**勿与新结果混用同一张表**）：
  1. **分离载入阶段**：tokenizer / 模型构造 / 搬到设备 分别计时。v1 只报一个 load_s——
     0.5B 竟用了 215.74s，一个数无法判断瓶颈在磁盘、在反序列化、还是在设备上下文初始化。
  2. **预热移出计时窗口**：v1 没做预热，结果两次 generate 是 4.105s / 0.102s（差 40 倍），
     那 4.1s 是首次调用开销，被当成"这一模型的时延"会误导。v2 先跑一次不计入测量的 warmup；
     **但冷的第一次照样单独记录**——"首次调用贵出 40 倍"本身就是必须报出来的事实。
  3. **逐 token 时间戳**：v1 只有总时长；本轮要的是 TTFT 与每 token 间隔（ITL），
     才能算 TPOT 与 stall（单请求间隔 P99）。做法是在 generate 里挂一个 LogitsProcessor，
     它在**每一步 forward 结束后**被调用，于是可以直接记录该步完成的时刻。
  4. **每步 synchronize**：不同步的话记到的是"算子下发时刻"而不是"算完时刻"，
     在异步执行的设备上这两者差着整个流水线深度，加起来的"总时长"是假的。
  5. **重复 3 次**：项目纪律是重复 <3 不出结论，故计时重复 3 次并报每次与中位数。

用法：
    python qwen_single_request.py --model Qwen/Qwen2.5-0.5B-Instruct --compare-cpu
    python qwen_single_request.py --model Qwen/Qwen2.5-7B-Instruct
    python qwen_single_request.py --model Qwen/Qwen2.5-7B-Instruct --compare-cpu
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import traceback

import torch

RESULT: dict = {}

#: 请求了却**没能产出**的测量环节。留空即"所有请求的测量都拿到了数"。
#:
#: 为什么要有这个字段（而不是靠 try/except 把异常吞进结果里就完事）：
#: 上一版就是这么写的，于是 fp32 对照因 `NameError` 整块失败，产物里只有
#: `{"error": ...}`、进程**退出码仍是 0**，链式脚本于是打印"全部完成"。
#: 这正是本项目反复踩的那类缺陷——产物看起来正常，实际是一个测量环节没产出，
#: 而且它不会报错、不会留痕，只会让"输出正确性已验证"这句话失去依据。
#: 所以：局部 catch 保留（先跑完的 fp16 数据不能因为对照崩了就丢弃），
#: 但**必须**在顶层显形 + 用非零退出码让外层看得见。
FAILED_BLOCKS: list[str] = []


def _fail(block: str, exc: BaseException) -> None:
    """记录一个失败的测量环节，并把异常如实打出来。"""
    FAILED_BLOCKS.append(block)
    print(f"[FAILED_BLOCK] {block}: {type(exc).__name__}: {exc}", file=sys.stderr)
    traceback.print_exc()


def mem() -> dict:
    free, total = torch.musa.mem_get_info()
    return {"free_mib": round(free / 2**20, 1), "total_mib": round(total / 2**20, 1)}


def percentile(values: list[float], q: float) -> float | None:
    """线性插值分位。空样本返回 ``None``——不许用 0 冒充"没数据"。

    与 ``ascend_opt.metrics.percentile`` 同法，但这里刻意不 import 项目包：
    本脚本是**设备侧探针**，一旦依赖项目代码，探针失败就分不清是设备的问题
    还是项目代码的问题。
    """
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


class StepRecorder:
    """LogitsProcessor：记录每一步 forward **完成后**的时刻。

    generate 的调用序列是：
        step 0  = 处理整段 prompt（prefill），产出第 1 个生成 token 的 logits
        step k>0 = 处理上一步产出的 1 个 token（decode，KV cache 命中）
    所以时间戳序列的**第一个**≈ 首 token 时刻（TTFT），相邻差 = token 间隔（ITL）。

    为什么在 __call__ 里 synchronize：本函数在 forward 返回后才被调用，但设备上的
    计算是异步的——不同步就只记到"下发时刻"。同步开销在此可忽略（前向已经算完，
    只是等队尾排空）。
    """

    def __init__(self) -> None:
        self.times: list[float] = []

    def reset(self) -> None:
        self.times = []

    def __call__(self, input_ids, scores):  # noqa: ANN001, ANN201 - 接口由 transformers 定
        torch.musa.synchronize()
        self.times.append(time.perf_counter())
        return scores


def _timed_phases(args) -> dict:
    """分阶段载入，把"载入慢"拆成可归因的三段。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    phases: dict[str, float] = {}

    t = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    phases["tokenizer_s"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    phases["from_pretrained_s"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    model = model.to("musa").eval()
    phases["to_device_s"] = round(time.perf_counter() - t, 2)

    phases["total_s"] = round(sum(phases.values()), 2)
    return {"phases": phases, "tokenizer": tokenizer, "model": model}


def _generate_timed(model, inputs, recorder, max_new_tokens: int, min_new_tokens: int):
    """一次贪心生成，返回 (输出 ids, 开始时刻, 结束时刻)。

    刻意把 ``recorder`` 走官方的 ``logits_processor`` 通道，而不是自己写解码循环：
    自写循环要自己管 KV cache 与位置编码，那部分一旦写错，"数字不对"就分不清是
    设备的问题还是循环的问题；而本脚本的第一要务是**把设备测对**。
    代价是时间戳落在"每步 forward 结束"而不是"token 真正吐给客户端"——对单请求
    同步测量而言两者只差一次采样开销，且这个差值本身就是我们要报的量级之一。
    """
    from transformers import LogitsProcessorList

    torch.manual_seed(0)
    recorder.reset()
    t_start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            logits_processor=LogitsProcessorList([recorder]),
        )
    torch.musa.synchronize()
    return out, t_start, time.perf_counter()


def main() -> int:  # noqa: C901 - 线性流程，拆函数反而更难跟着读
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--prompt",
        default="List the first eight prime numbers, comma separated.",
        help="默认取一个会产生足量解码步的 prompt，否则总时长无法拆出 token 间隔",
    )
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument(
        "--min-new-tokens",
        type=int,
        default=16,
        help="强制最少解码步数；不设时模型经常一两个 token 就 EOS，ITL 样本为 0",
    )
    ap.add_argument("--repeats", type=int, default=3, help="计时重复次数，项目纪律 <3 不出结论")
    ap.add_argument("--compare-cpu", action="store_true", help="额外做 fp32 logits 的 CPU 对照")
    ap.add_argument("--out", default=None, help="把报告写到该 JSON 路径")
    args = ap.parse_args()

    import transformers

    RESULT["model"] = args.model
    RESULT["prompt"] = args.prompt
    RESULT["script_version"] = "v2"
    RESULT["env"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_musa": getattr(__import__("torch_musa"), "__version__", "?"),
        "transformers": transformers.__version__,
        "device_name": torch.musa.get_device_name(0),
        "device_count": torch.musa.device_count(),
        "musa_available": torch.musa.is_available(),
    }
    RESULT["mem_before"] = mem()

    # ---------- 1) 载入（分阶段） ----------
    loaded = _timed_phases(args)
    tokenizer, model = loaded["tokenizer"], loaded["model"]
    RESULT["load"] = loaded["phases"]

    first_param = next(model.parameters())
    RESULT["device"] = {
        "param_device": str(first_param.device),
        "param_dtype": str(first_param.dtype),
        "n_params": sum(p.numel() for p in model.parameters()),
        "all_params_on_musa": all(p.device.type == "musa" for p in model.parameters()),
    }
    RESULT["mem_after_load"] = mem()
    RESULT["mem_load_delta_mib"] = round(
        RESULT["mem_before"]["free_mib"] - RESULT["mem_after_load"]["free_mib"], 1
    )

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("musa")
    RESULT["prompt_tokens"] = int(inputs["input_ids"].shape[1])

    recorder = StepRecorder()

    # ---------- 2) 预热：**不计入测量**，但如实记录 ----------
    # 为什么"冷启动那一次"必须单独留数字：它贵的量级可能是两个数量级（v1 实测 40 倍），
    # 于是它既是"必须预热"的证据，也是"服务端首请求体验"的真实数据。丢掉就只剩结论、没有依据。
    warm_out, warm_t0, warm_t1 = _generate_timed(
        model, inputs, recorder, max_new_tokens=8, min_new_tokens=4
    )
    RESULT["warmup"] = {
        "wall_s": round(warm_t1 - warm_t0, 3),
        "note": "冷启动单次调用，含首次 kernel/JIT/内存池开销；不计入下方测量值",
        "n_steps": len(recorder.times),
    }
    RESULT["mem_after_warmup"] = mem()

    # ---------- 3) 定时重复（预热之后，进计时窗口） ----------
    runs = []
    for i in range(args.repeats):
        out, t0, t1 = _generate_timed(
            model,
            inputs,
            recorder,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
        )
        prompt_len = int(inputs["input_ids"].shape[1])
        new_ids = out[0][prompt_len:]
        times = list(recorder.times)
        ttft_ms = round((times[0] - t0) * 1000, 3) if times else None
        itl_ms = [round((b - a) * 1000, 3) for a, b in zip(times, times[1:])]
        runs.append(
            {
                "i": i,
                "wall_s": round(t1 - t0, 3),
                "n_new_tokens": int(new_ids.shape[0]),
                "n_steps": len(times),
                "ttft_ms": ttft_ms,
                "itl_ms": itl_ms,
                "tpot_ms": round(statistics.fmean(itl_ms), 3) if itl_ms else None,
                "stall_p99_ms": (round(percentile(itl_ms, 99), 3) if itl_ms else None),
                "itl_max_ms": round(max(itl_ms), 3) if itl_ms else None,
                "text": tokenizer.decode(new_ids, skip_special_tokens=True),
                "ids": [int(x) for x in new_ids],
            }
        )

    RESULT["generation"] = {
        "repeats": args.repeats,
        "runs": runs,
        # 可复现性：逐 token id 完全相同，而不是"文本看起来一样"
        "deterministic": len({tuple(r["ids"]) for r in runs}) == 1,
        # 分位只统计成功请求；这里全部成功，故 n 全等于 runs 数
        "wall_s_median": round(
            statistics.median([r["wall_s"] for r in runs]), 3
        ),
        "ttft_ms_median": round(
            statistics.median([r["ttft_ms"] for r in runs if r["ttft_ms"] is not None]), 3
        ) if any(r["ttft_ms"] is not None for r in runs) else None,
        "tpot_ms_median": round(
            statistics.median([r["tpot_ms"] for r in runs if r["tpot_ms"] is not None]), 3
        ) if any(r["tpot_ms"] is not None for r in runs) else None,
    }
    RESULT["mem_after_generate"] = mem()
    RESULT["mem_generate_delta_mib"] = round(
        RESULT["mem_after_load"]["free_mib"] - RESULT["mem_after_generate"]["free_mib"], 1
    )

    peak = getattr(torch.musa, "max_memory_allocated", None)
    if callable(peak):
        RESULT["peak_allocated_mib"] = round(peak() / 2**20, 1)

    # ---------- 4) 强校验：fp32 首前向 logits，MUSA vs CPU ----------
    if args.compare_cpu:
        try:
            # 这个 import 必须在本块内、不能靠 _timed_phases 里的那次局部导入：
            # 局部导入随函数返回即失效，本块在 main() 作用域里用不到它。
            # 上一版就是漏了这句，整块对照以 NameError 告终（详见 FAILED_BLOCKS 的注释）。
            from transformers import AutoModelForCausalLM

            del model
            torch.musa.empty_cache()
            cpu_inputs = {k: v.to("cpu") for k, v in inputs.items()}
            ref = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float32
            ).to("cpu").eval()
            with torch.no_grad():
                cpu_logits = ref(**cpu_inputs).logits
            del ref
            m32 = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float32
            ).to("musa").eval()
            with torch.no_grad():
                musa_logits = m32(**inputs).logits
            torch.musa.synchronize()
            diff = (cpu_logits - musa_logits.to("cpu")).abs()
            RESULT["correctness_fp32_first_forward"] = {
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "cpu_argmax_last": int(cpu_logits[0, -1].argmax()),
                "musa_argmax_last": int(musa_logits[0, -1].argmax()),
                "argmax_match": bool(
                    cpu_logits[0, -1].argmax() == musa_logits[0, -1].argmax().to("cpu")
                ),
                "shape": list(cpu_logits.shape),
            }
        except Exception as exc:  # noqa: BLE001 - 保留已跑完的数据，但失败必须显形（见 _fail）
            RESULT["correctness_fp32_first_forward"] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
            _fail("correctness_fp32_first_forward", exc)

    RESULT["failed_blocks"] = list(FAILED_BLOCKS)
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(RESULT, fh, ensure_ascii=False, indent=2)
        print(f"\n[written] {args.out}")
    if FAILED_BLOCKS:
        print(
            f"\n[不完整] 有 {len(FAILED_BLOCKS)} 个请求的测量环节没有产出："
            f"{FAILED_BLOCKS}。本次运行**不得**记为「验证通过」。",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
