"""实验分组的**唯一定义**：什么算「同一个实验」。

汇总（``benchmark/collect_metrics.py``）、绘图（``benchmark/plot_p99.py``）与
导出校验（``benchmark/preflight.py``）三处必须用同一套判据。三处各写一份的后果
不是"难看"，而是**三者会悄悄分叉**。已经发生过：

* 汇总按 ``(运行批次, 负载, 配置, 硬件平台, 执行后端)`` 拆组，绘图却只按
  ``(负载, 配置, 硬件平台)`` 分组 —— 同一平台上两个后端的数字被并进同一根柱；
* 导出校验又只认 ``run_id`` 前缀，于是把已经被拆开的两组当成一组重新数成员，
  报出并不存在的"重复次数不一致"。

这三处**都不报错**，只是安静地给出错误的表/图/结论。把它们收敛到一个模块，
是"分组规则只有一份实现"的落地方式。

方向刻意选"宁可拆开、不要合并"：拆错会让一组落到重复次数不足而被看见；
并错只会得到一张看起来完全正常的表。

本模块是**纯函数**，不依赖 matplotlib、不碰文件系统，因此三个脚本与单元测试
可以共用同一实现，不需要各自再抄一份。
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "GROUP_DIMENSIONS",
    "REPEAT_SUFFIXES",
    "base_run_id",
    "group_dimensions",
    "group_key",
]

#: 进入分组键的维度，顺序即 ``group_key`` 串里的顺序。
#:
#: ``hardware_platform`` 与 ``backend`` 必须进键，理由是同一个失效模式：
#: 合并**不会报错**，只会产出一个"谁都没测过"的中位数。这一点已被复现——
#: 同负载同配置下，两个平台上的 TEST_DATA（100 与 200）被合成了一根值为 150
#: 的柱。后端同理：模拟的数字来自一阶设备模型，与真机口径不同，
#: 合起来算离散度毫无意义。
GROUP_DIMENSIONS: tuple[str, ...] = (
    "workload",
    "config",
    "hardware_platform",
    "backend",
)

#: 重复运行的 run_id 后缀。与 :func:`base_run_id` 同一口径；
#: 导出校验曾在这里另起一套（按 ``startswith`` 判前缀），两套规则迟早不一致。
REPEAT_SUFFIXES: tuple[str, ...] = tuple(f"-r{i}" for i in range(1, 10))

#: 维度缺值时的占位。刻意**不用空串**：空串在 CSV 里与"这一列不存在"看起来一样，
#: 而"没有这一列"和"这一列是空的"下游命运不同。
_UNKNOWN = "unknown"


def _cell(row: Mapping[str, Any], key: str, default: str = _UNKNOWN) -> str:
    """取一个标量维度值并压成字符串。缺值/空值统一落到 ``default``。"""
    value = row.get(key)
    if value is None:
        return default
    return str(value).strip() or default


def base_run_id(run_id: str) -> str:
    """把重复运行的 run_id 还原成它的批次名（去掉 ``-rN`` 后缀）。

    只去一层后缀，且只认 :data:`REPEAT_SUFFIXES` 里列出的形状：
    ``cmp-a-r1-r2`` 这种不像本项目会产生的名字，宁可原样返回让它自然落单，
    也不要猜出一层不存在的父级。
    """
    text = str(run_id or "")
    for suffix in REPEAT_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def group_dimensions(row: Mapping[str, Any]) -> tuple[str, ...]:
    """分组键里**除 run_id 之外**的维度取值。"""
    return tuple(_cell(row, name) for name in GROUP_DIMENSIONS)


def group_key(row: Mapping[str, Any]) -> str:
    """规范分组键：``<run_id_base>|<维度...>``，维度顺序同 :data:`GROUP_DIMENSIONS`。

    返回字符串而不是元组，是因为它要被写进 ``repeats.csv`` 的 ``group`` 列——
    那一列得能被 diff、能被 grep，且在 Excel 里一眼看出两组为何不同。

    只有**汇总层**（跨重复聚合）才带 ``run_id_base`` 前缀：那里必须知道
    "哪几行属于同一个批次"，成员清单也是按它落盘的。汇总层以下的绘图按
    :func:`group_dimensions` 分组——它逐行聚合分位数，本就不需要批次身份，
    多带一维会把"同一实验的几次重复"拆成几根柱。
    """
    base = base_run_id(str(row.get("run_id", "")))
    return "|".join((base, *group_dimensions(row)))
