from __future__ import annotations

import math
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


TRITON_LAUNCH_GRID_NDIM = 3


def is_z_prefix(prefix: str) -> bool:
    return prefix.startswith("z") and prefix.endswith("_") and prefix[1:-1].isdigit()


def is_reduction_prefix(prefix: str) -> bool:
    return prefix.startswith("r") and prefix.endswith("_") and prefix[1:-1].isdigit()


def reduction_prefixes(ndim: int) -> tuple[str, ...]:
    if ndim < 0:
        raise ValueError(f"Invalid reduction rank: {ndim}")
    return tuple(f"r{i}_" for i in range(ndim))


def get_reduction_prefixes(prefixes: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            (p for p in prefixes if is_reduction_prefix(p)), key=lambda p: int(p[1:-1])
        )
    )


def get_z_prefixes(prefixes: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted((p for p in prefixes if is_z_prefix(p)), key=lambda p: int(p[1:-1]))
    )


def pointwise_prefixes(ndim: int) -> tuple[str, ...]:
    if ndim < 0:
        raise ValueError(f"Invalid pointwise rank: {ndim}")
    if ndim == 0:
        return ()
    z_prefixes = tuple(f"z{i}_" for i in reversed(range(max(0, ndim - 2))))
    return z_prefixes + ("y", "x")[-ndim:]


def get_pointwise_prefixes(prefixes: Iterable[str]) -> tuple[str, ...]:
    prefixes = tuple(prefixes)
    return tuple(reversed(get_z_prefixes(prefixes))) + tuple(
        p for p in ("y", "x") if p in prefixes
    )


def get_pointwise_block_product(
    block_sizes: Mapping[str, int], *, exclude: str | None = None
) -> int:
    return math.prod(
        block_sizes[p] for p in get_pointwise_prefixes(block_sizes) if p != exclude
    )


def get_max_block(prefix: str, max_blocks: Mapping[str, int]) -> int:
    key = "Z0_" if is_z_prefix(prefix.lower()) else prefix.upper()
    if is_reduction_prefix(prefix.lower()) and key not in max_blocks:
        key = "R1_"
    return max_blocks[key]


def block_to_prefix(block: str) -> str:
    return block.removesuffix("BLOCK").lower()


def block_sort_key(block: str) -> tuple[int, int | str]:
    prefix = block_to_prefix(block)
    if is_reduction_prefix(prefix):
        return (0, int(prefix[1:-1]))
    if prefix in ("x", "y"):
        return (1, int(prefix == "y"))
    if is_z_prefix(prefix):
        return (1, int(prefix[1:-1]) + 2)
    return (2, block)
