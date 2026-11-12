# Owner(s): ["module: inductor"]

import itertools
import math
import sys
import types
import typing
import unittest

import sympy

from torch._inductor.codegen.simd import DerivedIterationRangesRoot, IterationRangesRoot
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import IndexingOptions, TritonKernel, TritonSymbols
from torch._inductor.runtime.tiling import (
    block_sort_key,
    block_to_prefix,
    get_max_block,
    get_pointwise_block_product,
    get_pointwise_prefixes,
    get_z_prefixes,
    pointwise_prefixes,
)
from torch._inductor.runtime.triton_heuristics import GridND
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.inductor_utils import MockGraphHandler
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv


try:
    import triton  # noqa: F401
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904


class TestSIMDRangeTrees(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_numbered_pointwise_axes(self):
        prefixes = ("z10_", "r0_", "y", "z2_", "x", "z0_")
        self.assertEqual(get_z_prefixes(prefixes), ("z0_", "z2_", "z10_"))
        self.assertEqual(
            get_pointwise_prefixes(prefixes), ("z10_", "z2_", "z0_", "y", "x")
        )
        self.assertEqual(pointwise_prefixes(5), ("z2_", "z1_", "z0_", "y", "x"))
        self.assertEqual(get_max_block("z10_", {"Z0_": 1024}), 1024)
        block_sizes = {"x": 8, "y": 4, "z0_": 2, "z10_": 16, "r0_": 32}
        self.assertEqual(get_pointwise_block_product(block_sizes), 1024)
        self.assertEqual(get_pointwise_block_product(block_sizes, exclude="x"), 128)
        self.assertEqual(get_pointwise_block_product(block_sizes, exclude="z10_"), 64)
        self.assertEqual(block_to_prefix("R1_BLOCK"), "r1_")
        self.assertEqual(block_to_prefix("Z10_BLOCK"), "z10_")
        self.assertEqual(
            sorted(["Z10_BLOCK", "XBLOCK", "Z2_BLOCK"], key=block_sort_key),
            ["XBLOCK", "Z2_BLOCK", "Z10_BLOCK"],
        )

    def _make_graph(self):
        return MockGraphHandler()

    @parametrize("ndim", [3, 4, 5])
    def test_nd_grid_decode_with_padding(self, ndim):
        numels = {"z0_": 3, "y": 7, "x": 5}
        blocks = {"Z0_BLOCK": 2, "YBLOCK": 2, "XBLOCK": 2}
        if ndim >= 4:
            numels = {"z1_": 2, **numels}
            blocks["Z1_BLOCK"] = 1
        if ndim == 5:
            numels = {"z2_": 2, **numels}
            blocks["Z2_BLOCK"] = 1
        grid = GridND(inductor_meta={"pointwise_prefixes": list(numels)})
        grid.generate(blocks)
        launch_grid = grid.eval_slow({f"{p}numel": n for p, n in numels.items()})
        self.assertEqual(launch_grid, (3, 4, 2 ** (ndim - 2)))

        with V.set_graph_handler(self._make_graph()):
            features = SIMDKernelFeatures(
                [], sympy.Integer(math.prod(numels.values())), sympy.Integer(1)
            )
            kernel = TritonKernel(
                {p: sympy.Integer(n) for p, n in numels.items()}
                | {"r0_": sympy.Integer(1)},
                features=features,
            )
            self.assertIs(kernel._get_grid_type(), GridND)
            expressions = [
                kernel.iteration_ranges_get_pid(tree)
                for tree in kernel.range_trees
                if not tree.is_reduction
            ]
        extents = tuple(
            (n + blocks[f"{p.upper()}BLOCK"] - 1) // blocks[f"{p.upper()}BLOCK"]
            for p, n in numels.items()
        )
        seen = []
        for program in itertools.product(*(range(n) for n in launch_grid)):
            tl = types.SimpleNamespace(
                program_id=lambda axis: program[axis],
                cdiv=lambda n, block: (n + block - 1) // block,
                int64=None,
                int32=None,
            )
            scope = {"tl": tl, **blocks, **{f"{p}numel": n for p, n in numels.items()}}
            coordinates = tuple(
                eval(
                    expr.replace(".to(tl.int64)", "").replace(".to(tl.int32)", ""),
                    scope,
                )
                for expr in expressions
            )
            self.assertEqual(coordinates[-2:], (program[1], program[0]))
            seen.append(coordinates)
        self.assertCountEqual(
            seen, list(itertools.product(*(range(n) for n in extents)))
        )

    def _make_kernel(self, *, persistent: bool):
        features = SIMDKernelFeatures([], sympy.Integer(4), sympy.Integer(512))
        return TritonKernel(
            {"x": sympy.Integer(4), "r0_": sympy.Integer(512)},
            features=features,
            override_persistent_reduction=persistent,
            override_cooperative_reduction=False,
        )

    def _make_parent_reduction_root(self):
        return IterationRangesRoot(
            "r0_index",
            sympy.Integer(512),
            "r0_",
            0,
            typing.cast(typing.Any, object()),
            is_loop=True,
            tensor_dim=1,
            grid_dim=None,
            has_zdim=False,
        )

    def _make_derived_root(self, r_tree, *, group_size=sympy.Integer(128)):
        reduced_block = FloorDiv(r_tree.block_size(), group_size)
        reduced_numel = FloorDiv(r_tree.numel, group_size)
        return DerivedIterationRangesRoot(
            r_tree,
            numel=reduced_numel,
            block_size=reduced_block,
            block_offset=FloorDiv(r_tree.block_offset(), group_size),
            named_constants=(
                (sympy.Symbol("nested_R0_GROUP_SIZE"), group_size, True),
                (sympy.Symbol("nested_R0_REDUCED_BLOCK"), reduced_block, True),
                (sympy.Symbol("nested_R0_REDUCED_NUMEL"), reduced_numel, False),
            ),
        )

    def test_derived_root_geometry(self):
        r_tree = self._make_parent_reduction_root()
        derived = self._make_derived_root(r_tree)

        self.assertEqual(derived.is_loop, r_tree.is_loop)
        self.assertEqual(derived.prefix, r_tree.prefix)
        self.assertEqual(derived.numel, FloorDiv(r_tree.numel, 128))
        self.assertEqual(derived.block_size(), FloorDiv(r_tree.block_size(), 128))
        self.assertEqual(derived.block_offset(), FloorDiv(r_tree.block_offset(), 128))
        self.assertTrue(derived.owns_mask(derived.mask_name()))
        self.assertFalse(derived.owns_mask(r_tree.mask_name()))
        self.assertNotEqual(derived.mask_name(), r_tree.mask_name())
        self.assertFalse(derived.supports_constant_mask())

    def test_derived_root_uses_active_range_tree_geometry(self):
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            kernel = self._make_kernel(persistent=False)
            x_tree, r_tree = kernel.range_trees
            derived = self._make_derived_root(r_tree)

            self.assertFalse(kernel._has_constant_mask(derived))

            with V.set_kernel_handler(kernel):
                x_entry = x_tree.full_range()
                derived_entry = derived.full_range()
                index = x_entry.symbol() * derived.numel + derived_entry.symbol()
                self.assertTrue(kernel.is_broadcasted(index))

                indexing = IndexingOptions(
                    index_str="",
                    mask_vars=OrderedSet([derived.mask_name()]),
                    expand_str=None,
                    _has_rindex=True,
                    index=index,
                    expand_shape=None,
                )
                self.assertFalse(indexing.has_rmask())

                with kernel.use_range_trees([x_tree, derived]):
                    self.assertFalse(kernel.is_broadcasted(index))
                    self.assertEqual(
                        TritonSymbols.get_block_shape(index),
                        ("XBLOCK", "(R0_BLOCK//128)"),
                    )
                    self.assertTrue(indexing.has_rmask())

                self.assertEqual(kernel.range_trees, [x_tree, r_tree])
                self.assertFalse(indexing.has_rmask())

    def test_use_range_trees_clears_simplify_indexing_cache(self):
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            kernel = self._make_kernel(persistent=False)
            x_tree, r_tree = kernel.range_trees
            derived = self._make_derived_root(r_tree)

            with V.set_kernel_handler(kernel):
                expr = x_tree.full_range().symbol() + r_tree.full_range().symbol()
                kernel.simplify_indexing(expr)
                self.assertGreater(kernel.simplify_indexing.cache_info().currsize, 0)

                with kernel.use_range_trees([x_tree, derived]):
                    self.assertEqual(kernel.simplify_indexing.cache_info().currsize, 0)
                    kernel.simplify_indexing(expr)
                    self.assertGreater(
                        kernel.simplify_indexing.cache_info().currsize, 0
                    )

                self.assertEqual(kernel.simplify_indexing.cache_info().currsize, 0)
                self.assertEqual(kernel.range_trees, [x_tree, r_tree])


instantiate_parametrized_tests(TestSIMDRangeTrees)


if __name__ == "__main__":
    run_tests()
