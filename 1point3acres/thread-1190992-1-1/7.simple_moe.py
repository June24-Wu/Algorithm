"""Simplified Mixture-of-Experts assignment using NumPy.

Goal
----
Implement only ``SimpleMoE.forward``. The method must perform the complete MoE
calculation:

1. Compute router logits for every token.
2. Apply a numerically stable softmax over experts.
3. Select the top-k experts for every token.
4. Renormalize the selected probabilities into combine weights.
5. Run only the selected experts.
6. Combine selected expert outputs with their router weights.
7. Count how many routing slots each expert receives.
8. Compute a load-balancing auxiliary loss.

This assignment uses one linear layer per expert so the routing and combining
math stays visible. Real MoE models commonly use an MLP for each expert.

Shapes
------
    x:                    (B, T, D)
    tokens:               (N, D), where N = B * T
    router_logits:        (N, E)
    router_probabilities: (N, E)
    selected_experts:     (N, K)
    combine_weights:      (N, K)
    expert_counts:        (E,)
    output:               (B, T, D)

    B = batch size
    T = sequence length
    D = model dimension
    N = number of tokens
    E = number of experts
    K = number of selected experts per token

Equations
---------
For token ``x_n`` and expert ``e``:

    router_logits[n] = x_n @ router_W + router_b
    router_probs[n] = softmax(router_logits[n])
    selected[n] = top_k(router_probs[n], K)

The selected probabilities must be renormalized because discarded experts no
longer participate:

    combine_weight[n, slot]
        = selected_probability[n, slot]
          / sum(selected_probabilities[n])

Each linear expert computes:

    expert_e(x_n) = x_n @ expert_W[e] + expert_b[e]

The final token output is:

    output[n] = sum(
        combine_weight[n, slot] * expert_selected[n, slot](x_n)
    )

Load balancing
--------------
Without an additional training signal, a router may send almost every token
to one expert while other experts remain idle. This assignment returns the
standard differentiable-style auxiliary statistic:

    importance[e] = mean(router_probs[:, e])
    load[e] = expert_counts[e] / (N * K)
    load_balance_loss = E * sum(importance[e] * load[e])

Balanced routing gives a value near 1. Collapsed top-1 routing gives a value
near the number of experts. During training, this value is multiplied by a
small coefficient and added to the main task loss. Production MoE systems may
also impose per-expert capacity limits, but token dropping is intentionally
outside this assignment.

Usage
-----
    python 7.simple_moe.py
    python 7.simple_moe.py --show-answer

Edit only the single student section inside ``forward``.
"""

from __future__ import annotations

import argparse
import base64
import sys
import zlib
from typing import NamedTuple

from numpy import reshape

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


class MoEResult(NamedTuple):
    """Observable values produced by one MoE forward pass."""

    output: np.ndarray
    router_probabilities: np.ndarray
    selected_experts: np.ndarray
    combine_weights: np.ndarray
    expert_counts: np.ndarray
    load_balance_loss: float


class SimpleMoE:
    """A sparse top-k Mixture-of-Experts layer with linear experts."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        seed: int = 0,
    ) -> None:
        if np is None:
            raise ModuleNotFoundError(
                "This assignment requires NumPy. Run: python -m pip install numpy"
            )
        if d_model <= 0 or num_experts <= 0:
            raise ValueError("d_model and num_experts must be positive")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.router_W = rng.normal(0.0, scale, size=(d_model, num_experts))
        self.router_b = np.zeros(num_experts)
        self.expert_W = rng.normal(
            0.0,
            scale,
            size=(num_experts, d_model, d_model),
        )
        self.expert_b = rng.normal(0.0, scale, size=(num_experts, d_model))

    def forward(self, x: np.ndarray) -> MoEResult:
        """Route tokens, combine selected experts, and return load statistics."""
        # ============== Student section: implement everything here ==============
        B, S, D = x.shape

        tokens = x.reshape(-1,self.d_model) # (N,D)
        token_count = tokens.shape[0]

        router_logits = tokens @ self.router_W + self.router_b # (N,D) @ (D,E) -> (N,E)

        def softmax(x):
            x = np.exp(x - np.max(x,axis=-1,keepdims=True))
            return x / np.sum(x,axis=-1,keepdims=True)
        router_prob = softmax(router_logits)

        top_k_experts = np.argsort(
            -router_prob,
            axis=-1,
            kind="stable"
        )[:,:self.top_k]

        top_k_prob = np.take_along_axis(
            router_prob,
            top_k_experts,
            axis=-1
        )
        top_k_prob = top_k_prob / np.sum(top_k_prob,axis=-1,keepdims=True)

        output_tokens = np.zeros_like(tokens,dtype=float)
        for expert_idx in range(self.num_experts):
            token_idx, slot_idx = np.where(top_k_experts == expert_idx)
            if token_idx.size == 0:
                continue
            expert_output = tokens[token_idx] @ self.expert_W[expert_idx] + self.expert_b[expert_idx]
            output_tokens[token_idx] += top_k_prob[token_idx,slot_idx,None] * expert_output

        expert_counts = np.bincount(top_k_experts.reshape(-1),minlength=self.num_experts)
        importance = np.mean(router_prob,axis=0)
        load = expert_counts / int(token_count * self.top_k)
        load_balance_loss = float(self.num_experts * np.sum(importance * load))

        return MoEResult(
            output=output_tokens.reshape(B,S,D),
            router_probabilities=router_prob.reshape(B,S,-1),
            selected_experts=top_k_experts.reshape(B,S,-1),
            combine_weights=top_k_prob.reshape(B,S,-1),
            expert_counts=expert_counts,
            load_balance_loss=load_balance_loss
        )
        # ============ End student section: implement everything here ============


def _assert_close(
    actual: np.ndarray | float,
    expected: np.ndarray | float,
    name: str,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-8,
) -> None:
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    if actual_array.shape != expected_array.shape:
        raise AssertionError(
            f"{name} has the wrong shape: expected {expected_array.shape}, "
            f"got {actual_array.shape}"
        )
    if not np.allclose(actual_array, expected_array, atol=atol, rtol=rtol):
        difference = np.abs(actual_array - expected_array)
        raise AssertionError(
            f"{name} has incorrect values: max absolute error={difference.max():.3e}"
        )


def _assert_integer_equal(
    actual: np.ndarray,
    expected: np.ndarray,
    name: str,
) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        raise AssertionError(
            f"{name} has the wrong shape: expected {expected_array.shape}, "
            f"got {actual_array.shape}"
        )
    if not np.array_equal(actual_array, expected_array):
        raise AssertionError(f"{name} has incorrect values")


def _reference_forward(moe: SimpleMoE, x: np.ndarray) -> MoEResult:
    """Independent loop-based oracle used only by the tests."""
    batch_size, sequence_length, _ = x.shape
    tokens = x.reshape(-1, moe.d_model)
    token_count = tokens.shape[0]

    router_logits = tokens @ moe.router_W + moe.router_b
    shifted_logits = router_logits - np.max(router_logits, axis=-1, keepdims=True)
    router_probabilities = np.exp(shifted_logits)
    router_probabilities /= np.sum(
        router_probabilities,
        axis=-1,
        keepdims=True,
    )

    selected_experts = np.argsort(
        -router_probabilities,
        axis=-1,
        kind="stable",
    )[:, : moe.top_k]
    selected_probabilities = np.take_along_axis(
        router_probabilities,
        selected_experts,
        axis=-1,
    )
    combine_weights = selected_probabilities / np.sum(
        selected_probabilities,
        axis=-1,
        keepdims=True,
    )

    output_tokens = np.zeros_like(tokens, dtype=float)
    expert_counts = np.zeros(moe.num_experts, dtype=int)
    for token_index in range(token_count):
        for slot_index in range(moe.top_k):
            expert_index = selected_experts[token_index, slot_index]
            expert_counts[expert_index] += 1
            expert_output = (
                tokens[token_index] @ moe.expert_W[expert_index]
                + moe.expert_b[expert_index]
            )
            output_tokens[token_index] += (
                combine_weights[token_index, slot_index] * expert_output
            )

    importance = np.mean(router_probabilities, axis=0)
    load = expert_counts / float(token_count * moe.top_k)
    load_balance_loss = float(moe.num_experts * np.sum(importance * load))

    return MoEResult(
        output=output_tokens.reshape(batch_size, sequence_length, moe.d_model),
        router_probabilities=router_probabilities.reshape(
            batch_size,
            sequence_length,
            moe.num_experts,
        ),
        selected_experts=selected_experts.reshape(
            batch_size,
            sequence_length,
            moe.top_k,
        ),
        combine_weights=combine_weights.reshape(
            batch_size,
            sequence_length,
            moe.top_k,
        ),
        expert_counts=expert_counts,
        load_balance_loss=load_balance_loss,
    )


def _assert_matches_reference(moe: SimpleMoE, x: np.ndarray, name: str) -> None:
    expected = _reference_forward(moe, x)
    actual = moe.forward(x)
    _assert_close(actual.output, expected.output, f"{name} output")
    _assert_close(
        actual.router_probabilities,
        expected.router_probabilities,
        f"{name} router probabilities",
    )
    _assert_integer_equal(
        actual.selected_experts,
        expected.selected_experts,
        f"{name} selected experts",
    )
    _assert_close(
        actual.combine_weights,
        expected.combine_weights,
        f"{name} combine weights",
    )
    _assert_integer_equal(
        actual.expert_counts,
        expected.expert_counts,
        f"{name} expert counts",
    )
    _assert_close(
        actual.load_balance_loss,
        expected.load_balance_loss,
        f"{name} load balance loss",
    )


def _test_top1_exact_routing_and_output() -> None:
    moe = SimpleMoE(d_model=2, num_experts=2, top_k=1, seed=1)
    moe.router_W[:] = np.eye(2)
    moe.router_b[:] = 0.0
    moe.expert_W[0] = 2.0 * np.eye(2)
    moe.expert_W[1] = -np.eye(2)
    moe.expert_b[:] = 0.0
    x = np.array([[[4.0, 0.0], [0.0, 3.0], [2.0, -1.0]]])
    result = moe.forward(x)
    expected_experts = np.array([[[0], [1], [0]]])
    expected_output = np.array([[[8.0, 0.0], [0.0, -3.0], [4.0, -2.0]]])
    _assert_integer_equal(result.selected_experts, expected_experts, "top-1 experts")
    _assert_close(result.combine_weights, np.ones((1, 3, 1)), "top-1 weights")
    _assert_close(result.output, expected_output, "top-1 output")
    _assert_integer_equal(result.expert_counts, np.array([2, 1]), "top-1 counts")


def _test_top2_weighted_combination_exact_values() -> None:
    moe = SimpleMoE(d_model=2, num_experts=2, top_k=2, seed=3)
    moe.router_W[:] = 0.0
    moe.router_b[:] = np.log(np.array([0.75, 0.25]))
    moe.expert_W[0] = np.eye(2)
    moe.expert_W[1] = 3.0 * np.eye(2)
    moe.expert_b[:] = 0.0
    x = np.array([[[2.0, -4.0], [1.0, 6.0]]])
    result = moe.forward(x)
    expected_weights = np.array([[[0.75, 0.25], [0.75, 0.25]]])
    _assert_integer_equal(
        result.selected_experts,
        np.array([[[0, 1], [0, 1]]]),
        "top-2 experts",
    )
    _assert_close(result.combine_weights, expected_weights, "top-2 weights")
    _assert_close(result.output, 1.5 * x, "top-2 combined output")
    _assert_integer_equal(result.expert_counts, np.array([2, 2]), "top-2 counts")


def _test_random_top1_numerical_values() -> None:
    moe = SimpleMoE(d_model=6, num_experts=4, top_k=1, seed=5)
    x = np.random.default_rng(7).normal(size=(2, 5, 6))
    _assert_matches_reference(moe, x, "random top-1")


def _test_random_top2_numerical_values() -> None:
    moe = SimpleMoE(d_model=8, num_experts=5, top_k=2, seed=11)
    x = np.random.default_rng(13).normal(size=(3, 4, 8))
    _assert_matches_reference(moe, x, "random top-2")


def _test_random_top3_numerical_values() -> None:
    moe = SimpleMoE(d_model=4, num_experts=6, top_k=3, seed=17)
    x = np.random.default_rng(19).normal(size=(1, 7, 4))
    _assert_matches_reference(moe, x, "random top-3")


def _test_unselected_experts_do_not_contribute() -> None:
    moe = SimpleMoE(d_model=3, num_experts=3, top_k=1, seed=23)
    moe.router_W[:] = 0.0
    moe.router_b[:] = np.array([20.0, 0.0, -20.0])
    moe.expert_W[0] = np.eye(3)
    moe.expert_W[1] = 1e9 * np.eye(3)
    moe.expert_W[2] = -1e9 * np.eye(3)
    moe.expert_b[:] = 0.0
    x = np.random.default_rng(29).normal(size=(2, 3, 3))
    result = moe.forward(x)
    _assert_close(result.output, x, "unselected experts")
    _assert_integer_equal(result.expert_counts, np.array([6, 0, 0]), "selected count")


def _test_all_experts_matches_dense_mixture() -> None:
    moe = SimpleMoE(d_model=5, num_experts=4, top_k=4, seed=31)
    x = np.random.default_rng(37).normal(size=(2, 3, 5))
    result = moe.forward(x)
    tokens = x.reshape(-1, 5)
    dense = np.zeros_like(tokens)
    probabilities = result.router_probabilities.reshape(-1, 4)
    for expert_index in range(4):
        expert_output = tokens @ moe.expert_W[expert_index] + moe.expert_b[expert_index]
        dense += probabilities[:, expert_index, None] * expert_output
    _assert_close(result.output, dense.reshape(x.shape), "dense mixture")
    _assert_integer_equal(result.expert_counts, np.full(4, 6), "all-expert counts")


def _test_router_probabilities_and_combine_weights_sum_to_one() -> None:
    moe = SimpleMoE(d_model=8, num_experts=7, top_k=3, seed=41)
    x = np.random.default_rng(43).normal(size=(2, 6, 8))
    result = moe.forward(x)
    _assert_close(
        np.sum(result.router_probabilities, axis=-1),
        np.ones((2, 6)),
        "router probability sums",
    )
    _assert_close(
        np.sum(result.combine_weights, axis=-1),
        np.ones((2, 6)),
        "combine weight sums",
    )


def _test_expert_counts_cover_every_routing_slot() -> None:
    moe = SimpleMoE(d_model=6, num_experts=5, top_k=2, seed=47)
    x = np.random.default_rng(53).normal(size=(3, 7, 6))
    result = moe.forward(x)
    expected_total = x.shape[0] * x.shape[1] * moe.top_k
    if int(np.sum(result.expert_counts)) != expected_total:
        raise AssertionError(
            f"expert counts must sum to {expected_total}, got {result.expert_counts.sum()}"
        )
    reconstructed = np.bincount(
        result.selected_experts.reshape(-1),
        minlength=moe.num_experts,
    )
    _assert_integer_equal(result.expert_counts, reconstructed, "reconstructed counts")


def _test_balanced_routing_has_unit_auxiliary_loss() -> None:
    moe = SimpleMoE(d_model=4, num_experts=4, top_k=1, seed=59)
    moe.router_W[:] = 12.0 * np.eye(4)
    moe.router_b[:] = 0.0
    x = np.eye(4).reshape(1, 4, 4)
    result = moe.forward(x)
    _assert_integer_equal(result.expert_counts, np.ones(4, dtype=int), "balanced counts")
    _assert_close(result.load_balance_loss, 1.0, "balanced auxiliary loss")


def _test_collapsed_routing_has_large_auxiliary_loss() -> None:
    moe = SimpleMoE(d_model=4, num_experts=4, top_k=1, seed=61)
    moe.router_W[:] = 0.0
    moe.router_b[:] = np.array([30.0, -30.0, -30.0, -30.0])
    x = np.random.default_rng(67).normal(size=(2, 5, 4))
    result = moe.forward(x)
    _assert_integer_equal(result.expert_counts, np.array([10, 0, 0, 0]), "collapsed counts")
    if result.load_balance_loss < 3.999999:
        raise AssertionError(
            "collapsed top-1 routing should have auxiliary loss near num_experts"
        )


def _test_large_router_logits_remain_finite() -> None:
    moe = SimpleMoE(d_model=4, num_experts=3, top_k=2, seed=71)
    moe.router_W *= 10000.0
    x = np.array(
        [[[1.0, -2.0, 3.0, -4.0], [-3.0, 2.0, 0.5, 7.0], [8.0, -1.0, 2.0, 0.0]]]
    )
    result = moe.forward(x)
    for name, values in (
        ("output", result.output),
        ("router probabilities", result.router_probabilities),
        ("combine weights", result.combine_weights),
    ):
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"large router logits produced non-finite {name}")
    _assert_matches_reference(moe, x, "large router logits")


def _test_single_token_output_and_metadata_shapes() -> None:
    moe = SimpleMoE(d_model=8, num_experts=5, top_k=2, seed=73)
    x = np.random.default_rng(79).normal(size=(1, 1, 8))
    result = moe.forward(x)
    if result.output.shape != (1, 1, 8):
        raise AssertionError(f"output has wrong shape: {result.output.shape}")
    if result.router_probabilities.shape != (1, 1, 5):
        raise AssertionError("router probabilities have wrong shape")
    if result.selected_experts.shape != (1, 1, 2):
        raise AssertionError("selected experts have wrong shape")
    if result.combine_weights.shape != (1, 1, 2):
        raise AssertionError("combine weights have wrong shape")
    if result.expert_counts.shape != (5,):
        raise AssertionError("expert counts have wrong shape")
    _assert_matches_reference(moe, x, "single token")


def _test_batches_do_not_affect_each_other() -> None:
    moe = SimpleMoE(d_model=6, num_experts=4, top_k=2, seed=83)
    x = np.random.default_rng(89).normal(size=(3, 5, 6))
    batch_result = moe.forward(x)
    for batch_index in range(3):
        single_result = moe.forward(x[batch_index : batch_index + 1])
        _assert_close(
            single_result.output[0],
            batch_result.output[batch_index],
            "batch output independence",
        )
        _assert_close(
            single_result.router_probabilities[0],
            batch_result.router_probabilities[batch_index],
            "batch router independence",
        )
        _assert_integer_equal(
            single_result.selected_experts[0],
            batch_result.selected_experts[batch_index],
            "batch selection independence",
        )


def _test_input_is_not_modified() -> None:
    moe = SimpleMoE(d_model=8, num_experts=4, top_k=2, seed=97)
    x = np.random.default_rng(101).normal(size=(2, 5, 8))
    original = x.copy()
    moe.forward(x)
    _assert_close(x, original, "input immutability")


def _test_equal_seeds_produce_equal_results() -> None:
    first = SimpleMoE(d_model=8, num_experts=5, top_k=2, seed=103)
    second = SimpleMoE(d_model=8, num_experts=5, top_k=2, seed=103)
    x = np.random.default_rng(107).normal(size=(2, 4, 8))
    first_result = first.forward(x)
    second_result = second.forward(x)
    _assert_close(first_result.output, second_result.output, "seed output")
    _assert_close(
        first_result.router_probabilities,
        second_result.router_probabilities,
        "seed router probabilities",
    )
    _assert_integer_equal(
        first_result.selected_experts,
        second_result.selected_experts,
        "seed selected experts",
    )
    _assert_close(
        first_result.load_balance_loss,
        second_result.load_balance_loss,
        "seed auxiliary loss",
    )


def run_tests() -> bool:
    """Run all numerical, routing, combining, and load-balancing tests."""
    if np is None:
        print("[ENVIRONMENT ERROR] NumPy is missing. Run: python -m pip install numpy")
        return False

    tests = (
        ("top-1 exact routing and output", _test_top1_exact_routing_and_output),
        ("top-2 exact weighted combination", _test_top2_weighted_combination_exact_values),
        ("random top-1 numerical values", _test_random_top1_numerical_values),
        ("random top-2 numerical values", _test_random_top2_numerical_values),
        ("random top-3 numerical values", _test_random_top3_numerical_values),
        ("unselected experts do not contribute", _test_unselected_experts_do_not_contribute),
        ("all experts equal dense mixture", _test_all_experts_matches_dense_mixture),
        ("probability and combine sums", _test_router_probabilities_and_combine_weights_sum_to_one),
        ("expert counts cover routing slots", _test_expert_counts_cover_every_routing_slot),
        ("balanced routing auxiliary loss", _test_balanced_routing_has_unit_auxiliary_loss),
        ("collapsed routing auxiliary loss", _test_collapsed_routing_has_large_auxiliary_loss),
        ("large router logits remain finite", _test_large_router_logits_remain_finite),
        ("single-token shapes and values", _test_single_token_output_and_metadata_shapes),
        ("batch independence", _test_batches_do_not_affect_each_other),
        ("input immutability", _test_input_is_not_modified),
        ("seed reproducibility", _test_equal_seeds_produce_equal_results),
    )

    print("Simple MoE Automated Tests")
    print("=" * 54)
    passed = 0
    for name, test in tests:
        try:
            test()
        except NotImplementedError as error:
            print(f"[INCOMPLETE] {name}: {error}")
        except AssertionError as error:
            print(f"[FAILED] {name}: {error}")
        except Exception as error:
            print(f"[ERROR] {name}: {type(error).__name__}: {error}")
        else:
            passed += 1
            print(f"[PASSED] {name}")

    print("-" * 54)
    print(f"Result: {passed}/{len(tests)} tests passed")
    if passed != len(tests):
        print("Update forward() and run the file again.")
    return passed == len(tests)


_ENCODED_ANSWER = (
    "eNq9VU2P2yAQve+vQDnZu06yvUai6qXH9rCqtIcoQtgeJygYXMBKdn99MfgLNo7a"
    "qqpvjGfmvfl4UEKFKqkuVJWJBl5l6LpDotmIkipF31K0/oy+ya8voFtudg/IfqvV"
    "6kW2BpCRZxA6Q4WscyYA2XgoDJQIrg0oY/9QUSIFplUCcUlLpA01TBtW6I3N4rLl"
    "1BQnotk7ZDbBzxZEAYSDOJpThgjC6LrRJ9qAc/aIzqjAmZP1py6OV5uS1LIEnk6O"
    "pJCtMNbbh/k8++fDg3NRXQ2KcHlkRo9O6IvP1v99RU/BOXeh+sQqW+cUG+Zadw2s"
    "6TUJzLYbV6Zxx/cM0JSs1viHaiGds2mUzGnOODMMusQ2kW1mEgLeidi6EN3WifNZ8"
    "svGvwOn0RBw8+bUN2wYL+nH6+lRddRSmQlv/aeATJR4ZTcj57DqAfe7DO18441sy"
    "PkQErjRJUPPQCiX4kg6hN8tP65pgafveL/n5ALseHL1LzDafhjCbce/GoOtp2kNG"
    "aVgod5BSU04O0MyaLI0bw3gyqrOePJW5L0uiW04XBETSFFxBCf7jWjroQfpbmThZ"
    "WT9WWHZIs2lGU4e+XIC5TJEm4EDrHRMyKow56YTfuf+PIH6VgvDRAujsU/ni7fY"
    "SeDuq94HqQ+DlPvQ1/2c0iGIfwo88wXPqYxgBjHsU8wu2pv9na5m6LsUcECPYcE"
    "zCg+zbrj7rZ+ERXDHZHG3Z1dmOi1azYS/bXG8B/PVZ3VjRU7t3ezRaqAiuSktv8"
    "rPPsxd+jiiu0VuLZP5Hf04E/sUSnLKqXsPpO7K9HExTxvcy23G8tElSPt+9W/Q+"
    "JAl0SRxMNCxUXffpuDRye7eN/iWcUQJdmUGGdhj+Ojnrcm56S1fdXhxP/4RITfMm"
    "1QiQeDo/P+IBIuJg9Pk9GET8QfLIJVf56Ht6w=="
)


def _show_answer() -> None:
    """Print the reference implementation without exposing it by default."""
    compressed = base64.b64decode(_ENCODED_ANSWER.encode("ascii"))
    print(zlib.decompress(compressed).decode("utf-8"))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified MoE assignment")
    parser.add_argument(
        "--show-answer",
        action="store_true",
        help="Print the reference forward implementation and skip tests.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.show_answer:
        _show_answer()
        return 0
    return 0 if run_tests() else 1


if __name__ == "__main__":
    raise SystemExit(main())
