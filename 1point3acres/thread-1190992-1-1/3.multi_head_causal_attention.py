"""Standalone multi-head causal self-attention assignment using NumPy.

Goal
----
Implement only ``MultiHeadCausalAttention.forward``. The method must perform
the complete attention calculation without calling assignment helper methods:

1. Project the input into Q, K, and V.
2. Split the model dimension into multiple heads.
3. Compute scaled dot-product attention scores.
4. Apply a left-to-right causal mask.
5. Apply a numerically stable softmax.
6. Compute the weighted sum of V.
7. Merge all heads.
8. Apply the output projection.
9. Store the probabilities in ``self.last_attention_weights``.

Shapes
------
    x:       (B, T, D)
    q, k, v: (B, H, T, Dh)
    scores:  (B, H, T, T)
    weights: (B, H, T, T)
    context: (B, H, T, Dh)
    output:  (B, T, D)

    B  = batch size
    T  = sequence length
    D  = model dimension
    H  = number of heads
    Dh = head dimension = D / H

Equations
---------
    Q = x @ Wq + bq
    K = x @ Wk + bk
    V = x @ Wv + bv

    scores = (Q @ K.transpose(-1, -2)) / sqrt(Dh)
    scores[..., i, j] = -infinity when j > i
    weights = softmax(scores, axis=-1)
    context = weights @ V
    output = merge_heads(context) @ Wo + bo

The causal mask keeps the diagonal and lower triangle. Position ``i`` may
attend to positions ``0`` through ``i``, but never to a future position.

Usage
-----
    python 3.multi_head_causal_attention.py
    python 3.multi_head_causal_attention.py --show-answer

Edit only the single student section inside ``forward``.
"""

from __future__ import annotations

import argparse
import base64
import sys
import zlib

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


class MultiHeadCausalAttention:
    """Standalone multi-head causal self-attention with output projection."""

    def __init__(self, d_model: int, num_heads: int, seed: int = 0) -> None:
        if np is None:
            raise ModuleNotFoundError(
                "This assignment requires NumPy. Run: python -m pip install numpy"
            )
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.Wq = rng.normal(0.0, scale, size=(d_model, d_model))
        self.Wk = rng.normal(0.0, scale, size=(d_model, d_model))
        self.Wv = rng.normal(0.0, scale, size=(d_model, d_model))
        self.Wo = rng.normal(0.0, scale, size=(d_model, d_model))
        self.bq = np.zeros(d_model)
        self.bk = np.zeros(d_model)
        self.bv = np.zeros(d_model)
        self.bo = np.zeros(d_model)
        self.last_attention_weights: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return multi-head causal attention output with shape (B, T, D)."""
        B, T, D = x.shape
        # ================= Student section: complete forward =================
        q = x @ self.Wq + self.bq # (B, T, D) @ (D,D) = (B, T, D)
        k = x @ self.Wk + self.bk # (B, T, D) @ (D,D) = (B, T, D)
        v = x @ self.Wv + self.bv # (B, T, D) @ (D,D) = (B, T, D)

        q = q.reshape(B,T,self.num_heads,self.head_dim) # (B, T, D) -> (B, T, H, D')
        k = k.reshape(B,T,self.num_heads,self.head_dim) # (B, T, D) -> (B, T, H, D')
        v = v.reshape(B,T,self.num_heads,self.head_dim) # (B, T, D) -> (B, T, H, D')

        q = q.transpose(0,2,1,3) #(B, T, H, D') -> (B, H, T, D')
        k = k.transpose(0,2,1,3) #(B, T, H, D') -> (B, H, T, D')
        v = v.transpose(0,2,1,3) #(B, T, H, D') -> (B, H, T, D')

        scores = q @ k.transpose(0,1,3,2) / np.sqrt(self.head_dim) # (B, H, T, D') @ (B, H, D', T) -> (B, H, T, T)

        # softmax
        def softmax(x):
            x = np.exp(x - np.max(x,axis=-1,keepdims=True))
            return x / np.sum(x,axis=-1,keepdims=True)

        mask = np.triu(np.ones((T,T),dtype=bool),k=1)
        scores = np.where(mask,-np.inf, scores)
        weight = softmax(scores) #  (B, H, T, T)
        output = weight @ v # (B,H,T,T) @ (B,H,T,D') = (B,H,T,D')
        output = output.transpose(0,2,1,3).reshape(B,T,-1) # (B,H,T,D') -> (B,T,H,D') -> (B,T,D)
        self.last_attention_weights = weight
        return output @ self.Wo + self.bo

        # =============== End student section: complete forward ===============


def _assert_close(
    actual: np.ndarray,
    expected: np.ndarray,
    name: str,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-8,
) -> None:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name} has the wrong shape: expected {expected.shape}, got {actual.shape}"
        )
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        difference = np.abs(actual - expected)
        raise AssertionError(
            f"{name} has incorrect values: max absolute error={difference.max():.3e}"
        )


def _reference_forward(
    attention: MultiHeadCausalAttention,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent loop-based oracle used only by the tests."""
    batch_size, sequence_length, _ = x.shape
    q = x @ attention.Wq + attention.bq
    k = x @ attention.Wk + attention.bk
    v = x @ attention.Wv + attention.bv

    q = q.reshape(batch_size, sequence_length, attention.num_heads, attention.head_dim)
    k = k.reshape(batch_size, sequence_length, attention.num_heads, attention.head_dim)
    v = v.reshape(batch_size, sequence_length, attention.num_heads, attention.head_dim)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    weights = np.zeros(
        (batch_size, attention.num_heads, sequence_length, sequence_length),
        dtype=float,
    )
    context = np.zeros_like(v)
    scale = np.sqrt(attention.head_dim)

    for batch_index in range(batch_size):
        for head_index in range(attention.num_heads):
            for query_index in range(sequence_length):
                allowed_scores = np.array(
                    [
                        np.dot(
                            q[batch_index, head_index, query_index],
                            k[batch_index, head_index, key_index],
                        )
                        / scale
                        for key_index in range(query_index + 1)
                    ]
                )
                shifted = allowed_scores - np.max(allowed_scores)
                probabilities = np.exp(shifted)
                probabilities /= np.sum(probabilities)
                weights[
                    batch_index,
                    head_index,
                    query_index,
                    : query_index + 1,
                ] = probabilities
                context[batch_index, head_index, query_index] = (
                    probabilities
                    @ v[batch_index, head_index, : query_index + 1]
                )

    merged = context.transpose(0, 2, 1, 3).reshape(
        batch_size, sequence_length, attention.d_model
    )
    output = merged @ attention.Wo + attention.bo
    return output, weights


def _assert_matches_reference(
    attention: MultiHeadCausalAttention,
    x: np.ndarray,
    name: str,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-8,
) -> None:
    expected_output, expected_weights = _reference_forward(attention, x)
    actual_output = attention.forward(x)
    _assert_close(actual_output, expected_output, f"{name} output", atol=atol, rtol=rtol)
    _assert_close(
        attention.last_attention_weights,
        expected_weights,
        f"{name} weights",
        atol=atol,
        rtol=rtol,
    )


def _uniform_prefix_attention() -> MultiHeadCausalAttention:
    attention = MultiHeadCausalAttention(d_model=4, num_heads=2, seed=0)
    attention.Wq[:] = 0.0
    attention.Wk[:] = 0.0
    attention.Wv[:] = np.eye(4)
    attention.Wo[:] = np.eye(4)
    attention.bq[:] = 0.0
    attention.bk[:] = 0.0
    attention.bv[:] = 0.0
    attention.bo[:] = 0.0
    return attention


def _test_uniform_prefix_average_exact_values() -> None:
    attention = _uniform_prefix_attention()
    x = np.array(
        [[[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, -1.0, 6.0], [-3.0, 4.0, 5.0, -2.0]]]
    )
    expected = np.stack(
        [x[:, : position + 1].mean(axis=1) for position in range(x.shape[1])],
        axis=1,
    )
    output = attention.forward(x)
    _assert_close(output, expected, "uniform prefix output")
    expected_weights = np.array(
        [[
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]],
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]],
        ]]
    )
    _assert_close(attention.last_attention_weights, expected_weights, "uniform weights")


def _test_nonuniform_attention_exact_values() -> None:
    attention = MultiHeadCausalAttention(d_model=2, num_heads=1, seed=0)
    attention.Wq[:] = np.eye(2)
    attention.Wk[:] = np.eye(2)
    attention.Wv[:] = np.eye(2)
    attention.Wo[:] = np.eye(2)
    attention.bq[:] = 0.0
    attention.bk[:] = 0.0
    attention.bv[:] = 0.0
    attention.bo[:] = 0.0
    x = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    expected_weights = np.array(
        [[[[1.0, 0.0, 0.0],
           [0.3302384506733431, 0.6697615493266569, 0.0],
           [0.24825507866341106, 0.24825507866341106, 0.5034898426731779]]]]
    )
    expected_output = (expected_weights @ x[:, None])[:, 0]
    _assert_close(attention.forward(x), expected_output, "nonuniform output")
    _assert_close(attention.last_attention_weights, expected_weights, "nonuniform weights")


def _test_random_numerical_single_head() -> None:
    attention = MultiHeadCausalAttention(d_model=6, num_heads=1, seed=3)
    x = np.random.default_rng(5).normal(size=(1, 4, 6))
    _assert_matches_reference(attention, x, "single-head random")


def _test_random_numerical_two_heads() -> None:
    attention = MultiHeadCausalAttention(d_model=8, num_heads=2, seed=7)
    x = np.random.default_rng(11).normal(size=(2, 5, 8))
    _assert_matches_reference(attention, x, "two-head random")


def _test_random_numerical_four_heads() -> None:
    attention = MultiHeadCausalAttention(d_model=12, num_heads=4, seed=13)
    x = np.random.default_rng(17).normal(size=(2, 3, 12))
    _assert_matches_reference(attention, x, "four-head random")


def _test_head_dimension_one() -> None:
    attention = MultiHeadCausalAttention(d_model=5, num_heads=5, seed=19)
    x = np.random.default_rng(23).normal(size=(2, 4, 5))
    _assert_matches_reference(attention, x, "head-dimension-one")


def _test_three_batches_are_numerically_correct() -> None:
    attention = MultiHeadCausalAttention(d_model=8, num_heads=4, seed=29)
    x = np.random.default_rng(31).normal(size=(3, 4, 8))
    _assert_matches_reference(attention, x, "three-batch random")


def _test_longer_sequence_is_numerically_correct() -> None:
    attention = MultiHeadCausalAttention(d_model=6, num_heads=3, seed=37)
    x = np.random.default_rng(41).normal(size=(1, 9, 6))
    _assert_matches_reference(attention, x, "longer sequence")


def _test_single_token_projection_exact_value() -> None:
    attention = MultiHeadCausalAttention(d_model=6, num_heads=3, seed=43)
    x = np.array([[[0.5, -1.0, 2.0, 0.3, -0.7, 1.2]]])
    expected = (x @ attention.Wv + attention.bv) @ attention.Wo + attention.bo
    _assert_close(attention.forward(x), expected, "single-token projection")
    _assert_close(
        attention.last_attention_weights,
        np.ones((1, 3, 1, 1)),
        "single-token weights",
    )


def _test_output_projection_and_bias_exact_value() -> None:
    attention = _uniform_prefix_attention()
    attention.Wo[:] = np.array(
        [[1.0, 0.2, -0.3, 0.5], [0.4, -1.0, 0.7, 0.0],
         [-0.2, 0.8, 1.3, -0.5], [0.6, 0.1, -0.4, 1.1]]
    )
    attention.bo[:] = np.array([0.1, -0.2, 0.3, -0.4])
    x = np.array([[[1.0, -2.0, 0.5, 3.0], [-1.0, 4.0, 2.0, 0.0]]])
    prefix_context = np.array([x[:, 0], x.mean(axis=1)]).transpose(1, 0, 2)
    expected = prefix_context @ attention.Wo + attention.bo
    _assert_close(attention.forward(x), expected, "output projection and bias")


def _test_future_tokens_do_not_change_prefix() -> None:
    attention = MultiHeadCausalAttention(d_model=8, num_heads=2, seed=47)
    rng = np.random.default_rng(53)
    original = rng.normal(size=(1, 6, 8))
    changed = original.copy()
    changed[:, 4:] = rng.normal(size=(1, 2, 8)) * 100.0
    original_output = attention.forward(original)
    changed_output = attention.forward(changed)
    _assert_close(
        changed_output[:, :4],
        original_output[:, :4],
        "causal prefix invariance",
        atol=1e-7,
        rtol=1e-7,
    )


def _test_weight_shape_sums_and_future_zeros() -> None:
    attention = MultiHeadCausalAttention(d_model=12, num_heads=3, seed=59)
    x = np.random.default_rng(61).normal(size=(2, 5, 12))
    output = attention.forward(x)
    if output.shape != (2, 5, 12):
        raise AssertionError(f"output must have shape (2, 5, 12), got {output.shape}")
    weights = attention.last_attention_weights
    if weights is None or weights.shape != (2, 3, 5, 5):
        raise AssertionError("last_attention_weights must have shape (B, H, T, T)")
    _assert_close(np.sum(weights, axis=-1), np.ones((2, 3, 5)), "weight row sums")
    future = np.triu(np.ones((5, 5), dtype=bool), k=1)
    if not np.all(weights[..., future] == 0.0):
        raise AssertionError("future attention weights must be exactly zero")


def _test_batches_do_not_affect_each_other() -> None:
    attention = MultiHeadCausalAttention(d_model=8, num_heads=4, seed=67)
    x = np.random.default_rng(71).normal(size=(3, 5, 8))
    batch_output = attention.forward(x)
    for batch_index in range(3):
        single_output = attention.forward(x[batch_index : batch_index + 1])
        _assert_close(single_output[0], batch_output[batch_index], "batch independence")


def _test_input_is_not_modified() -> None:
    attention = MultiHeadCausalAttention(d_model=8, num_heads=2, seed=73)
    x = np.random.default_rng(79).normal(size=(2, 4, 8))
    original = x.copy()
    attention.forward(x)
    _assert_close(x, original, "input immutability")


def _test_large_attention_scores_remain_finite() -> None:
    attention = MultiHeadCausalAttention(d_model=4, num_heads=2, seed=83)
    attention.Wq *= 10000.0
    attention.Wk *= 10000.0
    x = np.array([[[1.2, -0.7, 2.5, 0.3], [-1.0, 3.0, 0.4, -2.0], [2.2, 0.1, -1.5, 4.0]]])
    output = attention.forward(x)
    if not np.all(np.isfinite(output)):
        raise AssertionError("large finite scores must produce finite output")
    if not np.all(np.isfinite(attention.last_attention_weights)):
        raise AssertionError("large finite scores must produce finite weights")
    _assert_matches_reference(attention, x, "large-score stability", atol=1e-7, rtol=1e-7)


def _test_last_attention_weights_update_each_call() -> None:
    attention = MultiHeadCausalAttention(d_model=8, num_heads=2, seed=89)
    rng = np.random.default_rng(97)
    attention.forward(rng.normal(size=(1, 3, 8)))
    if attention.last_attention_weights.shape != (1, 2, 3, 3):
        raise AssertionError("first call stored the wrong attention shape")
    attention.forward(rng.normal(size=(2, 5, 8)))
    if attention.last_attention_weights.shape != (2, 2, 5, 5):
        raise AssertionError("second call did not replace the stored attention weights")


def _test_equal_seeds_produce_equal_outputs() -> None:
    first = MultiHeadCausalAttention(d_model=8, num_heads=2, seed=101)
    second = MultiHeadCausalAttention(d_model=8, num_heads=2, seed=101)
    x = np.random.default_rng(103).normal(size=(2, 4, 8))
    _assert_close(first.forward(x), second.forward(x), "seed reproducibility output")
    _assert_close(
        first.last_attention_weights,
        second.last_attention_weights,
        "seed reproducibility weights",
    )


def run_tests() -> bool:
    """Run every numerical and behavioral test."""
    if np is None:
        print("[ENVIRONMENT ERROR] NumPy is missing. Run: python -m pip install numpy")
        return False

    tests = (
        ("uniform prefix exact values", _test_uniform_prefix_average_exact_values),
        ("nonuniform exact values", _test_nonuniform_attention_exact_values),
        ("random numerical single head", _test_random_numerical_single_head),
        ("random numerical two heads", _test_random_numerical_two_heads),
        ("random numerical four heads", _test_random_numerical_four_heads),
        ("head dimension one", _test_head_dimension_one),
        ("three batches", _test_three_batches_are_numerically_correct),
        ("longer sequence", _test_longer_sequence_is_numerically_correct),
        ("single-token projection", _test_single_token_projection_exact_value),
        ("output projection and bias", _test_output_projection_and_bias_exact_value),
        ("causal prefix invariance", _test_future_tokens_do_not_change_prefix),
        ("weight shape, sums, and mask", _test_weight_shape_sums_and_future_zeros),
        ("batch independence", _test_batches_do_not_affect_each_other),
        ("input immutability", _test_input_is_not_modified),
        ("large-score numerical stability", _test_large_attention_scores_remain_finite),
        ("stored weights update", _test_last_attention_weights_update_each_call),
        ("seed reproducibility", _test_equal_seeds_produce_equal_outputs),
    )
    print("Multi-Head Causal Attention Automated Tests")
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
    if passed == len(tests):
        print("Congratulations! Multi-head causal attention passed all checks.")
        return True
    print("Update forward() and run again. Use --show-answer for help.")
    return False


_ENCODED_ANSWER = (
    "c-pO0O>f&U487-9a6QSGoh|LHfQO>k{z0)FhCyIrx;AV}wj?|1e*N^}xN(;SJ5(p~k$il"
    "VDHl{iDb*2G!8EmH3icjEzZC^4g)a$YU(Ve74Y*?=_ZaGcF`GhRUV6&QR-#ElN?n3=MRf-"
    "sa<~JjoU1GB>1Ve4<uTuDTkuX6)J9f+6xN#HJGLWzQ%Wj!W1$<hfeI&rKmhD9g{Kt0CO6"
    "3@6z<U+Dz59_lsejBByk|#l6W>%3fyU(TZiPq`|uF^{}|w7wDv=YHmhi}M4J$uR?%sRPM"
    "d{#*eVLm*i}`bPK+I|2n#*?-sbMgl0{1MZfsuq^lof%Qln7lUTR{WQ@BlGm%`n-_J7Btn"
    "#=K!pbye%^<CU`H(Pzg9xZ8>xt#1aflq$_^q>s8pzeU9#+G-eo4BY()gxPW%mGq3E_Pw^"
    "6_A|vdM{<`oF?A|l4LE>@f|Bt#LoIUl@6|jdrmR$G*?xZh6=tE&qn51I<zeMDWI)&?x$^t"
    "dN-NQpVUaPGuhq^AWV(z+}|N=O&v3jR6ke7C#`DY?TsBfN3Zt`*W98u9F6f6Qt(&y*@`n"
    "Bq%dU7E9^eu^d>S}S61Zj-2Y$R`L_~@6q!+ph41uw<??#v<`2adn9B"
)


def _show_answer() -> None:
    answer = zlib.decompress(base64.b85decode(_ENCODED_ANSWER)).decode("utf-8")
    print("Reference solution for MultiHeadCausalAttention.forward:")
    print("=" * 68)
    print(answer, end="")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone multi-head causal self-attention assignment"
    )
    parser.add_argument(
        "--show-answer",
        action="store_true",
        help="show the forward reference solution without running tests",
    )
    arguments = parser.parse_args()
    if arguments.show_answer:
        _show_answer()
        return 0
    return 0 if run_tests() else 1


if __name__ == "__main__":
    sys.exit(main())
