"""KV-cache consistency assignment using NumPy.

Goal
----
Implement only ``KVCachedCausalAttention.forward``. The same method must
support both full-sequence causal attention and cached autoregressive decoding.

The tests verify that:

    full_output[:, t] == cached_output_for_token_t

for every position, up to floating-point tolerance.

Shapes
------
    x:                  (B, Q, D)
    q, new_k, new_v:    (B, H, Q, Dh)
    cached_k, cached_v: (B, H, P, Dh)
    all_k, all_v:       (B, H, P + Q, Dh)
    scores:             (B, H, Q, P + Q)
    output:             (B, Q, D)

    B  = batch size
    Q  = number of new query tokens in this call
    P  = number of tokens already stored in the cache
    D  = model dimension
    H  = number of attention heads
    Dh = head dimension = D / H

Equations
---------
    Q_new = x @ Wq + bq
    K_new = x @ Wk + bk
    V_new = x @ Wv + bv

    K_all = concat(K_cache, K_new, axis=sequence)
    V_all = concat(V_cache, V_new, axis=sequence)

    scores = (Q_new @ K_all.transpose(-1, -2)) / sqrt(Dh)
    weights = softmax(mask(scores), axis=-1)
    context = weights @ V_all
    output = merge_heads(context) @ Wo + bo

For local query index ``i``, the global query position is ``P + i``. It may
attend only to key positions ``j <= P + i``. This detail matters when a call
contains multiple new tokens and a non-empty cache.

The returned cache must contain projected, split-head K and V values. Do not
cache input embeddings or attention outputs.

Usage
-----
    python 6.kv_cache_consistency.py
    python 6.kv_cache_consistency.py --show-answer

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


KVCache = tuple[np.ndarray, np.ndarray]


class KVCachedCausalAttention:
    """Multi-head causal self-attention with an explicit KV cache."""

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
        self.bq = rng.normal(0.0, scale, size=d_model)
        self.bk = rng.normal(0.0, scale, size=d_model)
        self.bv = rng.normal(0.0, scale, size=d_model)
        self.bo = rng.normal(0.0, scale, size=d_model)

    def forward(
        self,
        x: np.ndarray,
        cache: KVCache | None = None,
    ) -> tuple[np.ndarray, KVCache]:
        """Return outputs for new tokens and the updated ``(key, value)`` cache."""
        # ============== Student section: implement everything here ==============
        B, S, D = x.shape

        q = x @ self.Wq + self.bq # (B,S,D)
        k = x @ self.Wk + self.bk # (B,S,D)
        v = x @ self.Wv + self.bv # (B,S,D)
        
        q = q.reshape(B,S,self.num_heads,self.head_dim).transpose(0,2,1,3) # (B,S,D) -> (B,S,H,D') -> (B,H,S,D')
        k = k.reshape(B,S,self.num_heads,self.head_dim).transpose(0,2,1,3) # (B,S,D) -> (B,S,H,D') -> (B,H,S,D')
        v = v.reshape(B,S,self.num_heads,self.head_dim).transpose(0,2,1,3) # (B,S,D) -> (B,S,H,D') -> (B,H,S,D')

        if cache is None:
            past_len = 0
            all_k = k
            all_v = v
        else:
            cached_k, cached_v = cache
            all_k = np.concatenate((cached_k,k),axis=2) # (B, H, S + S', D')
            all_v = np.concatenate((cached_v,v),axis=2) # (B, H, S + S', D')
            past_len = cached_k.shape[2]

        # attention
        scores = q @ all_k.transpose(0,1,3,2) / np.sqrt(self.head_dim) # (B, H, S, D') @ (B, H, D', S + S') -> (B, H, S, S + S')
        mask = np.tril(np.ones((S,S),dtype=bool))
        past_mask = np.ones((S,past_len),dtype=bool)
        mask = np.concatenate((past_mask,mask),axis=1)

        scores = np.where(mask,scores,-np.inf)

        def softmax(x):
            x = np.exp(x - np.max(x,axis=-1,keepdims=True))
            return x / np.sum(x,axis=-1,keepdims=True)
        weight = softmax(scores)

        output = weight @ all_v # (B, H, S, S + S') @ (B, H, S + S', D') -> (B, H, S, D')
        output = output.transpose(0,2,1,3).reshape(B,S,D) # (B, H, S, D') -> (B, S, H, D') -> (B, H, D)
        return output @ self.Wo + self.bo, (all_k, all_v)
        
        # if past_len != 0:
        #     print(mask)
        #     print(scores[0][0])
        #     print(weight[0][0])

        # ============ End student section: implement everything here ============


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


def _split_heads(values: np.ndarray, attention: KVCachedCausalAttention) -> np.ndarray:
    batch_size, sequence_length, _ = values.shape
    return values.reshape(
        batch_size,
        sequence_length,
        attention.num_heads,
        attention.head_dim,
    ).transpose(0, 2, 1, 3)


def _reference_forward(
    attention: KVCachedCausalAttention,
    x: np.ndarray,
    cache: KVCache | None = None,
) -> tuple[np.ndarray, KVCache]:
    """Independent loop-based oracle used only by the tests."""
    batch_size, query_length, _ = x.shape
    q = _split_heads(x @ attention.Wq + attention.bq, attention)
    new_k = _split_heads(x @ attention.Wk + attention.bk, attention)
    new_v = _split_heads(x @ attention.Wv + attention.bv, attention)

    if cache is None:
        past_length = 0
        all_k = new_k.copy()
        all_v = new_v.copy()
    else:
        cached_k, cached_v = cache
        past_length = cached_k.shape[2]
        all_k = np.concatenate((cached_k, new_k), axis=2)
        all_v = np.concatenate((cached_v, new_v), axis=2)

    context = np.zeros_like(q)
    scale = np.sqrt(attention.head_dim)
    for batch_index in range(batch_size):
        for head_index in range(attention.num_heads):
            for query_index in range(query_length):
                allowed_length = past_length + query_index + 1
                scores = np.array(
                    [
                        np.dot(
                            q[batch_index, head_index, query_index],
                            all_k[batch_index, head_index, key_index],
                        )
                        / scale
                        for key_index in range(allowed_length)
                    ]
                )
                shifted = scores - np.max(scores)
                weights = np.exp(shifted)
                weights /= np.sum(weights)
                context[batch_index, head_index, query_index] = (
                    weights @ all_v[batch_index, head_index, :allowed_length]
                )

    merged = context.transpose(0, 2, 1, 3).reshape(
        batch_size,
        query_length,
        attention.d_model,
    )
    output = merged @ attention.Wo + attention.bo
    return output, (all_k, all_v)


def _assert_matches_reference(
    attention: KVCachedCausalAttention,
    x: np.ndarray,
    cache: KVCache | None,
    name: str,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-8,
) -> None:
    expected_output, expected_cache = _reference_forward(attention, x, cache)
    actual_output, actual_cache = attention.forward(x, cache)
    _assert_close(actual_output, expected_output, f"{name} output", atol=atol, rtol=rtol)
    _assert_close(actual_cache[0], expected_cache[0], f"{name} key cache")
    _assert_close(actual_cache[1], expected_cache[1], f"{name} value cache")


def _uniform_attention() -> KVCachedCausalAttention:
    attention = KVCachedCausalAttention(d_model=4, num_heads=2, seed=1)
    attention.Wq[:] = 0.0
    attention.Wk[:] = 0.0
    attention.Wv[:] = np.eye(4)
    attention.Wo[:] = np.eye(4)
    attention.bq[:] = 0.0
    attention.bk[:] = 0.0
    attention.bv[:] = 0.0
    attention.bo[:] = 0.0
    return attention


def _decode_one_token_at_a_time(
    attention: KVCachedCausalAttention,
    x: np.ndarray,
) -> tuple[np.ndarray, KVCache]:
    outputs = []
    cache = None
    for token_index in range(x.shape[1]):
        output, cache = attention.forward(x[:, token_index : token_index + 1], cache)
        outputs.append(output)
    return np.concatenate(outputs, axis=1), cache


def _decode_in_chunks(
    attention: KVCachedCausalAttention,
    x: np.ndarray,
    chunk_sizes: tuple[int, ...],
) -> tuple[np.ndarray, KVCache]:
    outputs = []
    cache = None
    start = 0
    for chunk_size in chunk_sizes:
        end = start + chunk_size
        output, cache = attention.forward(x[:, start:end], cache)
        outputs.append(output)
        start = end
    if start != x.shape[1]:
        raise AssertionError("chunk sizes must cover the complete sequence")
    return np.concatenate(outputs, axis=1), cache


def _test_uniform_prefix_exact_values() -> None:
    attention = _uniform_attention()
    x = np.array(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 0.0, -1.0, 2.0], [3.0, 4.0, 5.0, 6.0]]]
    )
    expected = np.stack(
        [x[:, : index + 1].mean(axis=1) for index in range(x.shape[1])],
        axis=1,
    )
    output, cache = attention.forward(x)
    _assert_close(output, expected, "uniform prefix output")
    expected_key_cache = np.zeros((1, 2, 3, 2))
    expected_value_cache = _split_heads(x, attention)
    _assert_close(cache[0], expected_key_cache, "uniform key cache")
    _assert_close(cache[1], expected_value_cache, "uniform value cache")


def _test_full_forward_matches_reference() -> None:
    attention = KVCachedCausalAttention(d_model=8, num_heads=2, seed=11)
    x = np.random.default_rng(13).normal(size=(2, 5, 8))
    _assert_matches_reference(attention, x, None, "full forward")


def _test_token_decode_matches_full_forward() -> None:
    attention = KVCachedCausalAttention(d_model=12, num_heads=3, seed=17)
    x = np.random.default_rng(19).normal(size=(2, 7, 12))
    full_output, full_cache = attention.forward(x)
    cached_output, cached_cache = _decode_one_token_at_a_time(attention, x)
    _assert_close(cached_output, full_output, "token decode output")
    _assert_close(cached_cache[0], full_cache[0], "token decode key cache")
    _assert_close(cached_cache[1], full_cache[1], "token decode value cache")


def _test_chunked_decode_matches_full_forward() -> None:
    attention = KVCachedCausalAttention(d_model=8, num_heads=4, seed=23)
    x = np.random.default_rng(29).normal(size=(1, 8, 8))
    full_output, full_cache = attention.forward(x)
    cached_output, cached_cache = _decode_in_chunks(attention, x, (3, 2, 3))
    _assert_close(cached_output, full_output, "chunked decode output")
    _assert_close(cached_cache[0], full_cache[0], "chunked key cache")
    _assert_close(cached_cache[1], full_cache[1], "chunked value cache")


def _test_nonempty_cache_multi_token_values() -> None:
    attention = KVCachedCausalAttention(d_model=6, num_heads=3, seed=31)
    x = np.random.default_rng(37).normal(size=(2, 6, 6))
    _, prefix_cache = _reference_forward(attention, x[:, :2])
    _assert_matches_reference(
        attention,
        x[:, 2:],
        prefix_cache,
        "multi-token call with nonempty cache",
    )


def _test_single_token_sequence() -> None:
    attention = KVCachedCausalAttention(d_model=8, num_heads=2, seed=41)
    x = np.random.default_rng(43).normal(size=(3, 1, 8))
    _assert_matches_reference(attention, x, None, "single token")


def _test_head_dimension_one() -> None:
    attention = KVCachedCausalAttention(d_model=4, num_heads=4, seed=47)
    x = np.random.default_rng(53).normal(size=(2, 5, 4))
    full_output, _ = attention.forward(x)
    cached_output, _ = _decode_one_token_at_a_time(attention, x)
    _assert_close(cached_output, full_output, "head dimension one")
    _assert_matches_reference(attention, x, None, "head dimension one reference")


def _test_cache_contains_projected_keys_and_values() -> None:
    attention = KVCachedCausalAttention(d_model=8, num_heads=2, seed=59)
    x = np.random.default_rng(61).normal(size=(2, 4, 8))
    _, cache = attention.forward(x)
    expected_k = _split_heads(x @ attention.Wk + attention.bk, attention)
    expected_v = _split_heads(x @ attention.Wv + attention.bv, attention)
    _assert_close(cache[0], expected_k, "projected key cache")
    _assert_close(cache[1], expected_v, "projected value cache")


def _test_cache_grows_without_changing_old_values() -> None:
    attention = KVCachedCausalAttention(d_model=8, num_heads=2, seed=67)
    x = np.random.default_rng(71).normal(size=(1, 5, 8))
    _, first_cache = attention.forward(x[:, :2])
    original_k = first_cache[0].copy()
    original_v = first_cache[1].copy()
    _, second_cache = attention.forward(x[:, 2:], first_cache)
    if second_cache[0].shape != (1, 2, 5, 4):
        raise AssertionError(f"updated key cache has wrong shape: {second_cache[0].shape}")
    if second_cache[1].shape != (1, 2, 5, 4):
        raise AssertionError(f"updated value cache has wrong shape: {second_cache[1].shape}")
    _assert_close(second_cache[0][:, :, :2], original_k, "old key cache values")
    _assert_close(second_cache[1][:, :, :2], original_v, "old value cache values")
    _assert_close(first_cache[0], original_k, "input key cache immutability")
    _assert_close(first_cache[1], original_v, "input value cache immutability")


def _test_output_projection_exact_value() -> None:
    attention = _uniform_attention()
    attention.Wo[:] = np.array(
        [[1.0, 0.2, -0.3, 0.5], [0.4, -1.0, 0.7, 0.0],
         [-0.2, 0.8, 1.3, -0.5], [0.6, 0.1, -0.4, 1.1]]
    )
    attention.bo[:] = np.array([0.1, -0.2, 0.3, -0.4])
    x = np.array([[[1.0, -2.0, 0.5, 3.0], [-1.0, 4.0, 2.0, 0.0]]])
    prefix_context = np.stack([x[:, 0], x.mean(axis=1)], axis=1)
    expected = prefix_context @ attention.Wo + attention.bo
    output, _ = attention.forward(x)
    _assert_close(output, expected, "output projection")


def _test_future_tokens_do_not_change_prefix() -> None:
    attention = KVCachedCausalAttention(d_model=8, num_heads=2, seed=73)
    rng = np.random.default_rng(79)
    original = rng.normal(size=(1, 7, 8))
    changed = original.copy()
    changed[:, 4:] = rng.normal(size=(1, 3, 8)) * 100.0
    original_output, _ = attention.forward(original)
    changed_output, _ = attention.forward(changed)
    _assert_close(changed_output[:, :4], original_output[:, :4], "causal prefix")


def _test_batches_do_not_affect_each_other() -> None:
    attention = KVCachedCausalAttention(d_model=8, num_heads=4, seed=83)
    x = np.random.default_rng(89).normal(size=(3, 6, 8))
    batch_output, _ = _decode_one_token_at_a_time(attention, x)
    for batch_index in range(3):
        single_output, _ = _decode_one_token_at_a_time(
            attention,
            x[batch_index : batch_index + 1],
        )
        _assert_close(single_output[0], batch_output[batch_index], "batch independence")


def _test_input_is_not_modified() -> None:
    attention = KVCachedCausalAttention(d_model=8, num_heads=2, seed=97)
    x = np.random.default_rng(101).normal(size=(2, 5, 8))
    original = x.copy()
    attention.forward(x)
    _assert_close(x, original, "input immutability")


def _test_large_scores_remain_finite_and_consistent() -> None:
    attention = KVCachedCausalAttention(d_model=4, num_heads=2, seed=103)
    attention.Wq *= 10000.0
    attention.Wk *= 10000.0
    x = np.array(
        [[[1.2, -0.7, 2.5, 0.3], [-1.0, 3.0, 0.4, -2.0], [2.2, 0.1, -1.5, 4.0]]]
    )
    full_output, full_cache = attention.forward(x)
    cached_output, cached_cache = _decode_one_token_at_a_time(attention, x)
    if not np.all(np.isfinite(full_output)) or not np.all(np.isfinite(cached_output)):
        raise AssertionError("large finite scores must produce finite outputs")
    _assert_close(cached_output, full_output, "large-score cached output", atol=1e-7, rtol=1e-7)
    _assert_close(cached_cache[0], full_cache[0], "large-score key cache")
    _assert_close(cached_cache[1], full_cache[1], "large-score value cache")


def run_tests() -> bool:
    """Run all numerical and cache-consistency tests."""
    if np is None:
        print("[ENVIRONMENT ERROR] NumPy is missing. Run: python -m pip install numpy")
        return False

    tests = (
        ("uniform prefix exact values", _test_uniform_prefix_exact_values),
        ("full forward numerical values", _test_full_forward_matches_reference),
        ("token decode equals full forward", _test_token_decode_matches_full_forward),
        ("chunked decode equals full forward", _test_chunked_decode_matches_full_forward),
        ("nonempty cache multi-token values", _test_nonempty_cache_multi_token_values),
        ("single-token sequence", _test_single_token_sequence),
        ("head dimension one", _test_head_dimension_one),
        ("cache projected values", _test_cache_contains_projected_keys_and_values),
        ("cache growth and immutability", _test_cache_grows_without_changing_old_values),
        ("output projection exact values", _test_output_projection_exact_value),
        ("causal prefix invariance", _test_future_tokens_do_not_change_prefix),
        ("batch independence", _test_batches_do_not_affect_each_other),
        ("input immutability", _test_input_is_not_modified),
        ("large-score numerical stability", _test_large_scores_remain_finite_and_consistent),
    )

    print("KV-Cache Consistency Automated Tests")
    print("=" * 52)
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

    print("-" * 52)
    print(f"Result: {passed}/{len(tests)} tests passed")
    if passed != len(tests):
        print("Update forward() and run the file again.")
    return passed == len(tests)


_ENCODED_ANSWER = (
    "eNqtVcGOm0AMvecrrJxAO2G36S1SVpV6rNRDVbWHaEVmwQQEDDAzELbqx9czAwSy"
    "SU+LFGGc5+dnjzExJpBU8sxl7K2ALoVFwqzV70DUgYi5lPzNuSIepbiDb7++GgP+w"
    "vdKIOztja182DyDbusCD7PAEf2ysxTr9foH6lYKqFpdt1qZ9CDwDLrKUSjgIgZN5G"
    "0dc40xHI9ejkTT8aJF/3h0IgLisXyvXEdpqLI/yKBpUb6FBYqTThmEJKwPVMprXFl"
    "oYxzwxZYY/G7gwVmvjf2XJIT5ApFPiHxCdAtENyG6S4omkGizeve12SDRlmGKPFbD"
    "s7HDOCv9hR57/zjKbqDsPobSFawlF6quFHpPDLYMPjH4fKuK/+Mu0m7jLDBL3ABA"
    "puzYuakyV82VHqQT0dPk50VxkbDwTgmtFws1Y7NJ4jBno2XA1ryTcAxwE3fYvrwX"
    "UAdRJSKaakE/z7uksNJ8BrzP1H7rvxd5O7Jzkd0s0r3DUUWHa06GJtVmX3bUtJP6"
    "6i/Bg/FosqlGau/quC2aXsWQaDKdVUI5ZZyoT+i5PGPx/sHuBNi5Nri5mgfOm/cw"
    "o5lPoH/YMXvIjiRpaW9gWHJlurlU8nydYlkb8Z9TlOjNOAaFg04jlcGGgJlI2BA5"
    "9jPNElpF4XWrNoa35L3nnodT2FB7c8Samqb2PyUtLUtyxuyU6kEM9rW3ZF2CHi1K"
    "taU3eO5zu8VcCY29JvKRwR28m+wS5Yk26X6E3X6/poUwjd9sM0y+xYaYvHZU4rC"
    "sYiyc19XjdjxlHiSMi7OaFmdlcXL+SWDghom5EvzVPxdAE6M="
)


def _show_answer() -> None:
    """Print the reference implementation without exposing it by default."""
    compressed = base64.b64decode(_ENCODED_ANSWER.encode("ascii"))
    print(zlib.decompress(compressed).decode("utf-8"))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KV-cache consistency assignment")
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
