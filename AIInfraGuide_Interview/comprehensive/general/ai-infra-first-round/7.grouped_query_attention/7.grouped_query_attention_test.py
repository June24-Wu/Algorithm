"""Automated tests for the Grouped Query Attention assignment.

Run this file after implementing ``GroupQueryAttention.forward`` in
``7.grouped_query_attention.py``.

Shapes
------
    x:          (B, S, D)
    q:          (B, Hkv, G, S, Dh)
    new_k/new_v:(B, Hkv, S, Dh)
    all_k/all_v:(B, Hkv, P + S, Dh)
    scores:     (B, Hkv, G, S, P + S)
    output:     (B, S, D)

    Hq  = number of query heads
    Hkv = number of key/value heads
    G   = query heads per KV head = Hq / Hkv
    Dh  = head dimension = D / Hq
    P   = cached prefix length

Equations
---------
    Q = x @ Wq
    K_new = x @ Wk
    V_new = x @ Wv

    scores = (Q @ K_all.transpose(-1, -2)) / sqrt(Dh)
    scores[..., i, j] = -infinity when j > P + i
    probabilities = softmax(scores, axis=-1)
    context = probabilities @ V_all
    output = merge_query_heads(context) @ Wo

Query heads in the same group share K and V, but each query head computes its
own scores and softmax distribution. The cache must stay compact with only
``Hkv`` heads; do not repeat cached K/V values to ``Hq`` heads.

Usage
-----
    python 7.grouped_query_attention_test.py
    python 7.grouped_query_attention_test.py --show-answer
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import sys
import zlib
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


_ASSIGNMENT_PATH = Path(__file__).with_name("7.grouped_query_attention.py")
_ASSIGNMENT_SPEC = importlib.util.spec_from_file_location(
    "grouped_query_attention_assignment", _ASSIGNMENT_PATH
)
if _ASSIGNMENT_SPEC is None or _ASSIGNMENT_SPEC.loader is None:
    raise ImportError(f"Cannot load assignment file: {_ASSIGNMENT_PATH}")
_ASSIGNMENT_MODULE = importlib.util.module_from_spec(_ASSIGNMENT_SPEC)
_ASSIGNMENT_SPEC.loader.exec_module(_ASSIGNMENT_MODULE)
GroupQueryAttention = _ASSIGNMENT_MODULE.GroupQueryAttention


def _assert_close(
    actual: np.ndarray,
    expected: np.ndarray,
    name: str,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-9,
) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
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
    model: GroupQueryAttention,
    x: np.ndarray,
    cache: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Independent oracle that explicitly repeats compact KV heads."""
    x = np.asarray(x, dtype=np.float64)
    batch, length, _ = x.shape
    q = (x @ model.Wq).reshape(
        batch, length, model.num_query_heads, model.head_dim
    ).transpose(0, 2, 1, 3)
    new_k = (x @ model.Wk).reshape(
        batch, length, model.num_kv_heads, model.head_dim
    ).transpose(0, 2, 1, 3)
    new_v = (x @ model.Wv).reshape(
        batch, length, model.num_kv_heads, model.head_dim
    ).transpose(0, 2, 1, 3)

    if cache is None:
        past_length = 0
        all_k, all_v = new_k, new_v
    else:
        past_k, past_v = cache
        past_length = past_k.shape[2]
        all_k = np.concatenate((past_k, new_k), axis=2)
        all_v = np.concatenate((past_v, new_v), axis=2)

    repeated_k = np.repeat(all_k, model.group_size, axis=1)
    repeated_v = np.repeat(all_v, model.group_size, axis=1)
    context = np.empty_like(q)
    for batch_index in range(batch):
        for head_index in range(model.num_query_heads):
            for local_position in range(length):
                last_key = past_length + local_position + 1
                logits = (
                    repeated_k[batch_index, head_index, :last_key]
                    @ q[batch_index, head_index, local_position]
                ) / np.sqrt(model.head_dim)
                probabilities = np.exp(logits - np.max(logits))
                probabilities /= np.sum(probabilities)
                context[batch_index, head_index, local_position] = (
                    probabilities
                    @ repeated_v[batch_index, head_index, :last_key]
                )

    merged = context.transpose(0, 2, 1, 3).reshape(
        batch, length, model.d_model
    )
    return merged @ model.Wo, (all_k, all_v)


def _project_compact_cache(
    model: GroupQueryAttention,
    prefix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a compact prefix cache without calling the student's forward method."""
    batch, length, _ = prefix.shape
    key = (prefix @ model.Wk).reshape(
        batch, length, model.num_kv_heads, model.head_dim
    ).transpose(0, 2, 1, 3)
    value = (prefix @ model.Wv).reshape(
        batch, length, model.num_kv_heads, model.head_dim
    ).transpose(0, 2, 1, 3)
    return key, value


def _test_full_forward_matches_reference() -> None:
    configurations = (
        (1, 1, 16, 4, 4),
        (2, 6, 24, 6, 2),
        (3, 5, 32, 8, 1),
        (2, 9, 12, 3, 1),
    )
    for case, (batch, length, dimension, query_heads, kv_heads) in enumerate(
        configurations
    ):
        model = GroupQueryAttention(dimension, query_heads, kv_heads, seed=10 + case)
        x = np.random.default_rng(100 + case).normal(
            size=(batch, length, dimension)
        )
        expected, expected_cache = _reference_forward(model, x)
        actual, actual_cache = model.forward(x)
        _assert_close(actual, expected, f"configuration {case} output")
        _assert_close(actual_cache[0], expected_cache[0], f"configuration {case} K")
        _assert_close(actual_cache[1], expected_cache[1], f"configuration {case} V")


def _test_cache_is_compact() -> None:
    model = GroupQueryAttention(32, 8, 2, seed=21)
    x = np.random.default_rng(22).normal(size=(3, 7, 32))
    _, (key_cache, value_cache) = model.forward(x)
    expected_shape = (3, 2, 7, 4)
    if key_cache.shape != expected_shape or value_cache.shape != expected_shape:
        raise AssertionError(
            f"cache must use Hkv heads with shape {expected_shape}, "
            f"got {key_cache.shape} and {value_cache.shape}"
        )


def _test_cache_contains_projected_values() -> None:
    model = GroupQueryAttention(16, 4, 2, seed=23)
    x = np.random.default_rng(24).normal(size=(2, 5, 16))
    _, (actual_k, actual_v) = model.forward(x)
    expected_k = (x @ model.Wk).reshape(2, 5, 2, 4).transpose(0, 2, 1, 3)
    expected_v = (x @ model.Wv).reshape(2, 5, 2, 4).transpose(0, 2, 1, 3)
    _assert_close(actual_k, expected_k, "projected key cache")
    _assert_close(actual_v, expected_v, "projected value cache")


def _decode_one_token_at_a_time(
    model: GroupQueryAttention,
    x: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    cache = None
    pieces = []
    for position in range(x.shape[1]):
        output, cache = model.forward(x[:, position : position + 1], cache)
        pieces.append(output)
    return np.concatenate(pieces, axis=1), cache


def _test_token_decode_matches_full_forward() -> None:
    for kv_heads in (8, 2, 1):
        model = GroupQueryAttention(32, 8, kv_heads, seed=30 + kv_heads)
        x = np.random.default_rng(40 + kv_heads).normal(size=(2, 8, 32))
        full_output, full_cache = model.forward(x)
        decoded_output, decoded_cache = _decode_one_token_at_a_time(model, x)
        _assert_close(decoded_output, full_output, f"Hkv={kv_heads} token decoding")
        _assert_close(decoded_cache[0], full_cache[0], f"Hkv={kv_heads} decoded K")
        _assert_close(decoded_cache[1], full_cache[1], f"Hkv={kv_heads} decoded V")


def _test_chunked_decode_matches_full_forward() -> None:
    model = GroupQueryAttention(24, 6, 2, seed=51)
    x = np.random.default_rng(52).normal(size=(2, 11, 24))
    full_output, full_cache = model.forward(x)
    boundaries = (0, 3, 5, 10, 11)
    cache = None
    pieces = []
    for start, end in zip(boundaries, boundaries[1:]):
        output, cache = model.forward(x[:, start:end], cache)
        pieces.append(output)
    _assert_close(np.concatenate(pieces, axis=1), full_output, "chunked output")
    _assert_close(cache[0], full_cache[0], "chunked key cache")
    _assert_close(cache[1], full_cache[1], "chunked value cache")


def _test_nonempty_cache_uses_rectangular_causal_mask() -> None:
    model = GroupQueryAttention(16, 4, 2, seed=61)
    x = np.random.default_rng(62).normal(size=(1, 8, 16))
    _, prefix_cache = model.forward(x[:, :3])
    expected, expected_cache = _reference_forward(model, x[:, 3:7], prefix_cache)
    actual, actual_cache = model.forward(x[:, 3:7], prefix_cache)
    _assert_close(actual, expected, "multi-token cached output")
    _assert_close(actual_cache[0], expected_cache[0], "multi-token cached K")
    _assert_close(actual_cache[1], expected_cache[1], "multi-token cached V")


def _test_direct_nonempty_cache_across_attention_modes() -> None:
    configurations = (
        (1, 1, 1, 16, 4, 4),
        (2, 5, 4, 24, 6, 2),
        (3, 7, 1, 32, 8, 1),
        (2, 2, 6, 8, 4, 2),
    )
    for case, (
        batch,
        prefix_length,
        new_length,
        dimension,
        query_heads,
        kv_heads,
    ) in enumerate(configurations):
        model = GroupQueryAttention(
            dimension, query_heads, kv_heads, seed=130 + case
        )
        rng = np.random.default_rng(140 + case)
        prefix = rng.normal(size=(batch, prefix_length, dimension))
        new_tokens = rng.normal(size=(batch, new_length, dimension))
        prefix_cache = _project_compact_cache(model, prefix)

        expected, expected_cache = _reference_forward(
            model, new_tokens, prefix_cache
        )
        actual, actual_cache = model.forward(new_tokens, prefix_cache)

        label = f"direct cache configuration {case}"
        _assert_close(actual, expected, f"{label} output")
        _assert_close(actual_cache[0], expected_cache[0], f"{label} K")
        _assert_close(actual_cache[1], expected_cache[1], f"{label} V")
        expected_shape = (
            batch,
            kv_heads,
            prefix_length + new_length,
            model.head_dim,
        )
        if actual_cache[0].shape != expected_shape:
            raise AssertionError(
                f"{label} cache must remain compact with shape {expected_shape}"
            )


def _test_explicit_zero_length_cache_matches_none() -> None:
    model = GroupQueryAttention(16, 4, 2, seed=151)
    x = np.random.default_rng(152).normal(size=(2, 5, 16))
    empty_shape = (2, model.num_kv_heads, 0, model.head_dim)
    empty_cache = (
        np.empty(empty_shape, dtype=np.float64),
        np.empty(empty_shape, dtype=np.float64),
    )

    expected, expected_cache = model.forward(x, None)
    actual, actual_cache = model.forward(x, empty_cache)
    _assert_close(actual, expected, "zero-length cache output")
    _assert_close(actual_cache[0], expected_cache[0], "zero-length cache K")
    _assert_close(actual_cache[1], expected_cache[1], "zero-length cache V")


def _test_uniform_attention_cache_boundary_exact_values() -> None:
    model = GroupQueryAttention(4, 2, 1, seed=161)
    model.Wq = np.zeros_like(model.Wq)
    model.Wk = np.zeros_like(model.Wk)
    model.Wv = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    )
    model.Wo = np.eye(4)

    prefix = np.array([[[2.0, 0.0, 9.0, 9.0], [0.0, 4.0, 9.0, 9.0]]])
    new_tokens = np.array(
        [[[6.0, 2.0, 9.0, 9.0], [10.0, 6.0, 9.0, 9.0], [14.0, 10.0, 9.0, 9.0]]]
    )
    prefix_cache = _project_compact_cache(model, prefix)
    actual, _ = model.forward(new_tokens, prefix_cache)

    all_values = np.concatenate((prefix[..., :2], new_tokens[..., :2]), axis=1)
    expected_rows = []
    for local_position in range(new_tokens.shape[1]):
        visible_count = prefix.shape[1] + local_position + 1
        mean_value = np.mean(all_values[0, :visible_count], axis=0)
        expected_rows.append(np.tile(mean_value, model.num_query_heads))
    expected = np.asarray(expected_rows)[None, :, :]
    _assert_close(actual, expected, "uniform cached causal boundary")


def _test_cached_chunk_future_tokens_do_not_change_earlier_outputs() -> None:
    model = GroupQueryAttention(24, 6, 2, seed=171)
    rng = np.random.default_rng(172)
    prefix = rng.normal(size=(2, 5, 24))
    new_tokens = rng.normal(size=(2, 4, 24))
    changed = new_tokens.copy()
    changed[:, 2:] = rng.normal(loc=1000.0, scale=50.0, size=(2, 2, 24))
    prefix_cache = _project_compact_cache(model, prefix)

    original_output, original_cache = model.forward(new_tokens, prefix_cache)
    changed_output, changed_cache = model.forward(changed, prefix_cache)
    _assert_close(
        changed_output[:, :2],
        original_output[:, :2],
        "cached chunk causal prefix",
    )
    _assert_close(
        original_cache[0][:, :, : prefix.shape[1]],
        prefix_cache[0],
        "original cached key prefix",
    )
    _assert_close(
        changed_cache[1][:, :, : prefix.shape[1]],
        prefix_cache[1],
        "changed cached value prefix",
    )


def _test_cache_growth_does_not_mutate_prefix() -> None:
    model = GroupQueryAttention(16, 4, 1, seed=71)
    x = np.random.default_rng(72).normal(size=(2, 6, 16))
    _, prefix_cache = model.forward(x[:, :2])
    saved = tuple(item.copy() for item in prefix_cache)
    _, grown_cache = model.forward(x[:, 2:], prefix_cache)
    _assert_close(prefix_cache[0], saved[0], "original key cache")
    _assert_close(prefix_cache[1], saved[1], "original value cache")
    _assert_close(grown_cache[0][:, :, :2], saved[0], "grown key prefix")
    _assert_close(grown_cache[1][:, :, :2], saved[1], "grown value prefix")


def _test_future_tokens_do_not_change_prefix() -> None:
    model = GroupQueryAttention(24, 6, 3, seed=81)
    rng = np.random.default_rng(82)
    original = rng.normal(size=(2, 7, 24))
    changed = original.copy()
    changed[:, 4:] = rng.normal(loc=500.0, scale=100.0, size=(2, 3, 24))
    original_output, _ = model.forward(original)
    changed_output, _ = model.forward(changed)
    _assert_close(changed_output[:, :4], original_output[:, :4], "causal prefix")


def _test_query_heads_keep_separate_distributions() -> None:
    model = GroupQueryAttention(4, 2, 1, seed=91)
    model.Wq = np.eye(4)
    model.Wo = np.eye(4)
    x = np.random.default_rng(92).normal(size=(1, 4, 4))
    expected, _ = _reference_forward(model, x)
    actual, _ = model.forward(x)
    _assert_close(actual, expected, "per-query-head distributions")
    if np.allclose(expected[0, -1, :2], expected[0, -1, 2:]):
        raise AssertionError("query heads sharing KV must still compute separate attention")


def _test_output_projection_is_applied() -> None:
    model = GroupQueryAttention(8, 4, 2, seed=101)
    x = np.random.default_rng(102).normal(size=(1, 4, 8))
    expected, _ = _reference_forward(model, x)
    actual, _ = model.forward(x)
    _assert_close(actual, expected, "output projection")
    model.Wo = np.zeros_like(model.Wo)
    zero_output, _ = model.forward(x)
    _assert_close(zero_output, np.zeros_like(zero_output), "zero output projection")


def _test_batches_are_independent_and_input_is_unchanged() -> None:
    model = GroupQueryAttention(12, 3, 1, seed=111)
    x = np.random.default_rng(112).normal(size=(3, 6, 12))
    saved = x.copy()
    batched, _ = model.forward(x)
    for batch_index in range(3):
        separate, _ = model.forward(x[batch_index : batch_index + 1])
        _assert_close(separate[0], batched[batch_index], f"batch {batch_index}")
    np.testing.assert_array_equal(x, saved)


def _test_large_scores_remain_finite() -> None:
    model = GroupQueryAttention(8, 4, 2, seed=121)
    model.Wq *= 10000.0
    model.Wk *= 10000.0
    x = np.random.default_rng(122).normal(size=(2, 5, 8))
    output, _ = model.forward(x)
    expected, _ = _reference_forward(model, x)
    if not np.all(np.isfinite(output)):
        raise AssertionError("large finite scores must produce finite outputs")
    _assert_close(output, expected, "large-score output", atol=1e-8, rtol=1e-8)


def run_tests() -> bool:
    """Run numerical, causal-mask, grouping, and cache tests."""
    if np is None:
        print("[ENVIRONMENT ERROR] NumPy is missing. Run: python -m pip install numpy")
        return False

    tests = (
        ("MHA, GQA, and MQA numerical values", _test_full_forward_matches_reference),
        ("compact KV cache shape", _test_cache_is_compact),
        ("projected KV cache values", _test_cache_contains_projected_values),
        ("token decoding equals full forward", _test_token_decode_matches_full_forward),
        ("chunked decoding equals full forward", _test_chunked_decode_matches_full_forward),
        ("rectangular cached causal mask", _test_nonempty_cache_uses_rectangular_causal_mask),
        ("direct nonempty cache for MHA, GQA, and MQA", _test_direct_nonempty_cache_across_attention_modes),
        ("explicit zero-length cache", _test_explicit_zero_length_cache_matches_none),
        ("uniform cached causal boundary values", _test_uniform_attention_cache_boundary_exact_values),
        ("cached chunk causal prefix invariance", _test_cached_chunk_future_tokens_do_not_change_earlier_outputs),
        ("cache growth and prefix immutability", _test_cache_growth_does_not_mutate_prefix),
        ("causal prefix invariance", _test_future_tokens_do_not_change_prefix),
        ("separate query-head distributions", _test_query_heads_keep_separate_distributions),
        ("output projection", _test_output_projection_is_applied),
        ("batch independence and input immutability", _test_batches_are_independent_and_input_is_unchanged),
        ("large-score numerical stability", _test_large_scores_remain_finite),
    )

    print("Grouped Query Attention Automated Tests")
    print("=" * 58)
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

    print("-" * 58)
    print(f"Result: {passed}/{len(tests)} tests passed")
    if passed != len(tests):
        print("Update forward() and run again. Use --show-answer for help.")
    return passed == len(tests)


_ENCODED_ANSWER = (
    "c-p;JTW{Mo6n@XII5SW{+GHhhiai)p(QVyUAXw3MMfPAA3T@FgVTn>m%8`ft_#TqFQL>Y44>g)9K"
    "0250+&(hvlt>AGh`cO5l~f#mR*G`Paz4KR{7fd3Ur8xR3R|s0lj(!j0!h=&lr*~V$kYx9tn;iOYX"
    "!G=uqNvr4bl9<_(|vn;+S!!;@C41{n1{6_FEEXc|y}UFs?c&FSFRhjCYhI^2FQy;p8hxX<|eW!Ic"
    "&JFnt3ba!%*17TW+XfXRVae*|HY!k>9kru0Lue$Pvu{3b*$JV(ux>B|mXU?tg>XIK*u^r2*eqT4e"
    "2cnsll$>&fU)eeg_&7fcylS@U?RHslJCyq<M53eo@{Df&hE`|kh<V`7z@KiQ;BrT~;V)Bk5REh0F"
    "<U9dwh#f4sl8}{B!HPnWOQzU?PEI0vHNx`U3&J+mO^k#cm}IMz!s<A5=*AsJd*$Z*lHfxZXF)Vx5"
    "rH$VVFN#E%HX~p28f2J$LehYkrUZf)S@_HSr->BFQ4{pHa3>*A38?S7`MS%Q^B`dy$A&kMV^JYxu"
    "i^0jB-z}RM!!?CaHFP9bUnuwvZ1(c{MuU#q#$9Dx!;!=OQDiyFpjsRbaj01K~!ElVAk<d$$^0lwr"
    "L6|Bu)9<Hf-a-!fSKpMpJ8zyL=5@;k=f<l>Qt<Q!pFbKP~G5FyC16C3KEgDQ)ZE_w<*<t6-U2p3b"
    ")j~+XZt)BF=>=qu+n$J{~f&-_IQ<d*2hrCopsVu*#9fh(;h@we*P~G2qw*lPw_xHU6t4+`D0g)z@"
    "S7EKxv7iykHfc`OkFOnest^?JC$i9ursFD<J5tca^%A0*p&QX`skQoS@*HhdEm3hSS5~N_J#vjd@"
    "O}y4udb7Q%32Y%-UX0SzE${)F{K(stD-m4tKe<3sD6m&mez1b%X)G|wO2<rVzm>mJ7PNVvYpIsAV"
    "Y;*B=Ibw9**}cfSUlW1L&0>J09=dJoZoK(N<)P*M;!}3WRXuLO8e(4&Mkut56WBV(V64div98#{2"
    "X=2QMM<J}?iRfT?9*^J(>HDhF037V4R&wlfhN6<QT^jq4T{f^L{zC$CO@l(O#_BHr;G%LZ(h?Da?"
    "iH_U>0fzIA8fmW(*+@(09znI(I9c1>b*-f|cqpq6H&POiBBw$-;h+v&_MHq)$!)$k!2NFXB@AroQ"
    "J_hh^=TH1`$J*&|@XeOviLmU4k=96=9Dh8o>A4vZA4jHQGMf1#+rv5ALp@hL+un5|*B9mp3F)mxu"
    "$Tvc|I7)Yf0-{s`A7;<QRz)_^rtg_Xnh&A7m>kn+G_6UF}C+wee<;xk88E1o(bAk#9!2byV}@;v~"
    "S$6=?^_uR^%^g_??SQiANNTGa_|?T`X^g0mik#2#-4|287acT7&NYZz|_9?G*hmz8TQ32@(u10%m"
    "kXz9K7@BCG}mu7Fp=n(FTvsp{`_Z*_<ElonWh8GR8Y_5Cyc>ZX;M=g80c>24ah2DH+D;JFU%{2a#"
    "f(^_U!Y;j?t#)ZqQ_I#P?Bj)7{KKyf_gDLE_q>kB~4tW4xpL+d^ugRBu!2"
)


def _decode_answer() -> str:
    compressed = base64.b85decode(_ENCODED_ANSWER.encode("ascii"))
    return zlib.decompress(compressed).decode("utf-8")


def _show_answer() -> None:
    """Print only the complete reference class."""
    print(_decode_answer(), end="")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped Query Attention assignment")
    parser.add_argument(
        "--show-answer",
        action="store_true",
        help="Print only the reference GroupQueryAttention class and skip tests.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv or sys.argv[1:])
    if arguments.show_answer:
        _show_answer()
        return 0
    return 0 if run_tests() else 1


if __name__ == "__main__":
    raise SystemExit(main())
