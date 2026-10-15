"""Temperature, top-k, top-p, and EOS sampling assignment using NumPy.

Goal
----
Implement every stage of autoregressive token sampling:

1. ``stable_softmax``
2. ``apply_temperature``
3. ``top_k_filter``
4. ``top_p_filter``
5. ``sample_next_token``
6. ``generate``

Core equations and rules
------------------------
Given vocabulary logits ``z`` and temperature ``T > 0``:

    scaled_logits = z / T
    probability[i] = exp(scaled_logits[i]) / sum(exp(scaled_logits))

For numerical stability, subtract the maximum logit before ``exp``:

    softmax(z) = exp(z - max(z)) / sum(exp(z - max(z)))

Temperature behavior:

    T < 1  makes the distribution sharper.
    T = 1  leaves the logits unchanged.
    T > 1  makes the distribution flatter.

Top-k keeps only the ``k`` largest logits and sets every other logit to
negative infinity.

Top-p, also called nucleus sampling, sorts tokens by probability and keeps the
smallest prefix whose cumulative probability reaches or exceeds ``p``. The
first token that crosses the threshold must remain in the candidate set.

Sampling order used by this assignment:

    logits
      -> divide by temperature
      -> apply top-k
      -> apply top-p
      -> stable softmax
      -> sample one token

Generation appends one sampled token at a time. It stops immediately after an
EOS token is appended, or after ``max_new_tokens`` tokens have been generated.
The returned sequence includes the EOS token.

Usage
-----
    python 5.temperature_topk_topp_eos.py
    python 5.temperature_topk_topp_eos.py --test_softmax
    python 5.temperature_topk_topp_eos.py --test_temperature
    python 5.temperature_topk_topp_eos.py --test_top_k
    python 5.temperature_topk_topp_eos.py --test_top_p
    python 5.temperature_topk_topp_eos.py --test_sampling
    python 5.temperature_topk_topp_eos.py --test_eos
    python 5.temperature_topk_topp_eos.py --show-answer

Multiple test flags may be combined. Edit only student sections.
"""

from __future__ import annotations

import argparse
import base64
from math import exp, log
import sys
import zlib
from collections.abc import Callable

from networkx import is_empty

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Return a numerically stable probability distribution."""
    # ===================== Student section: softmax =====================
    logits = np.exp(logits - max(logits))
    return logits / sum(logits)
    # =================== End student section: softmax ===================


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by a strictly positive temperature."""
    # =================== Student section: temperature ===================
    # scaled_logits = z / T
    # probability[i] = exp(scaled_logits[i]) / sum(exp(scaled_logits))
    return logits / temperature
    # raise NotImplementedError("Implement apply_temperature()")
    # ================= End student section: temperature =================


def top_k_filter(logits: np.ndarray, top_k: int | None) -> np.ndarray:
    """Keep the top-k logits and replace all other values with -infinity."""
    # ====================== Student section: top-k ======================
    if not top_k:
        return logits
    index = np.argpartition(logits,-top_k)[-top_k:]
    ans = np.full_like(logits,-np.inf)
    ans[index] = logits[index]
    return ans
    # ==================== End student section: top-k ====================


def top_p_filter(logits: np.ndarray, top_p: float | None) -> np.ndarray:
    """Keep the smallest high-probability prefix with cumulative mass >= p."""
    # ====================== Student section: top-p ======================
    if not top_p:
        return logits
    sorted_index = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_index]
    sorted_porb = stable_softmax(sorted_logits)
    cumulate_sum = np.cumsum(sorted_porb)
    remove = cumulate_sum > top_p
    if np.flatnonzero(remove).size > 0:
        remove[np.flatnonzero(remove)[0]] = False # make sure the first > top_p will also not be removed
    else:
        remove[0] = False
    kept_index = sorted_index[~remove]
    ans = np.full_like(logits,-np.inf)
    ans[kept_index] = logits[kept_index]
    # print(logits)
    # print(sorted_index)
    # print(sorted_logits)
    # print(sorted_porb)
    # print(cumulate_sum)
    # print(remove)
    # print(kept_index)
    # print(ans)
    return ans
    
    # ==================== End student section: top-p ====================


def sample_next_token(
    logits: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Apply all sampling stages and return one token ID."""
    # ==================== Student section: sampling ====================
    logits = apply_temperature(logits=logits,temperature=temperature)
    logits = top_k_filter(logits=logits,top_k=top_k)
    logits = top_p_filter(logits=logits,top_p=top_p)
    probability = stable_softmax(logits=logits)
    return int(rng.choice(probability.size,p=probability))
    # ================== End student section: sampling ==================


def generate(
    next_logits_fn: Callable[[np.ndarray], np.ndarray],
    prompt_token_ids: np.ndarray,
    *,
    max_new_tokens: int,
    eos_token_id: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> np.ndarray:
    """Generate tokens autoregressively and stop after EOS or the token limit."""
    # ====================== Student section: EOS =======================
    ans = prompt_token_ids.tolist()
    if ans and ans[-1] == eos_token_id:
        return ans

    for _ in range(max_new_tokens):
        logits = next_logits_fn(np.asarray(ans,dtype=int))
        next_token = sample_next_token(
            logits=logits,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        ans.append(next_token)
        if next_token == eos_token_id:
            break
    return np.asarray(ans,dtype=int)

    # ==================== End student section: EOS =====================


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


def _finite_indices(values: np.ndarray) -> set[int]:
    return set(np.flatnonzero(np.isfinite(values)).tolist())


def _test_softmax_large_logits() -> None:
    logits = np.array([1000.0, 1001.0, 1002.0])
    expected = np.array(
        [0.09003057317038046, 0.24472847105479764, 0.6652409557748218]
    )
    probabilities = stable_softmax(logits)
    _assert_close(probabilities, expected, "stable softmax")
    if not np.all(np.isfinite(probabilities)):
        raise AssertionError("softmax probabilities must be finite")


def _test_softmax_translation_invariance() -> None:
    logits = np.array([-3.2, 0.7, 2.1, -0.4, 4.8])
    _assert_close(
        stable_softmax(logits + 50000.0),
        stable_softmax(logits),
        "softmax translation invariance",
    )


def _test_temperature_exact_scaling() -> None:
    logits = np.array([-2.0, 0.5, 3.0, 7.5])
    _assert_close(
        apply_temperature(logits, 0.5),
        np.array([-4.0, 1.0, 6.0, 15.0]),
        "temperature 0.5",
    )
    _assert_close(
        apply_temperature(logits, 2.0),
        np.array([-1.0, 0.25, 1.5, 3.75]),
        "temperature 2.0",
    )


def _test_temperature_changes_sharpness() -> None:
    logits = np.array([-1.0, 0.2, 1.5, 3.0])
    cold = stable_softmax(apply_temperature(logits, 0.4))
    normal = stable_softmax(apply_temperature(logits, 1.0))
    hot = stable_softmax(apply_temperature(logits, 2.5))
    if not cold.max() > normal.max() > hot.max():
        raise AssertionError("lower temperature must produce a sharper distribution")


def _test_temperature_one_preserves_logits() -> None:
    logits = np.array([0.2, -4.0, 1.7, 3.3])
    original = logits.copy()
    _assert_close(apply_temperature(logits, 1.0), original, "temperature 1.0")
    _assert_close(logits, original, "temperature input immutability")


def _test_top_k_keeps_exact_candidates() -> None:
    logits = np.array([0.2, 3.0, -1.0, 1.5, 0.7])
    filtered = top_k_filter(logits, 2)
    if _finite_indices(filtered) != {1, 3}:
        raise AssertionError("top-k=2 must keep token IDs 1 and 3")
    _assert_close(filtered[[1, 3]], logits[[1, 3]], "top-k retained logits")


def _test_top_k_one_is_greedy_candidate() -> None:
    logits = np.array([-2.0, 0.3, 5.0, 1.2])
    filtered = top_k_filter(logits, 1)
    if _finite_indices(filtered) != {2}:
        raise AssertionError("top-k=1 must keep only the argmax token")


def _test_top_k_covering_vocabulary_is_no_op() -> None:
    logits = np.array([1.3, -0.7, 4.1, 2.2])
    original = logits.copy()
    _assert_close(top_k_filter(logits, 4), original, "top-k full vocabulary")
    _assert_close(logits, original, "top-k input immutability")


def _nucleus_logits() -> np.ndarray:
    probabilities = np.array([0.15, 0.50, 0.05, 0.30])
    return np.log(probabilities)


def _test_top_p_keeps_token_crossing_threshold() -> None:
    filtered = top_p_filter(_nucleus_logits(), 0.70)
    if _finite_indices(filtered) != {1, 3}:
        raise AssertionError(
            "top-p=0.70 must keep probabilities 0.50 and 0.30, including the crossing token"
        )


def _test_top_p_below_first_probability() -> None:
    filtered = top_p_filter(_nucleus_logits(), 0.49)
    if _finite_indices(filtered) != {1}:
        raise AssertionError("top-p=0.49 must keep only the 0.50-probability token")


def _test_top_p_preserves_original_token_order() -> None:
    logits = _nucleus_logits()
    filtered = top_p_filter(logits, 0.90)
    if _finite_indices(filtered) != {0, 1, 3}:
        raise AssertionError("top-p=0.90 selected the wrong original token IDs")
    _assert_close(filtered[[0, 1, 3]], logits[[0, 1, 3]], "top-p retained logits")


def _test_top_p_one_is_no_op() -> None:
    logits = _nucleus_logits()
    original = logits.copy()
    _assert_close(top_p_filter(logits, 1.0), original, "top-p=1.0")
    _assert_close(logits, original, "top-p input immutability")


def _test_sampling_top_k_one_is_deterministic() -> None:
    logits = np.array([0.1, 2.0, -1.0, 0.7])
    for seed in range(10):
        token = sample_next_token(logits, np.random.default_rng(seed), top_k=1)
        if token != 1:
            raise AssertionError("top-k=1 sampling must always return the argmax token")


def _test_sampling_is_reproducible_with_seed() -> None:
    logits = np.array([-0.2, 0.4, 1.1, 2.0, -1.5])
    first_rng = np.random.default_rng(41)
    second_rng = np.random.default_rng(41)
    first = [sample_next_token(logits, first_rng) for _ in range(30)]
    second = [sample_next_token(logits, second_rng) for _ in range(30)]
    if first != second:
        raise AssertionError("the same random seed must produce the same samples")


def _test_sampling_combines_top_k_and_top_p() -> None:
    logits = np.log(np.array([0.40, 0.30, 0.20, 0.10]))
    rng = np.random.default_rng(43)
    samples = {
        sample_next_token(logits, rng, top_k=3, top_p=0.60)
        for _ in range(200)
    }
    if not samples <= {0, 1}:
        raise AssertionError("combined top-k/top-p sampled a filtered token")
    if samples != {0, 1}:
        raise AssertionError("combined top-k/top-p did not sample both valid candidates")


def _test_generation_stops_after_eos() -> None:
    eos_token_id = 4

    def next_logits(tokens: np.ndarray) -> np.ndarray:
        logits = np.full(6, -10.0)
        logits[2 if len(tokens) < 3 else eos_token_id] = 10.0
        return logits

    generated = generate(
        next_logits,
        np.array([1]),
        max_new_tokens=10,
        eos_token_id=eos_token_id,
        rng=np.random.default_rng(47),
        top_k=1,
    )
    expected = np.array([1, 2, 2, 4])
    if not np.array_equal(generated, expected):
        raise AssertionError(f"expected EOS sequence {expected}, got {generated}")


def _test_generation_respects_max_new_tokens() -> None:
    def next_logits(tokens: np.ndarray) -> np.ndarray:
        logits = np.array([-3.0, -2.0, -1.0, 5.0, 0.0])
        return logits

    generated = generate(
        next_logits,
        np.array([2, 1]),
        max_new_tokens=3,
        eos_token_id=4,
        rng=np.random.default_rng(53),
        top_k=1,
    )
    expected = np.array([2, 1, 3, 3, 3])
    if not np.array_equal(generated, expected):
        raise AssertionError(f"expected length-limited sequence {expected}, got {generated}")


def _test_generation_does_not_continue_after_prompt_eos() -> None:
    calls = 0

    def next_logits(tokens: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.array([0.0, 1.0, 2.0, 3.0])

    prompt = np.array([2, 1, 3])
    generated = generate(
        next_logits,
        prompt,
        max_new_tokens=5,
        eos_token_id=3,
        rng=np.random.default_rng(59),
    )
    if not np.array_equal(generated, prompt):
        raise AssertionError("a prompt ending in EOS must be returned unchanged")
    if calls != 0:
        raise AssertionError("the model must not be called after a prompt-ending EOS")


_TEST_GROUPS = {
    "softmax": (
        ("large-logit stability", _test_softmax_large_logits),
        ("translation invariance", _test_softmax_translation_invariance),
    ),
    "temperature": (
        ("exact logit scaling", _test_temperature_exact_scaling),
        ("distribution sharpness", _test_temperature_changes_sharpness),
        ("temperature one", _test_temperature_one_preserves_logits),
    ),
    "top_k": (
        ("exact top-k candidates", _test_top_k_keeps_exact_candidates),
        ("top-k one", _test_top_k_one_is_greedy_candidate),
        ("full-vocabulary top-k", _test_top_k_covering_vocabulary_is_no_op),
    ),
    "top_p": (
        ("threshold-crossing token", _test_top_p_keeps_token_crossing_threshold),
        ("threshold below first probability", _test_top_p_below_first_probability),
        ("original token order", _test_top_p_preserves_original_token_order),
        ("top-p one", _test_top_p_one_is_no_op),
    ),
    "sampling": (
        ("deterministic top-k one", _test_sampling_top_k_one_is_deterministic),
        ("seed reproducibility", _test_sampling_is_reproducible_with_seed),
        ("combined top-k and top-p", _test_sampling_combines_top_k_and_top_p),
    ),
    "eos": (
        ("stop after EOS", _test_generation_stops_after_eos),
        ("max-new-token limit", _test_generation_respects_max_new_tokens),
        ("prompt already ends in EOS", _test_generation_does_not_continue_after_prompt_eos),
    ),
}


def run_tests(selected_groups: list[str] | None = None) -> bool:
    """Run selected test groups, or every group when none are selected."""
    if np is None:
        print("[ENVIRONMENT ERROR] NumPy is missing. Run: python -m pip install numpy")
        return False

    groups = selected_groups or list(_TEST_GROUPS)
    tests = [
        (group, test_name, test)
        for group in groups
        for test_name, test in _TEST_GROUPS[group]
    ]
    print("Temperature, Top-k, Top-p, and EOS Sampling Tests")
    print("=" * 58)
    passed = 0
    current_group = None
    for group, name, test in tests:
        if group != current_group:
            print(f"Stage: {group}")
            current_group = group
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
    if passed == len(tests):
        print("Congratulations! The sampling implementation passed all checks.")
        return True
    print("Update the student sections and run again. Use --show-answer for help.")
    return False


_ENCODED_ANSWER = (
    "c-qBQO>f&U487-9Xzj%rk}SCzbuYuV+m5>!fuYoLDnyn9S?QW$*l!>8VOvt1WWY8oho&Y"
    "UX+1uQ%suKf-Z4^jqR}mWoz<jCOc@h<s=FL<x#Bfe3ZLd^`4~LrGXx(8LxpkR&N_VP0@e"
    "B?NzpKY_1uEHXWD2|1KNizP1yK0ha#_1aKbgk%qIp*=3^a15q(rVMt(5J?$~Cbbv~QT?h"
    "%%fohnUQNfd@iVhNvFCpa%q;LnlcGF1oKgh=#Ip!I6)T=u}2b_9LH>_|Q-B~*47qT6Gq("
    "T<?KB3KiJ^fTt@jVN&!(Y;Y(s6a^35Z6LhhpM3rvRcleVVbT1<cM;OexR>V$3It{`qv`b"
    "*^EdiZ+Pq*Vb#$$5>(6a^}Z(|Pg#=6)c0z2Evm1p6dl#{#lG2`0Lp4(4@ByU@|r%8o{ZW"
    "{tTdnn9~dodwt4Bk^R10_HMGqPam|sjiqQiZgTP98GmpS!_{QyU4`OpDc^^2aDN{A&Ej6W"
    "Y+rO=+O!bsT&rQ_zH*}Q0b5AU;SCVqL9OerczhH*4^#X0Sy`G&=Fe1(rHRgW8lzG0~Y(w"
    "<G-D?piiJ(%69o|s}?P1d=BQr|qh&~<LBg3Zm&Vu1=;57qlD!XT-NGqUki_3`~T_po_xye"
    "my`G<k7e_k8kXBdrS@(#r8%bzalzqNdaTL~5B<W*N%90+%-Dz#=o7e_b1CFXl;RG8PIU4"
    "0?k%t)b}_r3c%pYzCev-hTB5uE00^d{GA@Z0+aad}$8OY0|EIrB6s8sNR%kL^jYoCZq=zR"
    "}TrOtgl8Oa>)gJv|E$FN;#xLB2q;h@081;k_%o+rIsSlg>)%_*V_j(MQafIZ0WLVr~~GeC"
    "W^LO5X$(ttZ6y@&X&^m+ubQStXr=h;9fE%-4J7pEgT6oE^t_IH$b(F%9n=Cal9iF504Za"
    "J*|%HqAE-TsYPe5<tyhBVu+pWRDAq9<J}>1l?!nf-gDQ+qu>c@xC!k1A7WuFxu&HcZ?8F"
    "XM8GQ)S;q?B~6!+PwAJDyD_l9DzjQZ>^;fiv(Cp1<CcuNH4AhZ&JqeTA2N&+%iOi;DHgwN"
    "5!s+CV?)r0Meu(DGW}TCzvm8V9ddMxrya<wU{(=cXX8;v5Xcd-x>SAh*}>1-!|cK@+R4h"
    "zegSbJCK3"
)


def _show_answer() -> None:
    answer = zlib.decompress(base64.b85decode(_ENCODED_ANSWER)).decode("utf-8")
    print("Reference solution (copy each implementation into its student section):")
    print("=" * 76)
    print(answer, end="")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Temperature, top-k, top-p, and EOS sampling assignment"
    )
    parser.add_argument(
        "--show-answer",
        action="store_true",
        help="show the reference solution without running tests",
    )
    parser.add_argument(
        "--test_all",
        action="store_true",
        help="run every test stage (also the default when no flag is given)",
    )
    for group in _TEST_GROUPS:
        parser.add_argument(
            f"--test_{group}",
            action="store_true",
            help=f"run only the {group} test stage",
        )
    arguments = parser.parse_args()
    if arguments.show_answer:
        _show_answer()
        return 0

    selected_groups = [
        group for group in _TEST_GROUPS if getattr(arguments, f"test_{group}")
    ]
    if arguments.test_all or not selected_groups:
        selected_groups = None
    return 0 if run_tests(selected_groups) else 1


if __name__ == "__main__":
    sys.exit(main())
