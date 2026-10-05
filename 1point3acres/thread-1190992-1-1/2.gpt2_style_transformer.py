"""Tiny GPT-2-style Transformer assignment implemented with NumPy.

Goal
----
Implement every computational element:

1. ``_softmax``
2. ``_gelu``
3. ``LayerNorm.__call__``
4. ``CausalSelfAttention.forward``
5. ``MLP.forward``
6. ``GPT2Block.forward``
7. ``TinyGPT2.forward``

The assignment focuses on causal attention, residual connections, and the
exact placement of LayerNorm in a GPT-2-style pre-norm Transformer.

Tensor shapes
-------------
    x:                 (B, T, D)
    q, k, v:           (B, H, T, Dh)
    attention_scores:  (B, H, T, T)
    attention_weights: (B, H, T, T)

    B  = batch size
    T  = sequence length
    D  = model dimension
    H  = number of attention heads
    Dh = head dimension = D / H

Core element equations
----------------------
    stable_softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi)
              * (x + 0.044715 * x^3)))

    mean = average(x, axis=-1)
    variance = average((x - mean)^2, axis=-1)
    LayerNorm(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta

    MLP(x) = GELU(x @ W1 + b1) @ W2 + b2

Causal multi-head self-attention
--------------------------------
    Q = x @ Wq + bq
    K = x @ Wk + bk
    V = x @ Wv + bv

    scores = (Q @ K.T) / sqrt(Dh)
    scores[i, j] = -infinity when j > i
    weights = softmax(scores, axis=-1)
    context = weights @ V
    output = concat(context_heads) @ Wo + bo

The causal mask prevents position i from reading any future position j > i.

GPT-2 pre-LayerNorm block
-------------------------
    after_attention = x + Attention(LayerNorm1(x))
    output = after_attention + MLP(LayerNorm2(after_attention))

LayerNorm is applied before each sublayer. The residual connection adds the
sublayer output back to the value that entered that residual branch. A final
LayerNorm is applied after the last Transformer block.

This differs from a post-norm block, which would apply LayerNorm after each
residual addition. GPT-2 uses the pre-norm arrangement shown above.

Usage
-----
    python 2.gpt2_style_transformer.py                  # Run every test.
    python 2.gpt2_style_transformer.py --test_softmax   # Run one stage.
    python 2.gpt2_style_transformer.py --test_gelu
    python 2.gpt2_style_transformer.py --test_layernorm
    python 2.gpt2_style_transformer.py --test_attention # Run one stage.
    python 2.gpt2_style_transformer.py --test_mlp
    python 2.gpt2_style_transformer.py --test_block
    python 2.gpt2_style_transformer.py --test_model
    python 2.gpt2_style_transformer.py --show-answer    # Show solutions.

Multiple test flags may be combined. Edit only student sections.
"""

from __future__ import annotations

import argparse
import base64
import sys
import zlib

from numpy import var

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax along the selected axis."""
    # ===================== Student section: softmax =====================
    # stable_softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    x = np.exp(x - np.max(x,axis=-1,keepdims=True))
    return x / np.sum(x,axis=-1,keepdims=True)
    # =================== End student section: softmax ===================


def _gelu(x: np.ndarray) -> np.ndarray:
    """Compute the tanh approximation of the GPT-2 GELU activation."""

    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    # ======================= Student section: GELU ======================
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    # ===================== End student section: GELU ====================


class LayerNorm:
    """LayerNorm over the final dimension with learnable scale and bias."""

    def __init__(self, d_model: int, epsilon: float = 1e-5) -> None:
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.epsilon = epsilon

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Normalize the final dimension, then apply gamma and beta."""
        # mean = average(x, axis=-1)
        # variance = average((x - mean)^2, axis=-1)
        # LayerNorm(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta

        # =================== Student section: LayerNorm ===================
        mean = np.mean(x,axis=-1,keepdims=True)
        variance = np.mean((x - mean) ** 2,axis=-1,keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(variance + self.epsilon) + self.beta
        # ================= End student section: LayerNorm =================


class CausalSelfAttention:
    """Multi-head self-attention with a left-to-right causal mask."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rng: np.random.Generator,
    ) -> None:
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
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
        """Return causal multi-head attention output with shape (B, T, D)."""
        # ================= Student section: causal attention =================
        """
            scores = (Q @ K.T) / sqrt(Dh)
            scores[i, j] = -infinity when j > i
            weights = softmax(scores, axis=-1)
            context = weights @ V
            output = concat(context_heads) @ Wo + bo
        """
        B, T, _ = x.shape
        Q = (x @ self.Wq + self.bq).reshape(B,T,self.num_heads,self.head_dim) # (B,T,H,D')
        K = (x @ self.Wk + self.bk).reshape(B,T,self.num_heads,self.head_dim) # (B,T,H,D')
        V = (x @ self.Wv + self.bv).reshape(B,T,self.num_heads,self.head_dim) # (B,T,H,D')

        # (B,T,H,D') -> (B,H,T,D')
        Q = Q.transpose(0,2,1,3) # (B,H,T,D')
        K = K.transpose(0,2,1,3) # (B,H,T,D')
        V = V.transpose(0,2,1,3) # (B,H,T,D')
        scores = (Q @ K.transpose(0,1,3,2)) / np.sqrt(self.head_dim) # (B, H, T, T)

        # mask
        casual_mask = np.triu(np.ones((T,T), dtype=bool),k=1)
        scores = np.where(casual_mask,-np.inf, scores)
        weight = _softmax(scores,axis=-1) # (B, H, T, T)
        self.last_attention_weights = weight.copy()
        result = weight @ V # (B, H, T, T) * (B,H,T,D') = (B,H,T,D')
        result = result.transpose(0,2,1,3) # (B,H,T,D') -> (B,T,H,D')
        result = result.reshape(B, T, self.d_model)
        return result @ self.Wo + self.bo
        # =============== End student section: causal attention ===============


class MLP:
    """The position-wise GPT-2 feed-forward network."""

    def __init__(self, d_model: int, rng: np.random.Generator) -> None:
        hidden_dim = 4 * d_model
        scale = 1.0 / np.sqrt(d_model)
        self.W1 = rng.normal(0.0, scale, size=(d_model, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0.0, scale, size=(hidden_dim, d_model))
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the expansion, GELU, and projection layers."""
        # ======================= Student section: MLP ======================
        raise NotImplementedError("Implement MLP.forward()")
        # ===================== End student section: MLP ====================


class GPT2Block:
    """One pre-LayerNorm GPT-2 Transformer block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rng: np.random.Generator,
    ) -> None:
        self.ln1 = LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads, rng)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply pre-norm attention and MLP residual branches."""
        # ================== Student section: GPT-2 block ==================
        raise NotImplementedError("Implement GPT2Block.forward()")
        # ================ End student section: GPT-2 block ================


class TinyGPT2:
    """Token embeddings, GPT-2 blocks, final LayerNorm, and a tied LM head."""

    def __init__(
        self,
        vocab_size: int = 17,
        context_length: int = 8,
        d_model: int = 8,
        num_heads: int = 2,
        num_layers: int = 2,
        seed: int = 0,
    ) -> None:
        if np is None:
            raise ModuleNotFoundError(
                "This assignment requires NumPy. Run: python -m pip install numpy"
            )
        if min(vocab_size, context_length, num_layers) <= 0:
            raise ValueError("vocab_size, context_length, and num_layers must be positive")

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = rng.normal(0.0, scale, size=(vocab_size, d_model))
        self.position_embedding = rng.normal(
            0.0, scale, size=(context_length, d_model)
        )
        self.blocks = [GPT2Block(d_model, num_heads, rng) for _ in range(num_layers)]
        self.ln_f = LayerNorm(d_model)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Return next-token logits with shape (B, T, vocab_size)."""
        # ==================== Student section: TinyGPT2 ====================
        raise NotImplementedError("Implement TinyGPT2.forward()")
        # ================== End student section: TinyGPT2 ==================


def _assert_close(
    actual: np.ndarray,
    expected: np.ndarray,
    name: str,
    atol: float = 1e-8,
    rtol: float = 1e-7,
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


def _test_softmax() -> None:
    x = np.array([[1000.0, 1001.0, 1002.0], [-3.0, 0.0, 3.0]])
    actual = _softmax(x, axis=-1)
    expected = np.array(
        [
            [0.09003057317038046, 0.24472847105479764, 0.6652409557748218],
            [0.0023556330807966803, 0.04731415522182404, 0.9503302116973793],
        ]
    )
    _assert_close(actual, expected, "softmax")
    _assert_close(np.sum(actual, axis=-1), np.ones(2), "softmax row sums")
    if not np.all(np.isfinite(actual)):
        raise AssertionError("softmax must remain finite for large input values")


def _test_gelu() -> None:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = np.array(
        [-0.04540230591222494, -0.15880800939172324, 0.0,
         0.8411919906082768, 1.954597694087775]
    )
    _assert_close(_gelu(x), expected, "GELU")


def _test_layer_norm() -> None:
    layer_norm = LayerNorm(4)
    x = np.array([[[1.0, 2.0, 4.0, 8.0], [-2.0, 0.0, 2.0, 4.0]]])
    output = layer_norm(x)
    _assert_close(np.mean(output, axis=-1), np.zeros((1, 2)), "LayerNorm mean")
    _assert_close(
        np.var(output, axis=-1),
        np.ones((1, 2)),
        "LayerNorm variance",
        atol=1e-5,
        rtol=1e-5,
    )
    layer_norm.gamma = np.array([1.0, 2.0, 0.5, -1.0])
    layer_norm.beta = np.array([0.2, -0.3, 0.4, 0.1])
    affine_output = layer_norm(x)
    normalized = output
    _assert_close(
        affine_output,
        normalized * layer_norm.gamma + layer_norm.beta,
        "LayerNorm gamma and beta",
    )


def _test_attention_mask_and_probabilities() -> None:
    rng = np.random.default_rng(3)
    attention = CausalSelfAttention(d_model=8, num_heads=2, rng=rng)
    x = rng.normal(size=(2, 4, 8))
    output = attention.forward(x)
    if output.shape != x.shape:
        raise AssertionError(f"attention output must have shape {x.shape}")

    weights = attention.last_attention_weights
    if weights is None or weights.shape != (2, 2, 4, 4):
        raise AssertionError("last_attention_weights must have shape (B, H, T, T)")
    _assert_close(np.sum(weights, axis=-1), np.ones((2, 2, 4)), "attention rows")
    future_positions = np.triu(np.ones((4, 4), dtype=bool), k=1)
    if not np.all(weights[..., future_positions] == 0.0):
        raise AssertionError("causal attention must assign zero weight to future positions")


def _test_attention_causality() -> None:
    rng = np.random.default_rng(5)
    attention = CausalSelfAttention(d_model=8, num_heads=2, rng=rng)
    original = rng.normal(size=(1, 5, 8))
    changed = original.copy()
    changed[:, 3:, :] += 100.0
    original_output = attention.forward(original)
    changed_output = attention.forward(changed)
    _assert_close(
        original_output[:, :3],
        changed_output[:, :3],
        "causal prefix output",
        atol=1e-7,
        rtol=1e-7,
    )


def _test_mlp() -> None:
    rng = np.random.default_rng(9)
    mlp = MLP(d_model=4, rng=rng)
    x = np.array([[[0.2, -0.1, 0.4, 0.7], [-0.5, 0.3, 0.8, -0.2]]])
    expected = np.array(
        [[
            [-0.62912910, 0.49657761, -0.60816312, 0.57936374],
            [-0.36798767, 1.20015662, -0.19822444, 0.24003330],
        ]]
    )
    _assert_close(mlp.forward(x), expected, "MLP output", atol=1e-7, rtol=1e-7)


def _test_pre_norm_residual_positions() -> None:
    rng = np.random.default_rng(7)
    block = GPT2Block(d_model=8, num_heads=2, rng=rng)
    x = rng.normal(size=(2, 3, 8))

    normalized_for_attention = block.ln1(x)
    after_attention = x + block.attention.forward(normalized_for_attention)
    normalized_for_mlp = block.ln2(after_attention)
    expected = after_attention + block.mlp.forward(normalized_for_mlp)
    actual = block.forward(x)
    _assert_close(actual, expected, "GPT-2 pre-norm residual block")


def _test_tiny_gpt2_end_to_end() -> None:
    model = TinyGPT2(
        vocab_size=13,
        context_length=6,
        d_model=8,
        num_heads=2,
        num_layers=2,
        seed=11,
    )
    token_ids = np.array([[1, 4, 2, 8], [3, 0, 5, 6]])
    logits = model.forward(token_ids)
    if logits.shape != (2, 4, 13):
        raise AssertionError(f"logits must have shape (2, 4, 13), got {logits.shape}")
    if not np.all(np.isfinite(logits)):
        raise AssertionError("logits must contain only finite values")


_TEST_GROUPS = {
    "softmax": (("stable softmax", _test_softmax),),
    "gelu": (("GELU values", _test_gelu),),
    "layernorm": (("LayerNorm normalization and affine transform", _test_layer_norm),),
    "attention": (
        ("attention mask and probabilities", _test_attention_mask_and_probabilities),
        ("attention causality", _test_attention_causality),
    ),
    "mlp": (("MLP projection", _test_mlp),),
    "block": (("pre-norm residual positions", _test_pre_norm_residual_positions),),
    "model": (("TinyGPT2 end-to-end shape", _test_tiny_gpt2_end_to_end),),
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
    print("Tiny GPT-2-Style Transformer Automated Tests")
    print("=" * 56)
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

    print("-" * 56)
    print(f"Result: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("Congratulations! The GPT-2-style Transformer passed all checks.")
        return True
    print("Update the student sections and run again. Use --show-answer for help.")
    return False


_ENCODED_ANSWER = (
    "c-pO2OOM+&5Wf3Yu*pSoY{mAbD6ohZX@j7bv<T2edk6x7mPQsKiIhpotL?Av@F80Ay3Pj4"
    "9v0#}K7KR9nej`KDpBZ$_Ol#{aESU_!rO%BR7(1|Ahc)dmN2f#mMqsXS>BD*w=;r&s$_+R9"
    "1VK1G+GPeT({=mf>Z!)&KkA7mmS2`3;I@Y;F{4|p)0zfB=T6!M1t-lpIW{)ma1!_vGm$q"
    "v)N0M?xF5nnC5F5<HeXV0Y$+w1~zbW<w5FbldK$pmQ5J0lAp;Oqo0##jV7;*iKe`a#xS"
    "VtQMF1|Z{GZ}whr_8O&mLlAL%2=KZR_PG|gyTr>W!X50QepSdgz1Z-8>g*nEXDEY}x{aH"
    "Nt^p25f?GDS3BF`3WF=DI&e!n9`p+Uk!yUx$>18(t9;LISN~wcwNVS;d|<4aKC4Aic{D"
    "G_^T;Pdi2Hzfkt8)~57=Cxws?ROU8W{`=Xv>Rs4WS((XIt{+?27PX+-F=Pel(TO$Wr!Bc5"
    "LgG=Cw1v;h^%o<v4D+-RIn)l-21c0z@;9wJ_#mZ_(TkomozkSFM<5n}L_4apGF2Gr0$~m"
    "vV4Bhz_+FRD#6iX{@Oc=-tH!D6J;%!$A8aqabL9RxghU>!wwh?Gpsn1Km}rkddz=O6Ln6"
    "Umc0B~^$Gavif#%Ae&3B6Vj02VGhoKqj#~+5~BGr;|)d~gCYGJwo``~8G{r~lj&1Joto0"
    "&ka<`ztMm8b{W(jHV~dTqHGH_-)lneBzNp{jDPSWDJL;Q^umr=yCZ3z1I*@$l$(LezMuw"
    "rdL#PbnhaLkSY1(Y?qL#f)RGIXma`0Bm1s;}mv^Q-@LFu}x4lRXU|ZH>W3G|26Z|m}}_u"
    "NsI|Qo@xBom4ZCAeg0D!+HpL#2E3tHMCcV^$JHMnKYiDDDp{U`8RmXzg^SkVqV<H3cNok"
    "z!E9ss{Ql{F^II*l>bqD`+|Tk<GCM8&J@kAkyN6rzb=1d>spxdunG&4Cc!;KM&z(2XnQc"
    "4?+9BEi?-_qIL0*YZiwbzka`n|XgXMHd1aa68K@#@F=Gr1S;eN|PFnhq?$Avb8m{Q%&w("
    "f~jV$Tb!#<#f#8Gi#6^PVZ=#Jj3t^`uYOU`kGRp4JYMJ2$*@W%aG%rPnwO0K@}8p8+7lD"
    "Z(flSQ<Vm3wcXcvDwp46nB1EA4NuYHp|aU&X5;sAKOwCOB^wKa48s)WPVFnc;^Sr4)UDw{"
    "pW!5CD?_A!7;_`+&h`yo@LN^;UgwH)4Et3-_z!#P5~I)-MG22<)*}Ho)&}9_f8loXQY$+"
    "*?(~g=>q"
)


def _show_answer() -> None:
    answer = zlib.decompress(base64.b85decode(_ENCODED_ANSWER)).decode("utf-8")
    print("Reference solution (copy each method body into its student section):")
    print("=" * 72)
    print(answer, end="")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tiny GPT-2-style Transformer attention and residual assignment"
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
    parser.add_argument("--test_softmax", action="store_true", help="test softmax")
    parser.add_argument("--test_gelu", action="store_true", help="test GELU")
    parser.add_argument(
        "--test_layernorm",
        "--test_layer_norm",
        dest="test_layernorm",
        action="store_true",
        help="test LayerNorm",
    )
    parser.add_argument(
        "--test_attention",
        "--test_self_attention",
        dest="test_attention",
        action="store_true",
        help="test causal self-attention",
    )
    parser.add_argument("--test_mlp", action="store_true", help="test the MLP")
    parser.add_argument(
        "--test_block", action="store_true", help="test the GPT-2 block"
    )
    parser.add_argument(
        "--test_model", action="store_true", help="test the complete TinyGPT2 model"
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
