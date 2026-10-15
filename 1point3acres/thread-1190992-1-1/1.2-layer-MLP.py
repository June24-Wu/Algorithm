"""Two-layer MLP implementation assignment using NumPy.

Goal: Implement ``TwoLayerMLP.forward`` and ``TwoLayerMLP.backward``.

Architecture: X -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> MSE Loss

Notation and shapes:
    X:  (N, D)       input batch
    W1: (D, H)       first-layer weights
    b1: (H,)         first-layer bias
    W2: (H, C)       second-layer weights
    b2: (C,)         second-layer bias
    y:  (N, C)       regression targets

Forward equations:
    z1     = X @ W1 + b1
    ac1    = ReLU(z1) = maximum(0, z1)
    z2     = ac1 @ W2 + b2
    loss   = (1 / (N * C)) * sum((z2 - y) ** 2)

Backward equations:
    dz2 = (2 / (N * C)) * (z2 - y)
    dW2 = ac1.T @ dz2
    db2 = sum(dz2, axis=0)
    dac1 = dz2 @ W2.T
    dz1 = dac1 * indicator(z1 > 0)
    dW1 = X.T @ dz1
    db1 = sum(dz1, axis=0)
    dX = dz1 @ W1.T

Central-difference gradient check for any scalar parameter theta:
    numerical_gradient(theta)
        = (loss(theta + epsilon) - loss(theta - epsilon)) / (2 * epsilon)

Usage:
    python 2-layer-MLP.py                 # Run the automated tests.
    python 2-layer-MLP.py --show-answer   # Show the reference solution.

Requirements:
* ``forward(X, y)`` returns ``(loss, scores)`` and caches values for backward.
* MSE is averaged over every element in the batch and output dimensions.
* ``backward()`` returns a dictionary with dX, dW1, db1, dW2, and db2.
* Edit only the two student sections. Do not modify the tests.


"""

from __future__ import annotations

import argparse
import base64
import sys
import zlib
from collections.abc import Callable

try:
    import numpy as np
except ModuleNotFoundError:  # Keep --show-answer available when NumPy is absent.
    np = None  # type: ignore[assignment]


class TwoLayerMLP:
    """A two-layer MLP that computes forward and backward passes only."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seed: int = 0,
    ) -> None:
        if np is None:
            raise ModuleNotFoundError(
                "This assignment requires NumPy. Run: python -m pip install numpy"
            )
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive")

        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.2, size=(input_dim, hidden_dim))
        self.b1 = rng.normal(0.0, 0.2, size=(hidden_dim,))
        self.W2 = rng.normal(0.0, 0.2, size=(hidden_dim, output_dim))
        self.b2 = rng.normal(0.0, 0.2, size=(output_dim,))
        self.cache: dict[str, np.ndarray] | None = None

    def forward(self, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.W1.shape[0]:
            raise ValueError("X 必须是形状为 (batch_size, input_dim) 的二维数组")

        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        scores = h1 @ self.W2 + self.b2
        if y.shape != scores.shape:
            raise ValueError(f"y 的形状必须是 {scores.shape}，实际为 {y.shape}")

        difference = scores - y
        loss = float(np.mean(difference ** 2))
        self.cache = {
            "X": X.copy(),
            "z1": z1,
            "h1": h1,
            "scores": scores,
            "y": y.copy(),
        }
        return loss, scores


    def backward(self) -> dict[str, np.ndarray]:
        if self.cache is None:
            raise RuntimeError("必须先调用 forward()，再调用 backward()")

        X = self.cache["X"]
        z1 = self.cache["z1"]
        h1 = self.cache["h1"]
        scores = self.cache["scores"]
        y = self.cache["y"]

        dscores = 2.0 * (scores - y) / scores.size
        dW2 = h1.T @ dscores
        db2 = np.sum(dscores, axis=0)
        dh1 = dscores @ self.W2.T
        dz1 = dh1 * (z1 > 0.0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        dX = dz1 @ self.W1.T

        return {"dX": dX, "dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def _make_fixed_case() -> tuple[TwoLayerMLP, np.ndarray, np.ndarray]:
    """Create deterministic data with active and inactive ReLU units."""
    model = TwoLayerMLP(2, 3, 2, seed=0)
    model.W1 = np.array([[0.5, -0.4, 0.3], [-0.2, 0.6, -0.5]])
    model.b1 = np.array([0.1, -0.2, 0.05])
    model.W2 = np.array([[0.7, -0.3], [0.2, 0.5], [-0.4, 0.8]])
    model.b2 = np.array([0.05, -0.1])
    X = np.array([[0.4, -0.7], [1.2, 0.3]])
    y = np.array([[0.2, -0.4], [0.8, 0.1]])
    return model, X, y


def _assert_close(
    actual: np.ndarray | float,
    expected: np.ndarray | float,
    name: str,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> None:
    """Report numerical errors with assignment-friendly diagnostics."""
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    if actual_array.shape != expected_array.shape:
        raise AssertionError(
            f"{name} has the wrong shape: expected {expected_array.shape}, "
            f"got {actual_array.shape}"
        )

    absolute_error = np.abs(actual_array - expected_array)
    relative_error = absolute_error / np.maximum(
        1e-12, np.abs(actual_array) + np.abs(expected_array)
    )
    if not np.allclose(actual_array, expected_array, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name} has incorrect values: max absolute error="
            f"{absolute_error.max():.3e}, max relative error="
            f"{relative_error.max():.3e}"
        )


def _numerical_gradient(
    value: np.ndarray,
    loss_function: Callable[[], float],
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Compute the numerical gradient of every element via central differences."""
    gradient = np.zeros_like(value, dtype=float)
    for index in np.ndindex(value.shape):
        original = value[index]
        value[index] = original + epsilon
        loss_plus = float(loss_function())
        value[index] = original - epsilon
        loss_minus = float(loss_function())
        value[index] = original
        gradient[index] = (loss_plus - loss_minus) / (2.0 * epsilon)
    return gradient


def _test_forward_values_and_input_safety() -> None:
    model, X, y = _make_fixed_case()
    X_before = X.copy()
    y_before = y.copy()
    loss, scores = model.forward(X, y)

    expected_scores = np.array([[0.150, 0.184], [0.394, -0.084]])
    _assert_close(scores, expected_scores, "scores")
    _assert_close(loss, 0.13556200000000004, "loss")
    if not np.array_equal(X, X_before) or not np.array_equal(y, y_before):
        raise AssertionError("forward() must not modify X or y in place")


def _test_forward_when_all_relu_units_are_active() -> None:
    model = TwoLayerMLP(1, 1, 1, seed=0)
    model.W1[:] = 2.0
    model.b1[:] = 1.0
    model.W2[:] = 3.0
    model.b2[:] = -1.0
    X = np.array([[0.5], [1.5], [2.5]])
    y = np.array([[4.0], [10.0], [16.0]])

    loss, scores = model.forward(X, y)

    _assert_close(scores, np.array([[5.0], [11.0], [17.0]]), "all-active scores")
    _assert_close(loss, 1.0, "all-active loss")


def _test_forward_when_all_relu_units_are_inactive() -> None:
    model = TwoLayerMLP(2, 2, 2, seed=0)
    model.W1[:] = np.array([[0.3, -0.2], [0.1, 0.4]])
    model.b1[:] = np.array([-10.0, -10.0])
    model.W2[:] = np.array([[2.0, -1.0], [0.5, 3.0]])
    model.b2[:] = np.array([0.25, -0.75])
    X = np.array([[1.0, -2.0], [-3.0, 0.5], [2.0, 1.0]])
    y = np.array([[0.0, 0.0], [1.0, -1.0], [-0.5, 0.5]])

    loss, scores = model.forward(X, y)

    expected_scores = np.tile(model.b2, (X.shape[0], 1))
    expected_loss = float(np.mean((expected_scores - y) ** 2))
    _assert_close(scores, expected_scores, "all-inactive scores")
    _assert_close(loss, expected_loss, "all-inactive loss")

    gradients = model.backward()
    for name in ("dX", "dW1", "db1", "dW2"):
        _assert_close(gradients[name], np.zeros_like(gradients[name]), name)


def _test_relu_zero_uses_zero_derivative() -> None:
    model = TwoLayerMLP(1, 1, 1, seed=0)
    model.W1[:] = 2.0
    model.b1[:] = -2.0
    model.W2[:] = 4.0
    model.b2[:] = 0.5
    X = np.array([[1.0]])  # z1 is exactly zero.
    y = np.array([[2.0]])

    model.forward(X, y)
    gradients = model.backward()

    _assert_close(gradients["dX"], np.zeros((1, 1)), "dX at ReLU zero")
    _assert_close(gradients["dW1"], np.zeros((1, 1)), "dW1 at ReLU zero")
    _assert_close(gradients["db1"], np.zeros(1), "db1 at ReLU zero")
    _assert_close(gradients["dW2"], np.zeros((1, 1)), "dW2 at ReLU zero")
    _assert_close(gradients["db2"], np.array([-3.0]), "db2 at ReLU zero")


def _test_batch_loss_equals_mean_of_sample_losses() -> None:
    model, X, y = _make_fixed_case()
    batch_loss, _ = model.forward(X, y)

    sample_losses = []
    for sample_index in range(X.shape[0]):
        sample_loss, _ = model.forward(
            X[sample_index : sample_index + 1],
            y[sample_index : sample_index + 1],
        )
        sample_losses.append(sample_loss)

    _assert_close(batch_loss, np.mean(sample_losses), "batch MSE decomposition")


def _test_forward_shape_validation() -> None:
    model, X, y = _make_fixed_case()
    try:
        model.forward(X[:, :1], y)
    except ValueError:
        pass
    else:
        raise AssertionError("forward() must raise ValueError for an invalid X shape")

    try:
        model.forward(X, y[:, :1])
    except ValueError:
        pass
    else:
        raise AssertionError("forward() must raise ValueError for an invalid y shape")


def _test_backward_requires_forward() -> None:
    model, _, _ = _make_fixed_case()
    try:
        model.backward()
    except RuntimeError as error:
        # NotImplementedError is a RuntimeError subclass; it must not count as a pass.
        if type(error) is RuntimeError:
            return
        raise
    raise AssertionError("backward() must raise RuntimeError before forward() is called")


def _test_backward_with_numerical_gradients() -> None:
    model, X, y = _make_fixed_case()
    model.forward(X, y)
    analytical = model.backward()

    expected_names = {"dX", "dW1", "db1", "dW2", "db2"}
    if set(analytical) != expected_names:
        raise AssertionError(
            f"backward() keys must be {sorted(expected_names)}, "
            f"got {sorted(analytical)}"
        )

    numerical = {
        "dX": _numerical_gradient(X, lambda: model.forward(X, y)[0]),
        "dW1": _numerical_gradient(model.W1, lambda: model.forward(X, y)[0]),
        "db1": _numerical_gradient(model.b1, lambda: model.forward(X, y)[0]),
        "dW2": _numerical_gradient(model.W2, lambda: model.forward(X, y)[0]),
        "db2": _numerical_gradient(model.b2, lambda: model.forward(X, y)[0]),
    }
    for name in ("dX", "dW1", "db1", "dW2", "db2"):
        _assert_close(analytical[name], numerical[name], name, atol=2e-6, rtol=2e-5)


def _test_numerical_gradients_with_batch_and_single_output() -> None:
    model = TwoLayerMLP(3, 2, 1, seed=0)
    model.W1[:] = np.array([[0.2, -0.3], [0.4, 0.5], [-0.6, 0.7]])
    model.b1[:] = np.array([0.4, -0.5])
    model.W2[:] = np.array([[0.8], [-0.2]])
    model.b2[:] = np.array([0.1])
    X = np.array([[0.2, -0.1, 0.3], [-0.4, 0.6, 0.2], [0.1, 0.5, -0.2]])
    y = np.array([[0.3], [-0.2], [0.8]])

    model.forward(X, y)
    analytical = model.backward()
    numerical = {
        "dX": _numerical_gradient(X, lambda: model.forward(X, y)[0]),
        "dW1": _numerical_gradient(model.W1, lambda: model.forward(X, y)[0]),
        "db1": _numerical_gradient(model.b1, lambda: model.forward(X, y)[0]),
        "dW2": _numerical_gradient(model.W2, lambda: model.forward(X, y)[0]),
        "db2": _numerical_gradient(model.b2, lambda: model.forward(X, y)[0]),
    }
    for name in ("dX", "dW1", "db1", "dW2", "db2"):
        _assert_close(
            analytical[name],
            numerical[name],
            f"single-output {name}",
            atol=2e-6,
            rtol=2e-5,
        )


def run_tests() -> bool:
    """Run all assignment tests and return True only when every test passes."""
    if np is None:
        print("[ENVIRONMENT ERROR] NumPy is missing. Run: python -m pip install numpy")
        return False

    tests = (
        ("forward values and input safety", _test_forward_values_and_input_safety),
        ("forward with all ReLU units active", _test_forward_when_all_relu_units_are_active),
        ("forward with all ReLU units inactive", _test_forward_when_all_relu_units_are_inactive),
        ("ReLU zero derivative convention", _test_relu_zero_uses_zero_derivative),
        ("batch loss decomposition", _test_batch_loss_equals_mean_of_sample_losses),
        ("forward shape validation", _test_forward_shape_validation),
        ("backward call order", _test_backward_requires_forward),
        ("backward numerical gradients", _test_backward_with_numerical_gradients),
        ("single-output numerical gradients", _test_numerical_gradients_with_batch_and_single_output),
    )
    print("Two-Layer MLP Automated Tests")
    print("=" * 48)
    passed = 0
    for name, test in tests:
        try:
            test()
        except NotImplementedError as error:
            print(f"[INCOMPLETE] {name}: {error}")
        except AssertionError as error:
            print(f"[FAILED] {name}: {error}")
        except Exception as error:  # Keep routine assignment failures concise.
            print(f"[ERROR] {name}: {type(error).__name__}: {error}")
        else:
            passed += 1
            print(f"[PASSED] {name}")

    print("-" * 48)
    print(f"Result: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("Congratulations! Both forward and backward passed all checks.")
        return True
    print("Update the student sections and run again. Use --show-answer for help.")
    return False


_ENCODED_ANSWER = (
    "c-nnbJ&)5s5bgad23PjsGqKA^InmHkpcAnaMW?mBmMs$7TH69AO2q{sbaWsA1rk(La0No>"
    "0^-kHE?mW5V0PCZ8*#!#H2Y?DX5Q?ZgfxOEV~+_7je^c13%ntuxt)fDF;ZJlt49+KkD#p"
    "b8J$G4jFgt*n+`R=&w~*MNg+VPvmh+%oQ_1iDMIS~koty@Bmxf;Cv&(p0v9rd$08;<oj6k"
    "t@mB4z(_n+9Hvbq&3JMR&tfKcA%b1~iu>G~Z{;;`v^JV+<<L<?`%kLLp_@oTtlOkDA3z9"
    "Ui$`h<+!tUMa<;B@<^Y!ZW`EGNno0`bHaIj|&?zV;DwzPalxMN3Bnv*BVyqX&e$AW=9kX"
    "(u&W3<2kjA|3OT>`h|TT3#6owy;-|6vyCHSbrF>Nvtuy}tVM<81r+{q@^5XT5BStXf`S5"
    "=E5JG@#Iyg+r)?Z<ZCjYw;=uCr3$YC>IV6z%?6+oJv4~n5S8G9qXPx#D@#Ayf#d$hZhdw"
    "IHMkkITEXgYz9$jD4rU@dN1ip{4!crEER2A%`HvSLKL`9f=8Xgg@(f<C?`e9ZV)g&g+eD"
    "O;C`0=^9m2Dv`pr$!sP6?>*v4EpYC3L>K4l6!+-X&LAx%@cKUo*`dSkl*R)rg>cD|bd!e"
    "bGxb<`@Q@za)r`_7A&Khwkg6^(s58wcdUI-@K>C}j;BEwO=T+H?{%DK5K9lk5qpuiPq9"
    "2Ve86r*MpLeYO)uUiKDSb7C-9)mrgebnzzI2H>iOD>$IoTHTEE4NpQ50_``b{4CsI$7!=m"
    "uHCoP&%gP=Gfu3FKl1fV^`p=z=iF*EA21tJfNo"
)


def _show_answer() -> None:
    answer = zlib.decompress(base64.b85decode(_ENCODED_ANSWER)).decode("utf-8")
    print("Reference solution (copy each method body into its student section):")
    print("=" * 64)
    print(answer, end="")


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-layer MLP forward/backward assignment")
    parser.add_argument(
        "--show-answer",
        action="store_true",
        help="show the forward and backward reference solution without running tests",
    )
    arguments = parser.parse_args()
    if arguments.show_answer:
        _show_answer()
        return 0
    return 0 if run_tests() else 1


if __name__ == "__main__":
    sys.exit(main())
