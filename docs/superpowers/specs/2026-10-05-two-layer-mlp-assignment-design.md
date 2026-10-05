# Two-Layer MLP Assignment Design

## Goal

Turn `1point3acres/thread-1190992-1-1/2-layer-MLP.py` into a self-contained
NumPy assignment. A learner implements the forward and backward passes of a
two-layer multilayer perceptron, then runs the file to receive immediate,
deterministic feedback.

## Model and loss

The network uses the following computation:

1. `z1 = X @ W1 + b1`
2. `h1 = ReLU(z1)`
3. `scores = h1 @ W2 + b2`
4. `loss = mean((scores - y) ** 2)`

The mean is taken over every element of `scores - y`. The backward pass must
therefore include the factor `2 / (batch_size * output_size)`.

## Learner interface

The file defines a `TwoLayerMLP` class initialized with input, hidden, and
output dimensions plus a deterministic seed. The learner edits only two
clearly marked methods:

- `forward(X, y)` returns `(loss, scores)` and saves the values required by
  backpropagation.
- `backward()` consumes the latest forward-pass cache and returns a dictionary
  containing `dX`, `dW1`, `db1`, `dW2`, and `db2`.

Calling `backward()` before `forward()` produces a clear error. Inputs with
incompatible shapes also produce concise validation errors.

## Self-checking behavior

Running the file without arguments executes deterministic tests that check:

- forward output shapes and numerical values;
- input immutability;
- every analytical gradient's shape and value;
- numerical gradients using centered finite differences;
- multiple small cases, including inactive ReLU units;
- the expected error when `backward()` is called without a cache.

Gradient comparison uses both absolute and relative tolerances and reports the
name of a failing gradient, its maximum absolute error, and its maximum
relative error. A missing implementation is reported as an unfinished
assignment instead of exposing a traceback. The program exits with status 0
only when all checks pass and with a nonzero status otherwise.

## Reference answer

The same file accepts `--show-answer`. In this mode it prints a complete
reference implementation of the two learner methods and does not run tests.
The answer text is compressed and encoded in the source so it is not visible
by default. This is a convenience barrier, not a security boundary; a learner
who inspects the source can recover it.

The test oracle is implemented independently of the encoded answer. Thus the
ordinary test path never needs to decode or execute the reference answer.

## Dependencies and scope

The assignment requires only Python and NumPy. It performs no parameter
updates or training loop because the requested learning objective is manual
forward propagation, backpropagation, and numerical gradient checking.
