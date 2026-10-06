# kalman-py

**Fast, exact and numerically robust Kalman filters for Python.**

kalman-py provides linear, extended and unscented Kalman filters with RTS smoothers,
maximum-likelihood noise learning and consistency diagnostics. It runs on NumPy out of the box,
and on an optional JAX backend that compiles the whole time loop.

<div class="grid cards" markdown>

- **⚡ Fast**

    The JAX backend runs a 2-D tracking filter at about 0.6 µs per step, 67× faster than
    pykalman. See [Benchmarks](benchmarks.md).

- **🎯 Exact**

    Same estimates as FilterPy and pykalman on linear problems, to rounding error. See
    [Migrating from FilterPy](migration.md).

- **🧮 Automatic Jacobians**

    Write the model once with `jax.numpy`; the EKF differentiates it and the UKF needs no
    Jacobians. See [Nonlinear filters](guide/nonlinear.md).

- **🛡️ Robust**

    Exactly symmetric covariances, and square-root forms that stay positive-definite even in
    float32. See [Numerical robustness](guide/robustness.md).

</div>

## Install

kalman-py isn't on PyPI yet. From a clone of the repository:

```bash
pip install -e .              # NumPy only
pip install -e ".[jax,plot]"  # with the JAX backend and plotting helpers
```

## Where to go next

- [Getting started](getting-started.md): a first filter, the conventions, and what a result
  contains.
- The [tutorials](tutorials/01_tracking.ipynb): target tracking, multi-rate sensor fusion and
  learning the noise from data, as runnable notebooks in `examples/`.
- The [API reference](api/filters.md).
