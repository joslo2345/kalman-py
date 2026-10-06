"""Shared test fixtures and JAX setup."""

try:
    import jax
except ImportError:
    pass
else:
    # JAX defaults to float32; equivalence tests compare against float64 libraries.
    jax.config.update("jax_enable_x64", True)
