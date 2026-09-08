import os

# Must be set before any test module imports JAX.
os.environ.setdefault("JAX_NUM_CPU_DEVICES", "3")
