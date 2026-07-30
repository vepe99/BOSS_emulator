# Experimental

Code that is not part of the supported inverse-modelling workflow.

- **`run_nuts.py`** — NumPyro NUTS posterior sampler. **Does not converge** on
  this posterior (chains freeze at their start: `r_hat` in the hundreds,
  `ESS ~ 1`). Kept as a starting point for future posterior/uncertainty work.
  Requires `numpyro` and `optimistix`. Run from the project root so it can
  import `common` from the parent directory:

  ```bash
  .venv/bin/python script/inv_model/experimental/run_nuts.py --run <run_dir> \
      --n-targets 8 --max-tree-depth 7
  ```
