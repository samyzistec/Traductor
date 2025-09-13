
# Test Suite for Transformer_Nahuatl_Espanol_FromScratch.ipynb

This test suite **does not execute training**. It dynamically **introspects** the notebook, extracts only **imports**, **function/class definitions**, and simple **constant assignments**, and then runs **lightweight sanity tests** on any matching building blocks it finds.

> ✅ Works even if your code only lives in the notebook — no refactor required.

## What it checks

- Presence of core symbols (e.g., `PositionalEncoding`, `MultiHeadAttention`, `Transformer`, `greedy_decode`, `compute_bleu`, `set_seed`).
- Shape contracts for `PositionalEncoding` and `MultiHeadAttention` (if present).
- Determinism in `set_seed` (if present).
- `compute_bleu` returns a numeric score for a trivial input (if present).
- `greedy_decode` can be called with a minimal dummy model interface (if present).

Any missing symbol results in a **skip**, not a failure. To enforce failures, run with `--strict-nb`.

## File layout

```
/mnt/data
├── Transformer_Nahuatl_Espanol_FromScratch.ipynb
├── pyproject.toml
├── README_tests.md
└── tests
    ├── conftest.py
    └── test_core.py
```

## How to run

```bash
pip install pytest nbformat
pytest -q
# or enforce no-skips for missing pieces:
pytest -q --strict-nb
```

## Notes

- The introspector keeps only **definitions** and **imports**, so heavy cells with training loops won’t run.
- If your class/function names differ, the suite will **skip** tests for those components. You can rename or add aliases to match if you want stricter coverage.
- Extend `tests/test_core.py` with additional checks specific to your implementation (e.g., label smoothing, masking shapes, padding handling).

Happy testing!
