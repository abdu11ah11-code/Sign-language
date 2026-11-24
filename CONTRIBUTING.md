# Contributing to SignBridge

Thanks for helping extend SignBridge. The most valuable contributions are
**new signs**, **broader training data**, and **validation by Deaf signers**.

## Development setup

```bash
pip install -e ".[dev]"
pytest                      # 52 tests, no camera/GPU needed
ruff check src tests        # lint
mypy src                    # strict type checking
```

The test suite mocks MediaPipe and synthesizes landmark windows, so it runs
anywhere — including CI without a webcam or TensorFlow GPU.

## Add a new sign

1. **Record it** (this also seeds the playback video):
   ```bash
   sign-collect --label WELCOME --mode asl --samples 50
   ```
   Record diverse takes: vary lighting, distance, and angle. If multiple
   people can sign, collect from all of them — diversity is what makes the
   model generalize.

2. **Add it to the lexicon** so Speech→Sign knows the word. Edit
   `src/sign_bridge/language/asl_lexicon.py` (or `psl_lexicon.py`):
   ```python
   LexiconEntry("WELCOME", "welcome"),
   ```
   For PSL, include the Urdu transliteration:
   ```python
   PslEntry("WELCOME", "welcome", "khush aamdeed"),
   ```

3. **Retrain**:
   ```bash
   sign-train --language asl
   ```

That's it — the new sign is recognized (Sign→Speech) and playable
(Speech→Sign).

## Add a new language

1. Create `src/sign_bridge/language/<lang>_lexicon.py` following the existing
   pattern (`known_signs()`, `all_labels()`, and any script-specific helpers).
2. Allow the new code in the CLIs: extend the `choices=["asl", "psl"]`
   arguments in `data/collector.py`, `models/trainer.py`,
   `models/evaluator.py`.
3. Add the language option to the sidebar radio in
   `ui/components/controls.py`.
4. Collect data and train as above. The `capture`, `models`, and `inference`
   layers are language-agnostic and need no changes.

## Improve recognition

- **More data** beats almost any modeling change for small vocabularies.
- Try the transformer: `sign-train --architecture transformer` and compare
  the `*.metrics.json` and confusion matrices.
- Tune `config.yaml` — window size, confidence threshold, lock frames,
  augmentation magnitudes. All values are validated on load.

## Code standards

- **Type hints everywhere**; `mypy --strict` must pass.
- **Google-style docstrings** on public functions and classes.
- Keep the **162-dim feature layout** stable — many modules depend on the
  exact offsets defined in `capture/holistic.py`. If you must change it,
  update `FEATURE_DIM`, the slices, and retrain all models.
- Add or update tests for any logic change. Prefer testing pure functions
  (preprocessing, smoothing, tokenization) over I/O.
- Run `ruff` and `mypy` before opening a PR.

## A note on respect

PSL and ASL belong to their communities. When adding signs or languages,
ground your choices in community-published resources (e.g. FESF for PSL) and,
wherever possible, have Deaf signers review them. Document your sources.
