# CLAUDE.md

Classical molecular dynamics (MD) simulator for liquid water, ethanol and their mixtures in a periodic cubic box (the project goal; O2 and H2 are only test molecules, kept out of the README and notebook). Python package `jostle` in `src/` (NumPy, SciPy, Numba, tqdm), used from `notebooks/jostle.ipynb`; there is no CLI.

**Read [docs/internals.md](docs/internals.md) first.** It is the single reference for the settings, warnings, `Results` and output files, the model, the package map, data representations, implementation details, units and input-file conversions, performance, tests and code conventions. Don't duplicate it here.

## Commands

- Install: `pip install -e .[dev]` (or `.[notebook]`). Tests run without installing.
- Tests: `python -m pytest`. Lint: `ruff check src tests`.
- Quick end-to-end run: `run(Config(box_length=2.0, end_time=5.0, seed=1, rdf_bin_width=0.01))` (the notebook's config).
- Execute the notebook in place: `jupyter nbconvert --to notebook --execute --inplace notebooks/jostle.ipynb`.

## Documentation upkeep

- `README.md` is a short, user-facing overview (banner, animation, quick start, a few settings, links). Keep details out of it.
- `docs/internals.md` holds the details. Keep it in sync when settings, outputs, constants or the package layout change.
- `docs/images/` holds the README figures, made from the notebook's run (2 nm box, 5 ps).

## User decisions (don't undo without asking)

- No config presets, no input config files, no CLI; settings are `Config` defaults + keyword overrides. Plotting is never configured in `Config`.
- Internal constants (`NEIGHBOUR_SKIN`, `RDF_CONFIDENCE`, `CUE_*`, `PHASE_SPACE_MAX_BONDS`, the `checks.py` thresholds) stay constants, not settings.
- `run()` makes no figures; figures are made one at a time from `Results` (no save-all).
- `run()` always prints the expected output size. No warnings for few molecules or for O2/H2 density. The default config must give no warnings (tested).
- The phase space is recorded every step over only the last `phase_space_window` ps, for at most 100 bonds per type: smooth but small.
- Old run folders (with `save_steps`) fail to load on purpose: no compatibility shims.
- Modelling approximations (no electrostatics, shifted-force LJ cutoff, isokinetic thermostat) are deliberate. The full-LJ energy is only reported (`lj_correction`: shift + tail), the forces stay shifted-force; no pressure. O2/H2 `MOLAR_VOLUMES` are liquid values at the boiling point on purpose.
- No clamp on the dihedral 1/sin(angle) terms: the singularity doesn't occur in practice.

## Working rules

- The notebook is committed **with its outputs** (about 2 MB); commit a re-executed notebook only when its results change.
- Only add tests for essential features (physics, the run pipeline, saving/loading, config validation), so the suite stays small. No tests for the notebook, VMD, plot styling, warning wording or other details.
- Keep new end-to-end checks on the shared `full_run`/`bare_run` in `test_simulation.py` rather than adding `run()` calls.
- Refactors keep results bit-identical unless a change is intended; compare against the previous commit.
- Follow the code conventions in docs/internals.md §11 (dense block comments, 80-char lines, vectorised NumPy).
