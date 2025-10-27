# Repository Guidelines

## LANGUAGE SETTINGS
无论什么时候回答 都需要使用简体中文回答  思维thinking 链式也需要使用中文


## Project Structure & Module Organization
- `train_py/` contains the multi-task Hi-C pipeline (config, data, model, trainer). Extend functionality inside these modules instead of creating standalone scripts.
- `data_generate/` and `data_generated/` keep intermediate assets, caches, and manifests. Respect existing names and keep heavy binaries out of version control.
- `data_process/` hosts preprocessing tools such as `merge_loops_soft.py`; park additional utilities here using descriptive filenames (e.g., `normalize_tad_labels.py`).
- Stage experiment outputs under `outputs/` using timestamped subfolders like `outputs/2024-05-09_loop/` to separate runs.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` prepares an isolated development environment.
- `pip install torch numpy pandas cooler matplotlib` restores the core runtime dependencies used across training and visualization scripts.
- `python -m train_py.train --config train_config.json` launches training with overrides layered onto defaults from `train_py/config.py`.
- `python -m data_process.merge_loops_soft --help` lists preprocessing options before merging loop calls from multiple tools.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; use snake_case for functions and variables, PascalCase for classes, and align configuration keys with their JSON counterparts.
- Preserve existing type hints and dataclasses. Introduce new dataclasses when growing configuration surfaces instead of loosely typed dictionaries.
- Format modules with `black train_py` and clean imports via `isort train_py` prior to review.

## Testing Guidelines
- Use `pytest` for unit and integration coverage, placing suites under `tests/train_py/` to mirror module layout (e.g., `tests/train_py/test_trainer.py`).
- Prefer lightweight synthetic tensors stored in `data_generated/cache/test_samples.pt` (create if needed) to keep trainer and data-loader tests fast.
- Target ≥80% coverage on new modules and annotate unavoidable skips directly in the test file.

## Commit & Pull Request Guidelines
- Write imperative, scope-first commit titles such as `train: stabilize gradient clipping`.
- Reference related tasks or experiment IDs in the body and attach key metrics or plots pulled from `outputs/` when they influence conclusions.
- Pull requests should summarize modeling or data updates, list touched config fields, and include sanity checks (loss curves, qualitative plots, or accuracy deltas) whenever behavior shifts.

## Data & Configuration Notes
- Default configs reference `/storu/zkyang/AAA_MIL/...` paths. Adjust overrides carefully and document any environment-specific differences in the PR description.
- Store large genomics assets under `data/`, `data_generated/`, or `stripenn_out/`, and prefer repository-relative paths when sharing notebooks or scripts.
