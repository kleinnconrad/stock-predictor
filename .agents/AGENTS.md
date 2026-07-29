# Agent Behavioral Rules for Stock Predictor

## Fundamental Data Cache Maintenance
Whenever you start working on this repository, you MUST proactively check if the fundamental data cache is outdated.
1. Check the current date. The fundamental data cache maintenance runs every month on the 15th (or the first time you start Antigravity IDE after this date).
2. Check the last modified date of the files in `data/raw/fundamentals/`. 
3. If the current date is the 15th of the month or later, and the files were last updated before the 15th of the current month, you MUST automatically run the `python scripts/update_fundamentals.py` script to refresh the fundamentals cache.
4. Inform the user that you are running the monthly fundamentals update in the background.

## 1. Directory Structure
- The `src` directory is strictly reserved for the core logic of the application. 
- Do not place auxiliary code in `src`. Place supporting files in meaningfully named directories (e.g., `utilities`, `scripts`, `data_generation`, `sql`).
- Maintain a separate `tests` directory for unit and integration tests to keep the core codebase clean.
- The `README.md` must contain the folder structure of the repository, including information on what to find in each folder. Every time something changes with the folder structure, the folder structure in the `README.md` must be updated.

## 2. Dependencies
- Always maintain a `requirements.txt` file at the root of the project.
- Any time a new Python dependency is introduced, it must be explicitly added to `requirements.txt` with appropriate version pinning to ensure reproducible builds.

## 3. Environment Execution
- Execute local scripts using the Conda `base` environment to ensure consistency across local development setups.
- Provide clear instructions in the `README.md` if the Conda environment requires updating due to new dependencies.

## 4. Code Quality & Formatting
- Avoid hardcoding paths without environment checks; use relative paths or standard library path resolution (e.g., `pathlib`).
- **Configuration & Parameters:** User-defined parameters and thresholds must *never* be hardcoded in Python scripts. They must always be defined centrally in `config/settings.yaml`, and scripts must dynamically load and use the parameters from this file.
- Use the standard Python `logging` module instead of `print()` statements for proper log leveling, observability, and debugging.
- Implement Python type hinting (PEP 484) across all function signatures and class definitions to improve readability and catch type errors early.
- Follow PEP 8 style guidelines for Python code.

## 5. Language & Documentation Style
- **Tone:** Maintain a sober, objective, and professional tone across all documentation (including `README.md`).
- **Vocabulary:** Use only as many adjectives as strictly required. Do not use superlatives (e.g., "fastest", "best", "most") or filler words.
- **Emojis:** Do not use emojis anywhere in the codebase, documentation, or user interfaces.
- **Language:** Everything in the project must be written in English (documentation, markdown files, code comments, variable names, etc.).
- **Table of Contents:** Every markdown file containing more than one paragraph must include a Table of Contents.
- **Docstrings:** All modules, classes, and public functions must include clear, structured docstrings (e.g., Google-style) detailing their purpose, arguments, and return types.

## 6. Linting & Validation
- Code must pass standard Python linting and formatting checks (e.g., `ruff`, `flake8`, or `black`) before being finalized.
- Always verify that configuration files (e.g., `.yml` or `.json` files) remain structurally compliant and correctly reflect any changes made to the codebase (such as script renames or moving files).

## 7. Security Best Practices
- Never hardcode secrets, passwords, or API keys in the repository. Always rely on environment variables or a `.env` file (ensure `.env` is included in `.gitignore`).

## 8. Error Handling
- Use specific exception handling (`try...except` blocks) instead of catching generic `Exception`s where possible, and provide informative error messages.

## 9. Version Control & Committing
- You MUST NEVER run `git commit` or push changes to the repository by yourself unless the user explicitly asks you to do so. Prepare the changes locally and inform the user so they can review and commit them manually.
