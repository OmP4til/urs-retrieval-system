# Project conventions

Rules for where things live in this repository. Follow them when adding files.

## Where files go

| Location | What belongs there | What does not |
|---|---|---|
| **root** | Entry points you run directly, plus configuration: `config.py`, `requirements*.txt`, `README.md`, `.env.example`, and CLI scripts such as `standalone_holistic_extraction_ocr.py` | Anything you would not run or edit as configuration |
| **`app/`** | The Streamlit UI | Business logic that could be unit-tested — put that in `utils/` |
| **`utils/`** | Importable library code. The only package `app/` and the CLIs import from | Scripts meant to be executed directly |
| **`tests/`** | Automated verification. **All** `test_*.py` files | Manual/one-off tools |
| **`scripts/`** | One-off maintenance and debugging tools, run by hand: DB checks, migrations, diagnostics | Anything the app imports at runtime |
| **`docs/`** | Reference and historical documentation | The branch README, which stays at the root |
| **`data/`** | `master_database.xlsm`, and sample documents under `data/samples/` | Generated output — write that to a temp directory |

## Naming

- **`tests/` uses `test_*.py`.** Nothing outside `tests/` may be named
  `test_*.py`. A future `pytest` run must collect the real tests and nothing
  else — manual scripts named `test_*` would be collected and would fail or,
  worse, write to the database.
- **`scripts/` uses a verb prefix**: `check_*`, `fix_*`, `migrate_*`,
  `clear_*`, `diagnose_*`. Never `test_*`.

## Is it a test or an entry point?

A file that **writes to the database, or is part of the product's normal
operation**, is an entry point — it belongs at the root regardless of its name.
A file that only **reads and asserts** is a test and belongs in `tests/`.

`standalone_holistic_extraction_ocr.py` writes extracted requirements to
Postgres, so it is an entry point despite being run manually.

## Paths

**Never hardcode a path to a data file**, and never build one from the current
working directory. Resolve through `config.py`, which anchors everything to
`PROJECT_ROOT` (derived from `config.py`'s own location):

```python
from config import MASTER_DB_PATH, SAMPLES_DIR, DATA_DIR
```

These are `pathlib.Path` objects and are overridable via the `MASTER_DB_PATH`
and `URS_DATA_DIR` environment variables. This is what lets the app and tests
run from any working directory.

A script that needs the project root on `sys.path` should derive it, not
hardcode it:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

## Secrets

No credentials in the repository — not in code, not in docs, not as a default
value. Read them from the environment and fail with a clear message when
missing, as `get_db_password()` does. `.env` is gitignored; document new keys in
`.env.example` instead.

## Branches

`unlimited-ocr-extraction` and `gemini-preprocessing` are kept independent, with
no shared backend switch. Do not add Gemini code to the OCR branch or vice
versa. They share the `urs_gemini` database schema on purpose, so the two
extraction approaches can be compared on the same documents.
