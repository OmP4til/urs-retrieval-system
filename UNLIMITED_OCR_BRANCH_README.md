# Unlimited-OCR Extraction Branch

Branch: `unlimited-ocr-extraction`

Runs Baidu's **Unlimited-OCR** model locally for requirement extraction — no API
key, no per-document cost, no data leaving the machine.

- Code: https://github.com/baidu/Unlimited-OCR
- Weights: https://huggingface.co/baidu/Unlimited-OCR (~3.34 GB, BF16)
- Paper: https://arxiv.org/abs/2606.23050

## Branch separation

This branch and the Gemini branch are kept deliberately independent so their
logic never mixes:

| Branch | Extraction | Entry point |
|---|---|---|
| `gemini-preprocessing` | Gemini 3.6 Flash (API) | `app/main_gemini.py`, `standalone_holistic_extraction_gemini.py` |
| `unlimited-ocr-extraction` | Unlimited-OCR (local) | `app/main_ocr.py`, `standalone_holistic_extraction_ocr.py` |

There is **no runtime backend switch**. This branch contains no Gemini code at
all — `utils/gemini_processor.py`, the Gemini standalone script, the Gemini docs,
and the dead Gemini helpers in `utils/extractors.py` are all removed here. To run
Gemini, check out `gemini-preprocessing`.

Both branches still write to the same `urs_gemini` PostgreSQL database through
`PostgresVectorStoreGemini`. That naming is historical and deliberately left
alone — a shared schema is what lets you compare the two extraction approaches on
the same documents.

## How extraction works

Unlimited-OCR is a document **parsing** VLM built on DeepSeek-OCR. It converts
page images to structured markdown; it does not do free-form reasoning. So
extraction is two explicit stages:

| Stage | What runs |
|---|---|
| 1. Get text out of the document | **Unlimited-OCR** parses page images into markdown, preserving tables, headings, and layout |
| 2. Text → requirement records | `RequirementStructurer` — deterministic rules over document structure and requirement language |

Stage 2 emits the same records the Gemini branch produced, so the database,
embeddings, and matching code are untouched:

```python
{
  'id', 'text', 'category', 'subcategory', 'confidence', 'source_context',
  'priority', 'technical_parameters', 'dependencies', 'compliance_standards',
  'notes', 'extraction_method'  # 'unlimited_ocr_holistic'
}
```

Trade-off worth knowing: stage 2 is rule-based rather than an LLM's judgement.
It is faster, free, reproducible run-to-run, and inspectable — but it will not
infer *implicit* requirements from context the way Gemini could. Stage 1 is a
clear upgrade: OCR reads scanned pages and complex tables that a text layer
cannot.

## Files

| File | Purpose |
|---|---|
| [utils/unlimited_ocr_processor.py](utils/unlimited_ocr_processor.py) | `UnlimitedOCRProcessor` (weights, PDF/image parsing, DOCX comment pairing) and `RequirementStructurer` |
| [app/main_ocr.py](app/main_ocr.py) | Streamlit app (renamed from `main_gemini.py`) |
| [standalone_holistic_extraction_ocr.py](standalone_holistic_extraction_ocr.py) | CLI runner |
| [test_unlimited_ocr_extraction.py](test_unlimited_ocr_extraction.py) | Tests stage 2 with no weights needed; `--with-model` runs the full pipeline |
| [requirements-unlimited-ocr.txt](requirements-unlimited-ocr.txt) | Extra deps pinned to Baidu's tested versions |
| [docs/UNLIMITED_OCR_MATCHING.md](docs/UNLIMITED_OCR_MATCHING.md) | Exactly how extraction and matching work, with the real thresholds |

Everything the app does not import lives in two folders, so the project root
only holds what you actually run:

| Folder | Contents |
|---|---|
| [docs/](docs/) | Reference and historical documentation |
| [scripts/](scripts/) | One-off maintenance and debugging scripts (DB checks, embedding migrations, diagnostics). Not imported by the app; run them directly. |

`config.py` holds the `UNLIMITED_OCR_*` settings and no longer defines any
`GEMINI_*` values.

## Comment handling

Both comment modes in the app work here. `UnlimitedOCRProcessor` implements
`extract_comments_and_responses` and
`extract_requirements_with_comments_holistically`.

This costs nothing in accuracy: comment extraction and requirement pairing were
already LLM-free on the Gemini branch. Comments come from the DOCX comment parts
(`get_docx_comments_with_text_mapping`) and are paired by exact substring match
first, then `intfloat/e5-large-v2` similarity above 0.75. That logic is reused.

## Usage

```bash
# Structuring stage only — no weights, no GPU. Good first check.
python test_unlimited_ocr_extraction.py

# Full pipeline, inspect without touching the database
python standalone_holistic_extraction_ocr.py "data/samples/G_URS Tablet Coating Machine 1.pdf" \
    --no-db --dump-json reqs.json --parsed-output ./ocr_out

# Full pipeline into the urs_gemini database
python standalone_holistic_extraction_ocr.py "data/samples/G_URS Tablet Coating Machine 1.pdf" \
    --comments "Initial analysis"

# Streamlit app
streamlit run app/main_ocr.py
```

## Verified

Stage 2 (structuring), on documents with a text layer:

- `URS for Tablet Coating Machine 1.doc` — 143 requirements (legacy .doc,
  converted via Word COM)
- `URS Coating Machine Rev 1 - GLATT comments 03092025.docx` — 67 requirements
  (all unique), 63 paired with their GLATT comments and authors; comment-only
  mode returns 109 commented segments

Stage 1 (the OCR model itself), on a genuinely scanned PDF with **zero**
extractable text:

- `9 URS-Auto Tablet Coating 75kg 18.08.2025.pdf`, 6 pages, at the app's
  defaults (300 DPI, base, `infer_multi`): **925 s ≈ 15.4 min**, 20,416
  characters, 54 requirements. Model load 13 s, ~7.6 GB RAM.
- Single page at 200 DPI: 147 s. So roughly **2.5 min/page** on CPU.

## Hardware requirements — read before running OCR

The weights are 3.34 GB in BF16. With activations and a 32k-token KV cache you
want **~6 GB+ of free VRAM** to run on GPU.

**This machine's GPU cannot hold it.** The NVIDIA T1000 has 4 GB total (~2 GB
free) and is Turing (SM 7.5), which has no native bfloat16 and cannot use the
`fa3` attention backend the upstream SGLang recipe requires.
`_select_device_and_dtype()` detects this and falls back to **CPU in bfloat16**
— correct results at roughly 2.5 minutes per page.

Two things had to be worked around for CPU to run at all, both in
`utils/unlimited_ocr_processor.py`:

- `modeling_unlimitedocr.py` calls `.cuda()` on its input tensors in 14 places
  regardless of where the model was loaded. `_install_cpu_cuda_shim()`
  redirects those to no-ops when CUDA is genuinely unavailable.
- CPU must use bfloat16, not float32: the model casts image tensors to
  bfloat16 unconditionally, so float32 weights raise a dtype mismatch.

Practical options:

1. **CPU** — works out of the box, ~2.5 min/page. Fine for occasional documents.
2. **A bigger GPU** (≥8 GB, Ampere or newer) — seconds per page.
3. **vLLM or SGLang server** — best throughput in bulk; see the upstream README.
   Not usable on this machine's GPU.

Install the OCR dependencies with:

```bash
pip install -r requirements-unlimited-ocr.txt
```

Note that pin set installs `torch` **CPU-only** on Windows. Nothing uses the
GPU until a CUDA build is installed.

## Tuning stage 2

The rules live in `RequirementStructurer` and are meant to be edited:

- `MODALS` — modal verbs mapped to priority and confidence
- `SPEC_PATTERNS` — non-modal signals that catch spec-table rows
- `CATEGORIES` — keyword lists per category
- `STANDARD_RE` / `PARAM_RE` — compliance standards and numeric parameters
- `MIN_LEN` / `MAX_LEN` — candidate length bounds
- `COMMENT_MATCH_THRESHOLD` — DOCX comment pairing threshold (0.75)
