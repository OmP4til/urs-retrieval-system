# Unlimited-OCR Extraction Branch

Branch: `unlimited-ocr-extraction`

Replaces the Gemini API extraction path with Baidu's **Unlimited-OCR** model
running locally — no API key, no per-document cost, no data leaving the machine.

- Code: https://github.com/baidu/Unlimited-OCR
- Weights: https://huggingface.co/baidu/Unlimited-OCR (~3.34 GB, BF16)
- Paper: https://arxiv.org/abs/2606.23050

## What changed vs. the Gemini branch

Unlimited-OCR is a document **parsing** VLM built on DeepSeek-OCR. It converts
page images to structured markdown. It does not do free-form reasoning, so it
cannot itself return "extract every requirement and judge its priority" the way
Gemini did. The pipeline is therefore split into two explicit stages:

| Stage | Gemini branch | This branch |
|---|---|---|
| 1. Get text out of the document | `pdfplumber` / `python-docx` text layer | **Unlimited-OCR** parses page images into markdown (tables, headings, layout preserved) |
| 2. Turn text into requirement records | One Gemini prompt, JSON back | `RequirementStructurer` — deterministic rules over document structure and requirement language |

Stage 2 output is schema-identical to the Gemini output, so the database,
embeddings, and matching code are untouched:

```python
{
  'id', 'text', 'category', 'subcategory', 'confidence', 'source_context',
  'priority', 'technical_parameters', 'dependencies', 'compliance_standards',
  'notes', 'extraction_method'  # 'unlimited_ocr_holistic'
}
```

Trade-off worth knowing: stage 2 is now rule-based rather than an LLM's
judgement. It is faster, free, reproducible run-to-run, and inspectable — but it
will not infer *implicit* requirements from context the way Gemini could. Stage 1
is a clear upgrade: OCR reads scanned pages and complex tables that the text
layer cannot.

## New files

| File | Purpose |
|---|---|
| [utils/unlimited_ocr_processor.py](utils/unlimited_ocr_processor.py) | `UnlimitedOCRProcessor` (model loading, PDF/image parsing) and `RequirementStructurer` (stage 2) |
| [utils/processor_factory.py](utils/processor_factory.py) | `get_processor()` — returns the backend named by `EXTRACTION_BACKEND` |
| [standalone_holistic_extraction_ocr.py](standalone_holistic_extraction_ocr.py) | CLI runner, mirrors the Gemini standalone script |
| [test_unlimited_ocr_extraction.py](test_unlimited_ocr_extraction.py) | Tests stage 2 with no weights needed; `--with-model` runs the full pipeline |
| [requirements-unlimited-ocr.txt](requirements-unlimited-ocr.txt) | Extra deps pinned to Baidu's tested versions |

`config.py` gains `EXTRACTION_BACKEND` (`unlimited_ocr` by default) plus
`UNLIMITED_OCR_*` settings. The Gemini path is left intact — flip
`EXTRACTION_BACKEND=gemini` to get it back.

`app/main_gemini.py` now builds its processor through `get_processor()` instead
of constructing `GeminiProcessor` directly, so the Streamlit app follows
`EXTRACTION_BACKEND` too. It shows the active backend in a caption, only demands
`GEMINI_API_KEY` when the Gemini backend is selected, and routes PDFs/images
through the OCR model (DOCX keeps using its native text layer).

## Comment handling

The app's two comment-aware modes are supported on both backends —
`UnlimitedOCRProcessor` implements `extract_comments_and_responses` and
`extract_requirements_with_comments_holistically` with signatures identical to
the Gemini ones.

This costs nothing in accuracy: comment extraction and requirement pairing were
already LLM-free on the Gemini branch. Comments come from the DOCX comment parts
(`get_docx_comments_with_text_mapping`) and are paired by exact substring match
first, then `intfloat/e5-large-v2` similarity above 0.75. That logic is reused
verbatim.

Verified on `URS Coating Machine Rev 1 - GLATT comments 03092025.docx`:
67 requirements (67 unique, no duplicates), 63 of them paired with their GLATT
comments and authors; comment-only mode returns 109 commented segments.

## Usage

```bash
# Structuring stage only — no weights, no GPU. Good first check.
python test_unlimited_ocr_extraction.py

# Full pipeline, inspect without touching the database
python standalone_holistic_extraction_ocr.py "G_URS Tablet Coating Machine 1.pdf" \
    --no-db --dump-json reqs.json --parsed-output ./ocr_out

# Full pipeline into the urs_gemini database
python standalone_holistic_extraction_ocr.py "G_URS Tablet Coating Machine 1.pdf" \
    --comments "Initial analysis"
```

In application code:

```python
from utils.processor_factory import get_processor

processor = get_processor()                     # honours EXTRACTION_BACKEND
text = processor.parse_document(path)           # OCR backend only
reqs = processor.extract_requirements_holistically(text, filename)
```

## Hardware requirements — read before running `--with-model`

The weights are 3.34 GB in BF16. With activations and a 32k-token KV cache you
want **~6 GB+ of free VRAM**.

**This machine's GPU cannot run it.** The NVIDIA T1000 has 4 GB total (~1.9 GB
free) and is Turing (SM 7.5), which has no native bfloat16 and cannot use the
`fa3` attention backend the upstream SGLang recipe requires. `_select_device_and_dtype()`
detects this and falls back to CPU float32 automatically — correct results, but
expect **minutes per page** rather than seconds.

Practical options:

1. **CPU** — works out of the box, slow. Fine for one-off batch runs.
2. **A bigger GPU** (≥8 GB, Ampere or newer) — the intended path. Install the
   CUDA torch build from `requirements-unlimited-ocr.txt` first.
3. **vLLM or SGLang server** — best throughput for many documents; see the
   upstream README. Not usable on this machine's GPU.

The installed environment is also Python 3.13 with a CPU-only `torch 2.8.0`,
and `einops` / `addict` / `easydict` / `torchvision` / `psutil` are missing —
all needed by the model's `trust_remote_code` modules:

```bash
pip install -r requirements-unlimited-ocr.txt
```

## Tuning stage 2

The rules live in `RequirementStructurer` and are meant to be edited:

- `MODALS` — modal verbs mapped to priority and confidence
- `SPEC_PATTERNS` — non-modal signals that catch spec-table rows
- `CATEGORIES` — keyword lists per category
- `STANDARD_RE` / `PARAM_RE` — compliance standards and numeric parameters
- `MIN_LEN` / `MAX_LEN` — candidate length bounds

Current behaviour on `G_URS Tablet Coating Machine 1.pdf` (text layer, 40k chars):
139 requirements — 68 critical, 38 high, 28 medium, 5 low; 20 carry technical
parameters and 5 cite compliance standards.
