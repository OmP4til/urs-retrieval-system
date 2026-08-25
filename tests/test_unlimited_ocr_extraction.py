"""
Test the Unlimited-OCR extraction backend.

Two modes:

  Structuring only (default) - no model weights needed. Uses the PDF's existing
  text layer to exercise the requirement-structuring stage, so you can validate
  the rules without a GPU or a 3.4 GB download:

      python tests/test_unlimited_ocr_extraction.py

  Full pipeline - downloads and runs the Unlimited-OCR weights, parsing page
  images with the VLM before structuring:

      python tests/test_unlimited_ocr_extraction.py --with-model
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

# Resolve the project root from this file so the test runs from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.unlimited_ocr_processor import UnlimitedOCRProcessor, RequirementStructurer

from config import SAMPLES_DIR

DEFAULT_PDF = str(SAMPLES_DIR / "G_URS Tablet Coating Machine 1.pdf")


def report(requirements, source_label):
    print(f"\n{'=' * 80}")
    print(f"Extracted {len(requirements)} requirements via {source_label}")
    print(f"{'=' * 80}")

    if not requirements:
        print("No requirements found.")
        return

    print("\nBy category:")
    for cat, n in Counter(r['category'] for r in requirements).most_common():
        print(f"  {cat:<16} {n}")

    print("\nBy priority:")
    counts = Counter(r['priority'] for r in requirements)
    for pri in ('critical', 'high', 'medium', 'low'):
        if pri in counts:
            print(f"  {pri:<16} {counts[pri]}")

    with_params = sum(1 for r in requirements if r['technical_parameters'])
    with_stds = sum(1 for r in requirements if r['compliance_standards'])
    print(f"\nWith technical parameters: {with_params}")
    print(f"With compliance standards: {with_stds}")

    print("\nFirst 5 requirements:")
    for r in requirements[:5]:
        print(f"\n  {r['id']} [{r['category']}/{r['priority']}, conf {r['confidence']}]")
        print(f"    section: {r['source_context'][:70]}")
        print(f"    text:    {r['text'][:150]}")
        if r['technical_parameters']:
            print(f"    params:  {r['technical_parameters']}")
        if r['compliance_standards']:
            print(f"    stds:    {r['compliance_standards']}")

    # Schema check: must match what the Gemini path produced.
    expected = {'id', 'text', 'category', 'subcategory', 'confidence', 'source_context',
                'priority', 'technical_parameters', 'dependencies', 'compliance_standards',
                'notes', 'extraction_method'}
    missing = expected - set(requirements[0])
    if missing:
        print(f"\n[FAIL] Requirement schema is missing keys: {sorted(missing)}")
        return False
    print("\n[OK] Requirement schema matches the Gemini-compatible contract")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Document to test against")
    parser.add_argument("--with-model", action="store_true",
                        help="Run the real Unlimited-OCR model (downloads ~3.4 GB of weights)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if args.with_model:
        print(f"Parsing {args.pdf} with Unlimited-OCR...")
        processor = UnlimitedOCRProcessor(device=args.device, dpi=args.dpi)
        text = processor.parse_document(args.pdf)
        print(f"Parsed {len(text):,} characters of markdown")
        reqs = processor.extract_requirements_holistically(text, args.pdf)
        ok = report(reqs, "Unlimited-OCR (full pipeline)")
    else:
        from utils.extractors import extract_text_from_pdf
        print(f"Reading text layer of {args.pdf} (structuring stage only)...")
        text = extract_text_from_pdf(args.pdf)
        print(f"Read {len(text):,} characters")
        reqs = RequirementStructurer().structure(text, args.pdf)
        ok = report(reqs, "RequirementStructurer (text layer, no OCR model)")

    print("\nTest complete!")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
