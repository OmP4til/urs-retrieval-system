#!/usr/bin/env python3
"""
Standalone Holistic Extraction Script for the Unlimited-OCR Branch

Mirrors standalone_holistic_extraction_gemini.py, but parses documents with
Baidu's Unlimited-OCR model running locally instead of calling the Gemini API.
Results are saved to the same urs_gemini PostgreSQL database, so downstream
search and matching are unchanged.

Usage:
    python standalone_holistic_extraction_ocr.py <file_path> [options]

Example:
    python standalone_holistic_extraction_ocr.py "G_URS Tablet Coating Machine 1.pdf" --comments "Initial analysis"
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.unlimited_ocr_processor import UnlimitedOCRProcessor
from utils.postgres_vectorstore_gemini import PostgresVectorStoreGemini
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def process_document_holistically(file_path: str,
                                  comments: str = None,
                                  matched_doc: str = None,
                                  model_name: str = "baidu/Unlimited-OCR",
                                  device: str = None,
                                  dpi: int = 300,
                                  image_mode: str = "base",
                                  parsed_output: str = None,
                                  dump_json: str = None,
                                  no_db: bool = False):
    """
    Process a document using Unlimited-OCR and store in the urs_gemini database.

    Args:
        file_path: Path to the document to process
        comments: Optional comments to add to the requirements
        matched_doc: Optional matched document name
        model_name: HuggingFace model id or local weights path
        device: Force 'cuda' or 'cpu'; auto-detected when None
        dpi: PDF rasterisation DPI
        image_mode: 'base' or 'gundam' (single-page only)
        parsed_output: Directory to keep the raw parsed markdown
        dump_json: Write the structured requirements to this JSON file
        no_db: Skip the database write (parse and inspect only)
    """

    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return False

    filename = os.path.basename(file_path)
    print(f"[OCR] Starting holistic extraction for: {filename}")

    try:
        # Step 1: Connect to database (unless inspecting only)
        vs_gemini = None
        if not no_db:
            print("[DB] Connecting to urs_gemini database...")
            vs_gemini = PostgresVectorStoreGemini()
            print("[DB] Connected to urs_gemini database")

        # Step 2: Initialise the OCR processor
        print(f"[OCR] Initializing Unlimited-OCR processor ({model_name})...")
        processor = UnlimitedOCRProcessor(
            model_name=model_name,
            device=device,
            dpi=dpi,
            image_mode=image_mode,
        )

        # Step 3: Parse the document with the OCR model
        print("[OCR] Parsing document (this downloads ~3.4 GB of weights on first run)...")
        full_text = processor.parse_document(file_path, output_path=parsed_output)

        if not full_text or len(full_text.strip()) < 100:
            print("[ERROR] Failed to parse meaningful text from document")
            print(f"        Parsed text length: {len(full_text) if full_text else 0}")
            return False

        print(f"[OCR] Parsed {len(full_text):,} characters from document")

        if parsed_output:
            os.makedirs(parsed_output, exist_ok=True)
            md_path = os.path.join(parsed_output, f"{Path(filename).stem}.parsed.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"[OCR] Raw parsed markdown written to {md_path}")

        # Step 4: Structure the requirements
        print("[OCR] Structuring requirements from parsed document...")
        requirements = processor.extract_requirements_holistically(
            full_document_text=full_text,
            document_name=filename,
        )

        if not requirements:
            print("[ERROR] No requirements extracted from the parsed document")
            return False

        print(f"[OK] Extracted {len(requirements)} requirements!")

        # Step 5: Summary
        print("\nAnalysis Summary:")
        categories = {}
        priorities = {}
        for req in requirements:
            cat = req.get('category', 'unknown')
            pri = req.get('priority', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            priorities[pri] = priorities.get(pri, 0) + 1

        print("\nBy Category:")
        for cat, count in sorted(categories.items(), key=lambda kv: -kv[1]):
            print(f"  - {cat}: {count}")

        print("\nBy Priority:")
        for pri in ('critical', 'high', 'medium', 'low'):
            if pri in priorities:
                print(f"  - {pri}: {priorities[pri]}")

        if dump_json:
            with open(dump_json, "w", encoding="utf-8") as f:
                json.dump(requirements, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] Structured requirements written to {dump_json}")

        # Step 6: Store in database
        if no_db:
            print("\n[SKIP] --no-db set, not writing to the database")
            print("\nSample Requirements (first 3):")
            for i, req in enumerate(requirements[:3]):
                print(f"\n{i + 1}. {req['text']}")
                print(f"   Category: {req.get('category')} | Priority: {req.get('priority')} "
                      f"| Confidence: {req.get('confidence')}")
            return True

        print("\n[DB] Storing requirements in urs_gemini database...")
        requirement_texts = [req['text'] for req in requirements]

        success = vs_gemini.add_requirements(
            requirements=requirement_texts,
            document_name=filename,
            comments=comments,
            matched_document_name=matched_doc,
        )

        if not success:
            print("[ERROR] Failed to store requirements in database")
            return False

        print(f"[OK] Successfully stored {len(requirement_texts)} requirements in urs_gemini database!")

        print("\nSample Requirements (first 3):")
        for i, req in enumerate(requirements[:3]):
            print(f"\n{i + 1}. {req['text']}")
            print(f"   Category: {req.get('category')} | Priority: {req.get('priority')} "
                  f"| Confidence: {req.get('confidence')}")

        if len(requirements) > 3:
            print(f"\n... and {len(requirements) - 3} more requirements")

        stats = vs_gemini.get_stats()
        print("\nDatabase Statistics:")
        print(f"  Total requirements: {stats.get('total_requirements', 0)}")
        print(f"  Documents processed: {len(stats.get('documents', {}))}")

        return True

    except Exception as e:
        print(f"[ERROR] Error during holistic analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Holistic requirement extraction using Baidu Unlimited-OCR for the urs_gemini database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python standalone_holistic_extraction_ocr.py "document.pdf"
    python standalone_holistic_extraction_ocr.py "document.pdf" --comments "Initial analysis"
    python standalone_holistic_extraction_ocr.py "document.pdf" --no-db --dump-json reqs.json
    python standalone_holistic_extraction_ocr.py "scan.png" --image-mode gundam
        """
    )

    parser.add_argument("file_path", help="Path to the document to process (PDF, image, or DOCX)")
    parser.add_argument("--comments", default=None,
                        help="Optional comments to add to the extracted requirements")
    parser.add_argument("--matched", default=None,
                        help="Optional matched document name for reference")
    parser.add_argument("--clear-first", action="store_true",
                        help="Clear database before processing (use with caution)")
    parser.add_argument("--model", default=os.getenv("UNLIMITED_OCR_MODEL", "baidu/Unlimited-OCR"),
                        help="HuggingFace model id or local path to the weights")
    parser.add_argument("--device", default=os.getenv("UNLIMITED_OCR_DEVICE") or None,
                        choices=("cuda", "cpu"), help="Force a device; auto-detected by default")
    parser.add_argument("--dpi", type=int, default=int(os.getenv("UNLIMITED_OCR_DPI", "300")),
                        help="PDF rasterisation DPI")
    parser.add_argument("--image-mode", default=os.getenv("UNLIMITED_OCR_IMAGE_MODE", "base"),
                        choices=("base", "gundam"),
                        help="Single-page parsing mode; multi-page always uses base")
    parser.add_argument("--parsed-output", default=None,
                        help="Directory to keep the raw parsed markdown")
    parser.add_argument("--dump-json", default=None,
                        help="Write the structured requirements to this JSON file")
    parser.add_argument("--no-db", action="store_true",
                        help="Parse and structure only; skip the database write")

    args = parser.parse_args()

    if args.clear_first:
        if args.no_db:
            print("[ERROR] --clear-first cannot be combined with --no-db")
            sys.exit(1)
        print("[DB] Clearing urs_gemini database...")
        vs_gemini = PostgresVectorStoreGemini()
        if vs_gemini.clear_database():
            print("[OK] Database cleared")
        else:
            print("[ERROR] Failed to clear database")
            return

    success = process_document_holistically(
        file_path=args.file_path,
        comments=args.comments,
        matched_doc=args.matched,
        model_name=args.model,
        device=args.device,
        dpi=args.dpi,
        image_mode=args.image_mode,
        parsed_output=args.parsed_output,
        dump_json=args.dump_json,
        no_db=args.no_db,
    )

    if success:
        print(f"\nSuccessfully processed {os.path.basename(args.file_path)}!")
        print("You can now use the Streamlit app to search and analyze the extracted requirements.")
    else:
        print(f"\nFailed to process {os.path.basename(args.file_path)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
