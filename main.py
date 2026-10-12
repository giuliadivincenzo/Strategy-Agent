"""
CLI entry point.

Usage:
    python main.py path/to/document.pdf
    python main.py path/to/document.docx --model google/gemma-4-26b-a4b-it:free
    python main.py path/to/document.pdf --out outputs/result.json
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent import AgentError, analyze_document
from parser import UnsupportedFileType, extract_text


def main():
    ap = argparse.ArgumentParser(description="Strategy Document Analysis Agent")
    ap.add_argument("input_file", help="Path to a .pdf or .docx strategy document")
    ap.add_argument("--model", default=None, help="Override the OpenRouter model ID")
    ap.add_argument("--out", default=None, help="Path to write the JSON output (default: outputs/<name>.json)")
    ap.add_argument("--verbose", action="store_true", help="Print agent retry log")
    args = ap.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/3] Extracting text and images from {input_path.name}...")
    try:
        text, warnings, images = extract_text(str(input_path))
    except UnsupportedFileType as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if warnings:
        print(f"  Extraction warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")

    if images:
        print(f"  Extracted {len(images)} image(s) for visual analysis: "
              f"{', '.join(img['location'] for img in images)}")

    if not text.strip() and not images:
        print("Error: no text or images could be extracted from this document.", file=sys.stderr)
        sys.exit(1)

    print(f"[2/3] Sending to model for analysis (this may take a moment on free-tier models)...")
    try:
        result, log = analyze_document(text, warnings, images=images, model=args.model)
    except AgentError as e:
        print(f"Error: agent failed to produce valid output.\n{e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print("\n--- Agent log ---")
        for line in log:
            print(line)
        print("--- end log ---\n")

    out_path = Path(args.out) if args.out else Path("outputs") / f"{input_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.model_dump(), indent=2))

    print(f"[3/3] Done. Structured analysis written to {out_path}")


if __name__ == "__main__":
    main()
