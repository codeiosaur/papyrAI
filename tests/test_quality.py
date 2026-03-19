#!/usr/bin/env python
"""
Quick quality testing script for wiki generation.

Usage:
  python test_quality.py output/vault/SubjectName/*.md

This script evaluates generated wiki pages against quality metrics.
Use it to benchmark improvements from prompt/parameter changes.
"""

import sys
from pathlib import Path

# Allow running via: python test_quality.py ...
if __package__ in (None, ""):
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from pdfwiki.quality_metrics import validate_wiki_page, format_quality_report, batch_quality_report


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    # Collect all markdown files from arguments
    page_files = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.is_file() and path.suffix == ".md":
            page_files.append(path)
        elif path.is_dir():
            page_files.extend(path.glob("*.md"))
        else:
            # Glob pattern
            page_files.extend(Path(".").glob(arg))
    
    if not page_files:
        print(f"No markdown files found in arguments: {sys.argv[1:]}")
        sys.exit(1)
    
    print(f"\n📊 Analyzing {len(page_files)} wiki pages...\n")
    
    # Extract valid wikilink names from file names
    # Assumption: each .md file is a valid wiki page name
    valid_names = [f.stem for f in page_files]
    
    pages_data = {}
    for file_path in page_files:
        content = file_path.read_text(encoding="utf-8")
        concept = file_path.stem
        pages_data[concept] = content
        
        # Print individual quality report
        quality = validate_wiki_page(content, concept, valid_names)
        print(format_quality_report(quality))
        print()
    
    # Print batch summary
    print(batch_quality_report(pages_data, valid_names))


if __name__ == "__main__":
    main()
