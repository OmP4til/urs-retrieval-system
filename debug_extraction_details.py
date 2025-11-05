"""
Debug script to analyze extraction in detail.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from utils.extractors import extract_structured_content
from utils.preprocess import enhanced_rule_based_requirements

def analyze_detailed(file_path: str):
    """Detailed analysis of extraction."""
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: {file_path}")
    print(f"{'='*80}\n")
    
    with open(file_path, 'rb') as f:
        class FileWrapper:
            def __init__(self, file_obj, name):
                self.file = file_obj
                self.name = name
            
            def read(self):
                return self.file.read()
            
            def seek(self, pos):
                return self.file.seek(pos)
        
        wrapper = FileWrapper(f, os.path.basename(file_path))
        
        pages = extract_structured_content(wrapper)
        
        print(f"Total pages/chunks: {len(pages)}\n")
        
        total_reqs = 0
        table_row_pages = 0
        
        for idx, page in enumerate(pages, 1):
            content = page.get("content", "")
            content_type = page.get("content_type", "")
            section = page.get("section", "Unknown")
            
            if not content.strip():
                continue
            
            reqs = enhanced_rule_based_requirements(content)
            
            if content_type == "docx-table-row":
                table_row_pages += 1
            
            total_reqs += len(reqs)
            
            if len(reqs) > 0 and idx <= 20:  # Show first 20 for analysis
                print(f"Page {idx} ({content_type}) - Section: {section}")
                print(f"  Requirements found: {len(reqs)}")
                print(f"  Content preview: {content[:150]}...")
                if reqs:
                    print(f"  First req: {reqs[0][:100]}...")
                print()
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total pages: {len(pages)}")
        print(f"Table row pages: {table_row_pages}")
        print(f"Total requirements extracted: {total_reqs}")
        print(f"Average requirements per page: {total_reqs / len(pages):.2f}")

if __name__ == "__main__":
    analyze_detailed("URS Coating Machine Rev 1 - GLATT comments 03092025.docx")
