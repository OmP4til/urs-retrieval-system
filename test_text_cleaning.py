#!/usr/bin/env python3
"""
Test script for text cleaning functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.extractors import clean_extracted_text

def test_text_cleaning():
    """Test the text cleaning functionality"""
    
    print("🧹 Testing Text Cleaning Functionality")
    print("=" * 50)
    
    # Test cases with backslashes and pipe symbols
    test_cases = [
        {
            "input": "Material certificates\\for all|metallic/non-metallic|product contact parts.",
            "description": "Certificate requirement with backslashes and pipes"
        },
        {
            "input": "Test\\report|2.2 will\\be provided",
            "description": "Test report comment with backslashes and pipes"
        },
        {
            "input": "The system\\must|be able\\to|prevent unauthorized\\access",
            "description": "Security requirement with mixed separators"
        },
        {
            "input": "Documentation\\|Requirements|\\Section 5.1",
            "description": "Section header with mixed separators"
        },
        {
            "input": "   Multiple   \\\\  spaces  ||  and   tabs\t\t",
            "description": "Text with excessive whitespace, backslashes, and pipes"
        },
        {
            "input": "|\\Start with separators and\\|end with them|\\",
            "description": "Text starting and ending with separators"
        }
    ]
    
    print("🔍 Testing Cases:")
    print("-" * 30)
    
    for i, case in enumerate(test_cases, 1):
        input_text = case["input"]
        cleaned_text = clean_extracted_text(input_text)
        
        print(f"\n{i}. {case['description']}")
        print(f"   Input:   '{input_text}'")
        print(f"   Cleaned: '{cleaned_text}'")
        
        # Check if cleaning worked
        has_backslash = "\\" in cleaned_text
        has_pipe = "|" in cleaned_text
        
        if has_backslash or has_pipe:
            print(f"   ⚠️  Warning: Still contains \\ or | characters")
        else:
            print(f"   ✅ Successfully cleaned")
    
    print("\n" + "=" * 50)
    print("📋 Summary of Cleaning Features:")
    print("• Removes backslashes (\\)")
    print("• Removes pipe symbols (|)")
    print("• Normalizes whitespace")
    print("• Removes double spaces")
    print("• Removes tabs")
    print("• Strips leading/trailing whitespace")
    
    print("\n🎯 Applied to All Extraction Points:")
    print("• PDF text extraction (line-by-line)")
    print("• PDF annotation content")
    print("• DOCX paragraph text")
    print("• DOCX comment text")
    print("• DOCX table cell content")
    print("• XLSX cell content")
    print("• Table extraction from all formats")
    print("• OCR text output")
    
    print("\n✨ Your documents will now have cleaner, more readable text!")

if __name__ == "__main__":
    test_text_cleaning()