"""
Test Smart Comment Extraction
- Shows section-level comments that apply to multiple requirements
- Shows individual requirement comments
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.smart_comment_extraction import extract_requirements_with_smart_comments, format_smart_output


def test_smart_extraction(docx_file):
    """Test smart comment extraction on a DOCX file"""
    print(f"\n{'='*80}")
    print(f"Testing Smart Comment Extraction on: {docx_file}")
    print(f"{'='*80}\n")
    
    with open(docx_file, 'rb') as f:
        docx_bytes = f.read()
    
    requirements = extract_requirements_with_smart_comments(docx_bytes)
    
    print(f"Total requirements extracted: {len(requirements)}")
    
    # Count section comments vs individual comments
    section_comments_count = len([r for r in requirements if r['section_comment']])
    individual_comments_count = len([r for r in requirements if r['comment']])
    
    print(f"Requirements under commented sections: {section_comments_count}")
    print(f"Requirements with individual comments: {individual_comments_count}")
    
    # Show unique sections with comments
    unique_section_comments = {}
    for req in requirements:
        if req['section_comment'] and req['section_title']:
            section_key = f"{req['section_number'][:3] if req['section_number'] else 'N/A'}"
            if section_key not in unique_section_comments:
                unique_section_comments[section_key] = {
                    'title': req['section_title'],
                    'comment': req['section_comment'],
                    'author': req['section_comment_author']
                }
    
    print(f"\n📂 Sections with Comments: {len(unique_section_comments)}")
    for sec_num, sec_info in sorted(unique_section_comments.items()):
        print(f"   {sec_num}: {sec_info['title']}")
        print(f"      └─ Comment by {sec_info['author']}: {sec_info['comment'][:100]}...")
    
    # Show some examples
    print(f"\n{'='*80}")
    print("SAMPLE OUTPUT (First 10 requirements)")
    print(f"{'='*80}\n")
    
    output = format_smart_output(requirements[:10])
    print(output)
    
    # Save full output
    output_file = f"smart_extracted_{os.path.basename(docx_file).replace('.docx', '')}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(format_smart_output(requirements))
    
    print(f"\n✅ Full output saved to: {output_file}")
    
    return requirements


if __name__ == "__main__":
    # Test on both files
    files = [
        "URS Coating Machine Rev 1 - GLATT comments 03092025.docx",
        "Novugen_URS IGL (1).docx"
    ]
    
    all_requirements = []
    
    for file in files:
        if os.path.exists(file):
            reqs = test_smart_extraction(file)
            all_requirements.extend(reqs)
        else:
            print(f"⚠️ File not found: {file}")
    
    # Summary
    print(f"\n\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Total requirements across all files: {len(all_requirements)}")
    print(f"Requirements with section-level comments: {len([r for r in all_requirements if r['section_comment']])}")
    print(f"Requirements with individual comments: {len([r for r in all_requirements if r['comment']])}")
    print(f"Requirements with BOTH types of comments: {len([r for r in all_requirements if r['comment'] and r['section_comment']])}")
