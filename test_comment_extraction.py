"""
Test the improved comment extraction on actual URS files
Tests both: URS Coating Machine and Novugen_URS IGL
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.improved_comment_extraction import extract_requirements_with_comments, format_requirements_output

def test_file(docx_file: str):
    """Test extraction on a single file"""
    if not os.path.exists(docx_file):
        print(f"❌ File not found: {docx_file}")
        return None
    
    print("\n" + "="*80)
    print(f"TESTING: {os.path.basename(docx_file)}")
    print("="*80)
    
    # Read file
    with open(docx_file, 'rb') as f:
        docx_bytes = f.read()
    
    # Extract
    print("🔍 Extracting requirements with comment range mapping...")
    requirements = extract_requirements_with_comments(docx_bytes)
    
    # Statistics
    print(f"\n✅ Extraction complete!")
    print(f"   Total requirements: {len(requirements)}")
    
    with_comments = [r for r in requirements if r['comment']]
    print(f"   Requirements with comments: {len(with_comments)}")
    
    if with_comments:
        print(f"\n📝 Comment authors:")
        authors = set(r['comment_author'] for r in with_comments if r['comment_author'])
        for author in sorted(authors):
            count = sum(1 for r in with_comments if r['comment_author'] == author)
            print(f"      - {author}: {count} comments")
    
    # Show sample requirements with comments
    print("\n" + "-"*80)
    print("SAMPLE REQUIREMENTS WITH COMMENTS (First 5)")
    print("-"*80)
    
    sample_count = 0
    for req in requirements:
        if req['comment']:
            print(f"\n{sample_count + 1}. [{req['section_number'] or 'No section'}]")
            print(f"   Requirement: {req['requirement'][:150]}{'...' if len(req['requirement']) > 150 else ''}")
            print(f"   💬 Comment ({req['comment_author']}): {req['comment'][:150]}{'...' if len(req['comment']) > 150 else ''}")
            
            sample_count += 1
            if sample_count >= 5:
                break
    
    if sample_count == 0:
        print("\n⚠️ No comments found in document")
        print("   Showing first 3 requirements without comments:")
        for i, req in enumerate(requirements[:3], 1):
            print(f"\n{i}. [{req['section_number'] or 'No section'}]")
            print(f"   {req['requirement'][:200]}{'...' if len(req['requirement']) > 200 else ''}")
    
    # Export to file
    base_name = os.path.splitext(os.path.basename(docx_file))[0]
    output_file = f"extracted_{base_name}_with_comments.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"REQUIREMENTS EXTRACTED FROM: {docx_file}\n")
        f.write(f"Total: {len(requirements)} requirements\n")
        f.write(f"With comments: {len(with_comments)} requirements\n\n")
        f.write(format_requirements_output(requirements))
    
    print(f"\n💾 Full output saved to: {output_file}")
    
    return {
        'file': docx_file,
        'total': len(requirements),
        'with_comments': len(with_comments),
        'requirements': requirements
    }


def main():
    """Test both URS files"""
    print("="*80)
    print("TESTING IMPROVED COMMENT EXTRACTION ON ACTUAL URS FILES")
    print("="*80)
    print(f"\nCurrent directory: {os.getcwd()}\n")
    
    # Test files - these should be in the current directory
    test_files = [
        "URS Coating Machine Rev 1 - GLATT comments 03092025.docx",
        "Novugen_URS IGL (1).docx"
    ]
    
    results = []
    
    for docx_file in test_files:
        result = test_file(docx_file)
        if result:
            results.append(result)
    
    # Overall summary
    if results:
        print("\n" + "="*80)
        print("OVERALL SUMMARY")
        print("="*80)
        
        for result in results:
            file_name = os.path.basename(result['file'])
            print(f"\n📄 {file_name}")
            print(f"   Total requirements: {result['total']}")
            if result['total'] > 0:
                print(f"   With comments: {result['with_comments']} ({result['with_comments']/result['total']*100:.1f}%)")
            else:
                print(f"   With comments: 0")
        
        total_reqs = sum(r['total'] for r in results)
        total_comments = sum(r['with_comments'] for r in results)
        
        print(f"\n🎯 GRAND TOTAL:")
        print(f"   Files processed: {len(results)}")
        print(f"   Total requirements: {total_reqs}")
        print(f"   Requirements with comments: {total_comments}")
        if total_reqs > 0:
            print(f"   Comment coverage: {total_comments/total_reqs*100:.1f}%")
        else:
            print(f"   Comment coverage: 0%")
        
        print("\n✅ Extraction test complete!")
        print(f"📁 Output files created in: {os.getcwd()}")
    else:
        print("\n❌ No files were successfully processed")
        print(f"\nExpected files in: {os.getcwd()}")
        for f in test_files:
            print(f"   - {f}")


if __name__ == "__main__":
    main()
