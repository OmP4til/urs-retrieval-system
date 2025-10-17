"""
Enhanced Comment Display - Shows Complete Comments Properly
Handles both section-level and requirement-level comments
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.smart_comment_extraction import extract_requirements_with_smart_comments


def display_with_complete_comments(requirements: list) -> str:
    """
    Display requirements with complete, properly formatted comments
    
    Logic:
    1. If a SECTION HEADER has a comment → Show it once at the start, applies to all reqs in that section
    2. If a REQUIREMENT has a comment → Show it only for that specific requirement
    3. Show FULL comment text, no truncation
    """
    output = []
    current_section = None
    section_comment_displayed = set()
    
    output.append("="*100)
    output.append(" URS REQUIREMENTS WITH COMMENTS")
    output.append("="*100)
    
    for i, req in enumerate(requirements, 1):
        # Extract main section number (e.g., "5.4" from "5.4.2.13")
        section_parts = req['section_number'].split('.') if req['section_number'] else []
        main_section = '.'.join(section_parts[:2]) if len(section_parts) >= 2 else req['section_number']
        
        # Check if we're entering a new section
        if main_section and main_section != current_section:
            current_section = main_section
            
            # Display section header with comment if exists and not yet shown
            if req['section_comment'] and main_section not in section_comment_displayed:
                output.append("\n" + "="*100)
                output.append(f"📂 SECTION {main_section}: {req['section_title'] or ''}")
                output.append("="*100)
                output.append(f"\n🔖 SECTION-LEVEL COMMENT (applies to ALL requirements in this section):")
                output.append(f"   Author: {req['section_comment_author']}")
                output.append(f"   Date: {req['section_comment_date'] or 'Not specified'}")
                output.append(f"\n   📝 {req['section_comment']}")
                output.append("\n" + "-"*100 + "\n")
                section_comment_displayed.add(main_section)
            elif main_section not in section_comment_displayed:
                # Just show a section divider
                output.append(f"\n{'─'*100}")
                output.append(f"📂 Section {main_section}")
                output.append(f"{'─'*100}\n")
                section_comment_displayed.add(main_section)
        
        # Display requirement number and text
        req_num = req['section_number'] if req['section_number'] else f"#{i}"
        output.append(f"\n[{req_num}] {req['requirement']}")
        
        # Display individual requirement comment if exists
        if req['comment']:
            output.append(f"\n   💬 REQUIREMENT COMMENT:")
            output.append(f"      Author: {req['comment_author']}")
            output.append(f"      Date: {req['comment_date'] or 'Not specified'}")
            output.append(f"\n      📝 {req['comment']}")
            output.append("")
    
    output.append("\n" + "="*100)
    output.append(" END OF DOCUMENT")
    output.append("="*100)
    
    return '\n'.join(output)


def generate_report(docx_file: str):
    """Generate complete report with proper comment display"""
    print(f"\n{'='*100}")
    print(f" PROCESSING: {os.path.basename(docx_file)}")
    print(f"{'='*100}\n")
    
    with open(docx_file, 'rb') as f:
        docx_bytes = f.read()
    
    requirements = extract_requirements_with_smart_comments(docx_bytes)
    
    # Statistics
    total_reqs = len(requirements)
    reqs_with_section_comments = len([r for r in requirements if r['section_comment']])
    reqs_with_individual_comments = len([r for r in requirements if r['comment']])
    reqs_with_both = len([r for r in requirements if r['comment'] and r['section_comment']])
    
    print(f"📊 STATISTICS:")
    print(f"   Total Requirements: {total_reqs}")
    print(f"   Requirements under commented sections: {reqs_with_section_comments}")
    print(f"   Requirements with individual comments: {reqs_with_individual_comments}")
    print(f"   Requirements with BOTH types of comments: {reqs_with_both}")
    
    # Show unique sections
    unique_sections = {}
    for req in requirements:
        if req['section_number']:
            parts = req['section_number'].split('.')
            main_sec = '.'.join(parts[:2]) if len(parts) >= 2 else req['section_number']
            if main_sec not in unique_sections:
                unique_sections[main_sec] = {
                    'has_comment': bool(req['section_comment']),
                    'num_reqs': 0
                }
            unique_sections[main_sec]['num_reqs'] += 1
    
    print(f"\n📂 SECTIONS FOUND: {len(unique_sections)}")
    for sec_num in sorted(unique_sections.keys(), key=lambda x: [int(n) if n.isdigit() else n for n in x.split('.')]):
        sec_info = unique_sections[sec_num]
        comment_indicator = "💬" if sec_info['has_comment'] else "  "
        print(f"   {comment_indicator} Section {sec_num}: {sec_info['num_reqs']} requirements")
    
    # Generate formatted output
    formatted_output = display_with_complete_comments(requirements)
    
    # Save to file
    output_file = f"COMPLETE_COMMENTS_{os.path.basename(docx_file).replace('.docx', '.txt')}"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(formatted_output)
    
    print(f"\n✅ Complete output saved to: {output_file}")
    
    # Show a sample
    print(f"\n{'='*100}")
    print(" SAMPLE OUTPUT (First 5 requirements with comments)")
    print(f"{'='*100}\n")
    
    sample_count = 0
    for req in requirements:
        if req['comment'] and sample_count < 5:
            print(f"[{req['section_number'] or 'N/A'}] {req['requirement'][:100]}...")
            print(f"   💬 {req['comment'][:150]}...")
            print()
            sample_count += 1
    
    return requirements


if __name__ == "__main__":
    files = [
        "URS Coating Machine Rev 1 - GLATT comments 03092025.docx",
        "Novugen_URS IGL (1).docx"
    ]
    
    all_reqs = []
    for file in files:
        if os.path.exists(file):
            reqs = generate_report(file)
            all_reqs.extend(reqs)
        else:
            print(f"⚠️ File not found: {file}\n")
    
    # Overall summary
    print(f"\n\n{'='*100}")
    print(" OVERALL SUMMARY - ALL FILES")
    print(f"{'='*100}")
    print(f"Total requirements: {len(all_reqs)}")
    print(f"With section-level comments: {len([r for r in all_reqs if r['section_comment']])}")
    print(f"With individual comments: {len([r for r in all_reqs if r['comment']])}")
    print(f"\n✨ All complete comments properly extracted and displayed!")
