"""
Improved DOCX Comment Extraction with Proper Range Mapping
Uses commentRangeStart/commentRangeEnd to accurately map comments to requirements
"""
import zipfile
import xml.etree.ElementTree as ET
import re
from typing import Dict, List, Tuple
import io


def extract_requirements_with_comments(docx_bytes: bytes) -> List[Dict]:
    """
    Extract requirements from DOCX with properly mapped comments.
    
    Returns:
        List of dicts with format:
        {
            'requirement': str,  # The requirement text
            'comment': str,      # Associated comment (if any)
            'comment_author': str,  # Comment author
            'comment_date': str,    # Comment date
            'section_number': str   # Section number (e.g., "5.2.1")
        }
    """
    requirements = []
    
    try:
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as docx:
            # Step 1: Read all comments
            comments_map = {}
            try:
                comments_xml = docx.read('word/comments.xml')
                comments_tree = ET.fromstring(comments_xml)
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                
                for comment in comments_tree.findall('.//w:comment', ns):
                    cid = comment.attrib.get(f"{{{ns['w']}}}id")
                    author = comment.attrib.get(f"{{{ns['w']}}}author", "Unknown")
                    date = comment.attrib.get(f"{{{ns['w']}}}date", "")
                    
                    # Extract comment text
                    comment_text = ''.join(comment.itertext()).strip()
                    
                    comments_map[cid] = {
                        'text': comment_text,
                        'author': author,
                        'date': date
                    }
            except KeyError:
                pass  # No comments in document
            
            # Step 2: Read document and map comments to text using ranges
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            # Track comment ranges
            active_comment_id = None
            
            # Process all paragraphs
            for para in tree.findall('.//w:p', ns):
                # Check for comment range start
                comment_start = para.find('.//w:commentRangeStart', ns)
                if comment_start is not None:
                    active_comment_id = comment_start.attrib.get(f"{{{ns['w']}}}id")
                
                # Extract paragraph text
                text_nodes = para.findall('.//w:t', ns)
                para_text = ''.join([t.text or '' for t in text_nodes]).strip()
                
                # Check for comment range end
                comment_end = para.find('.//w:commentRangeEnd', ns)
                
                # If we have text, process it
                if para_text and len(para_text) > 10:
                    # Detect if this is a requirement
                    is_requirement = (
                        # Numbered sections (e.g., 5.2.1, 7.3.2)
                        re.match(r'^\d+(\.\d+)+', para_text) or
                        # Strong requirement keywords
                        re.search(r'\b(shall|must|should|required|will|provide)\b', para_text, re.I) or
                        # System/equipment specifications
                        re.search(r'\b(system|supplier|vendor|equipment|specification)\b', para_text, re.I) or
                        # Technical specifications with measurements
                        re.search(r'\d+\s*(kg|l|ml|°c|bar|psi|rpm|%|v|amp)', para_text, re.I)
                    )
                    
                    if is_requirement:
                        # Extract section number if present
                        section_match = re.match(r'^(\d+(?:\.\d+)+)', para_text)
                        section_number = section_match.group(1) if section_match else ""
                        
                        # Get associated comment
                        comment_info = comments_map.get(active_comment_id) if active_comment_id else None
                        
                        requirements.append({
                            'requirement': para_text,
                            'comment': comment_info['text'] if comment_info else None,
                            'comment_author': comment_info['author'] if comment_info else None,
                            'comment_date': comment_info['date'] if comment_info else None,
                            'section_number': section_number
                        })
                
                # Reset active comment when range ends
                if comment_end is not None:
                    active_comment_id = None
            
            # Step 3: Also process tables (requirements are often in tables)
            for table in tree.findall('.//w:tbl', ns):
                active_comment_id = None
                
                for row in table.findall('.//w:tr', ns):
                    row_texts = []
                    row_comment_id = None
                    
                    for cell in row.findall('.//w:tc', ns):
                        # Check for comment range in cell
                        comment_start = cell.find('.//w:commentRangeStart', ns)
                        if comment_start is not None:
                            row_comment_id = comment_start.attrib.get(f"{{{ns['w']}}}id")
                        
                        # Extract cell text
                        cell_text = ''.join([t.text or '' for t in cell.findall('.//w:t', ns)]).strip()
                        if cell_text:
                            row_texts.append(cell_text)
                        
                        comment_end = cell.find('.//w:commentRangeEnd', ns)
                        if comment_end is not None:
                            row_comment_id = None
                    
                    # Combine row text
                    row_text = ' | '.join(row_texts)
                    
                    if row_text and len(row_text) > 10:
                        # Check if it's a requirement (not a header)
                        is_requirement = (
                            re.match(r'^\d+(\.\d+)+', row_text) or
                            re.search(r'\b(shall|must|should|required|specification)\b', row_text, re.I) or
                            # Skip obvious table headers
                            not re.match(r'^(sr\.?\s*no|topics?|page\s*no)', row_text, re.I)
                        ) and not (
                            # Skip TOC entries
                            re.match(r'^\d+\.\d+\s*\|\s*[A-Z\s]+\s*\|\s*\d+$', row_text)
                        )
                        
                        if is_requirement:
                            section_match = re.match(r'^(\d+(?:\.\d+)+)', row_text)
                            section_number = section_match.group(1) if section_match else ""
                            
                            comment_info = comments_map.get(row_comment_id) if row_comment_id else None
                            
                            requirements.append({
                                'requirement': row_text,
                                'comment': comment_info['text'] if comment_info else None,
                                'comment_author': comment_info['author'] if comment_info else None,
                                'comment_date': comment_info['date'] if comment_info else None,
                                'section_number': section_number
                            })
    
    except Exception as e:
        print(f"Error extracting requirements: {e}")
        import traceback
        traceback.print_exc()
    
    return requirements


def format_requirements_output(requirements: List[Dict]) -> str:
    """
    Format requirements in a clean, readable way.
    Groups by section and clearly shows comments.
    """
    output = []
    current_section = ""
    
    for i, req in enumerate(requirements, 1):
        # Add section header if changed
        section = req['section_number']
        if section and section != current_section:
            major_section = section.split('.')[0]
            output.append(f"\n{'='*80}")
            output.append(f"SECTION {section}")
            output.append(f"{'='*80}\n")
            current_section = section
        
        # Format requirement
        output.append(f"{i}. Requirement: {req['requirement']}")
        
        # Add comment if present
        if req['comment']:
            output.append(f"   💬 Comment ({req['comment_author']}): {req['comment']}")
        
        output.append("")  # Blank line
    
    return '\n'.join(output)


# Example usage function
def test_extraction(docx_path: str):
    """Test the extraction on a DOCX file"""
    with open(docx_path, 'rb') as f:
        docx_bytes = f.read()
    
    print("🔍 Extracting requirements with comments...")
    requirements = extract_requirements_with_comments(docx_bytes)
    
    print(f"\n✅ Extracted {len(requirements)} requirements")
    
    # Count requirements with comments
    with_comments = sum(1 for r in requirements if r['comment'])
    print(f"   📝 {with_comments} have comments")
    
    # Display formatted output
    print("\n" + "="*80)
    print("EXTRACTED REQUIREMENTS WITH COMMENTS")
    print("="*80)
    print(format_requirements_output(requirements))
    
    return requirements


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        docx_file = sys.argv[1]
    else:
        docx_file = "Novugen_URS IGL (1).docx"
    
    test_extraction(docx_file)
