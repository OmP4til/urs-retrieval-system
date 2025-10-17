"""
Smart DOCX Comment Extraction with Section-Level Comment Propagation
- Comments on section headers → Applied to all requirements in that section
- Comments on specific requirements → Applied only to that requirement
"""
import zipfile
import xml.etree.ElementTree as ET
import re
from typing import Dict, List, Tuple
import io


def extract_requirements_with_smart_comments(docx_bytes: bytes) -> List[Dict]:
    """
    Extract requirements with intelligent comment mapping:
    - Section header comments → propagate to all requirements in that section
    - Individual requirement comments → apply only to that requirement
    
    Returns:
        List of dicts with format:
        {
            'requirement': str,
            'comment': str,           # Individual requirement comment
            'comment_author': str,
            'comment_date': str,
            'section_number': str,
            'section_comment': str,   # Comment from section header (if any)
            'section_comment_author': str,
            'section_comment_date': str
        }
    """
    requirements = []
    section_comments = {}  # Store comments by section number
    
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
            
            # Step 2: Parse document and extract requirements with comments
            document_xml = docx.read('word/document.xml')
            tree = ET.fromstring(document_xml)
            
            active_comment_id = None
            current_section = None
            
            # Process paragraphs
            for para in tree.findall('.//w:p', ns):
                # Check for comment range start
                comment_start = para.find('.//w:commentRangeStart', ns)
                if comment_start is not None:
                    active_comment_id = comment_start.attrib.get(f"{{{ns['w']}}}id")
                
                # Extract paragraph text
                text_elements = para.findall('.//w:t', ns)
                para_text = ''.join([t.text or '' for t in text_elements]).strip()
                
                if para_text:
                    # Check if this is a SECTION HEADER
                    section_header_match = re.match(r'^(\d+(?:\.\d+)*)\s+([A-Z\s/\-&]+)$', para_text)
                    
                    if section_header_match:
                        # This is a section header!
                        section_num = section_header_match.group(1)
                        section_title = section_header_match.group(2).strip()
                        current_section = section_num
                        
                        # If there's a comment on this section header, store it
                        if active_comment_id:
                            comment_info = comments_map.get(active_comment_id)
                            if comment_info:
                                section_comments[section_num] = {
                                    'text': comment_info['text'],
                                    'author': comment_info['author'],
                                    'date': comment_info['date'],
                                    'section_title': section_title
                                }
                                print(f"📌 Section comment found: {section_num} - {comment_info['text'][:80]}...")
                    
                    else:
                        # This might be a requirement
                        is_requirement = (
                            len(para_text) > 15 and
                            (
                                re.match(r'^\d+(\.\d+)+', para_text) or
                                re.search(r'\b(shall|must|should|will|required?|specification|provide|ensure|design|system|equipment)\b', para_text, re.I)
                            ) and not (
                                re.match(r'^(page|section|table of contents|sr\.?\s*no|topics?|revision|document|prepared|reviewed|approved)', para_text, re.I)
                            )
                        )
                        
                        if is_requirement:
                            # Extract section number from requirement text
                            section_match = re.match(r'^(\d+(?:\.\d+)+)', para_text)
                            section_number = section_match.group(1) if section_match else ""
                            
                            # Determine which section this requirement belongs to
                            req_section = section_number if section_number else current_section
                            
                            # Get individual requirement comment
                            req_comment_info = comments_map.get(active_comment_id) if active_comment_id else None
                            
                            # Get section-level comment (if this requirement belongs to a commented section)
                            section_comment_info = None
                            if req_section:
                                # Check for exact match or parent section
                                for sec_num, sec_comment in section_comments.items():
                                    if req_section.startswith(sec_num):
                                        section_comment_info = sec_comment
                                        break
                            
                            requirements.append({
                                'requirement': para_text,
                                'comment': req_comment_info['text'] if req_comment_info else None,
                                'comment_author': req_comment_info['author'] if req_comment_info else None,
                                'comment_date': req_comment_info['date'] if req_comment_info else None,
                                'section_number': section_number,
                                'section_comment': section_comment_info['text'] if section_comment_info else None,
                                'section_comment_author': section_comment_info['author'] if section_comment_info else None,
                                'section_comment_date': section_comment_info['date'] if section_comment_info else None,
                                'section_title': section_comment_info['section_title'] if section_comment_info else None
                            })
                
                # Check for comment range end
                comment_end = para.find('.//w:commentRangeEnd', ns)
                if comment_end is not None:
                    active_comment_id = None
            
            # Step 3: Process tables
            for table in tree.findall('.//w:tbl', ns):
                active_comment_id = None
                
                for row in table.findall('.//w:tr', ns):
                    row_texts = []
                    row_comment_id = None
                    
                    for cell in row.findall('.//w:tc', ns):
                        # Check for comment range
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
                    
                    row_text = ' | '.join(row_texts)
                    
                    if row_text and len(row_text) > 10:
                        is_requirement = (
                            re.match(r'^\d+(\.\d+)+', row_text) or
                            re.search(r'\b(shall|must|should|required|specification)\b', row_text, re.I) or
                            not re.match(r'^(sr\.?\s*no|topics?|page\s*no)', row_text, re.I)
                        ) and not (
                            re.match(r'^\d+\.\d+\s*\|\s*[A-Z\s]+\s*\|\s*\d+$', row_text)
                        )
                        
                        if is_requirement:
                            section_match = re.match(r'^(\d+(?:\.\d+)+)', row_text)
                            section_number = section_match.group(1) if section_match else ""
                            
                            # Determine section
                            req_section = section_number if section_number else current_section
                            
                            # Get individual comment
                            req_comment_info = comments_map.get(row_comment_id) if row_comment_id else None
                            
                            # Get section comment
                            section_comment_info = None
                            if req_section:
                                for sec_num, sec_comment in section_comments.items():
                                    if req_section.startswith(sec_num):
                                        section_comment_info = sec_comment
                                        break
                            
                            requirements.append({
                                'requirement': row_text,
                                'comment': req_comment_info['text'] if req_comment_info else None,
                                'comment_author': req_comment_info['author'] if req_comment_info else None,
                                'comment_date': req_comment_info['date'] if req_comment_info else None,
                                'section_number': section_number,
                                'section_comment': section_comment_info['text'] if section_comment_info else None,
                                'section_comment_author': section_comment_info['author'] if section_comment_info else None,
                                'section_comment_date': section_comment_info['date'] if section_comment_info else None,
                                'section_title': section_comment_info['section_title'] if section_comment_info else None
                            })
    
    except Exception as e:
        print(f"Error extracting requirements: {e}")
        import traceback
        traceback.print_exc()
    
    return requirements


def format_smart_output(requirements: List[Dict]) -> str:
    """
    Format requirements with smart comment display:
    - Section comments shown at the beginning of each section
    - Individual requirement comments shown with each requirement
    """
    output = []
    current_section = ""
    section_comment_shown = set()
    
    for i, req in enumerate(requirements, 1):
        section = req['section_number'][:3] if req['section_number'] and len(req['section_number']) >= 3 else ""
        
        # Show section comment when section changes
        if section and section != current_section:
            current_section = section
            
            # Check if this section has a comment
            if req['section_comment'] and section not in section_comment_shown:
                output.append(f"\n{'='*80}")
                output.append(f"📂 SECTION {section}: {req['section_title'] or 'No Title'}")
                output.append(f"{'='*80}")
                output.append(f"📌 SECTION COMMENT ({req['section_comment_author']}):")
                output.append(f"   {req['section_comment']}")
                output.append(f"{'='*80}\n")
                section_comment_shown.add(section)
        
        # Show requirement
        output.append(f"{i}. Requirement: {req['requirement']}")
        
        # Show individual requirement comment (if any)
        if req['comment']:
            output.append(f"   💬 Requirement Comment ({req['comment_author']}): {req['comment']}")
        
        output.append("")  # Blank line
    
    return '\n'.join(output)


# Backward compatibility function names
extract_requirements_with_comments = extract_requirements_with_smart_comments
format_requirements_output = format_smart_output
