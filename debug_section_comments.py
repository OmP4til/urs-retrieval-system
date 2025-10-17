"""
Debug script to see all section headers and check if any have comments
"""
import zipfile
import xml.etree.ElementTree as ET
import re

docx_file = "URS Coating Machine Rev 1 - GLATT comments 03092025.docx"

with open(docx_file, 'rb') as f:
    with zipfile.ZipFile(f) as docx:
        # Get comments
        try:
            comments_xml = docx.read('word/comments.xml')
            comments_tree = ET.fromstring(comments_xml)
            ns_c = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            comments_map = {}
            for comment in comments_tree.findall('.//w:comment', ns_c):
                cid = comment.attrib.get(f"{{{ns_c['w']}}}id")
                author = comment.attrib.get(f"{{{ns_c['w']}}}author", "Unknown")
                text = ''.join(comment.itertext()).strip()
                comments_map[cid] = {'author': author, 'text': text}
            
            print(f"Total comments in document: {len(comments_map)}\n")
        except:
            comments_map = {}
        
        # Parse document
        document_xml = docx.read('word/document.xml')
        tree = ET.fromstring(document_xml)
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        print("SECTION HEADERS AND COMMENTS:")
        print("="*80)
        
        for para in tree.findall('.//w:p', ns):
            # Check for comment range
            comment_start = para.find('.//w:commentRangeStart', ns)
            comment_id = comment_start.attrib.get(f"{{{ns['w']}}}id") if comment_start is not None else None
            
            # Get text
            text = ''.join([t.text or '' for t in para.findall('.//w:t', ns)]).strip()
            
            if text:
                # Check if it looks like a section header
                section_header_match = re.match(r'^(\d+(?:\.\d+)*)\s+([A-Z\s/\-&]+)$', text)
                
                if section_header_match:
                    section_num = section_header_match.group(1)
                    section_title = section_header_match.group(2)
                    
                    if comment_id:
                        comment_info = comments_map.get(comment_id, {})
                        print(f"\n✅ SECTION WITH COMMENT:")
                        print(f"   Section: {section_num} {section_title}")
                        print(f"   Comment by {comment_info.get('author', 'Unknown')}: {comment_info.get('text', 'No text')[:100]}...")
                    else:
                        print(f"\n📄 Section (no comment): {section_num} {section_title}")

print("\n" + "="*80)
print("CHECKING SAMPLE PARAGRAPHS FOR COMMENT PATTERNS:")
print("="*80)

with open(docx_file, 'rb') as f:
    with zipfile.ZipFile(f) as docx:
        document_xml = docx.read('word/document.xml')
        tree = ET.fromstring(document_xml)
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        count = 0
        for para in tree.findall('.//w:p', ns):
            comment_start = para.find('.//w:commentRangeStart', ns)
            if comment_start is not None:
                comment_id = comment_start.attrib.get(f"{{{ns['w']}}}id")
                text = ''.join([t.text or '' for t in para.findall('.//w:t', ns)]).strip()
                
                comment_info = comments_map.get(comment_id, {})
                print(f"\n💬 Comment ID {comment_id} by {comment_info.get('author')}:")
                print(f"   Text: {text[:150]}")
                print(f"   Comment: {comment_info.get('text', '')[:150]}")
                
                count += 1
                if count >= 10:
                    break
