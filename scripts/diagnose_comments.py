"""
Diagnose comment matching issues in URS Coating Machine document
Shows what text has comments in DOCX vs what requirements were extracted
"""
import sys
sys.path.insert(0, 'c:\\vv')

import zipfile
import io
from lxml import etree
from collections import defaultdict
from typing import Dict, List, Any

# OOXML namespaces
OOXML_NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
}

def get_comments_with_text(file_path: str):
    """Extract comments and their associated text from DOCX"""
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        if "word/comments.xml" not in z.namelist():
            print("No comments found in document")
            return {}
        
        # Parse comments
        comments_xml = z.read("word/comments.xml")
        root = etree.fromstring(comments_xml)
        
        comments_dict = {}
        for c in root.xpath("//w:comment", namespaces=OOXML_NS):
            cid = c.get(f"{{{OOXML_NS['w']}}}id")
            text_nodes = c.xpath('.//w:t', namespaces=OOXML_NS)
            text = ' '.join([t.text for t in text_nodes if t is not None and t.text])
            author = (c.xpath("@w:author", namespaces=OOXML_NS) or [None])[0]
            comments_dict[cid] = {
                "id": cid,
                "text": text.strip(),
                "author": author
            }
        
        # Parse document to find commented text
        document_xml = z.read("word/document.xml")
        doc_root = etree.fromstring(document_xml)
        
        text_to_comments = {}
        comment_text_buffers = defaultdict(list)
        active_comment_ids = []
        
        for elem in doc_root.iter():
            try:
                tag = etree.QName(elem.tag).localname
            except:
                tag = elem.tag.split('}')[-1]
            
            if tag == "commentRangeStart":
                cid = elem.get(f"{{{OOXML_NS['w']}}}id")
                if cid and cid in comments_dict:
                    active_comment_ids.append(cid)
                    comment_text_buffers.setdefault(cid, [])
            
            elif tag == "commentRangeEnd":
                cid = elem.get(f"{{{OOXML_NS['w']}}}id")
                if cid in active_comment_ids:
                    active_comment_ids.remove(cid)
                if cid and cid in comment_text_buffers:
                    parts = comment_text_buffers[cid]
                    commented_text = ' '.join(parts).strip()
                    if commented_text:
                        text_to_comments[commented_text] = comments_dict[cid]
                    comment_text_buffers.pop(cid, None)
            
            elif tag == "t":
                if active_comment_ids:
                    value = elem.text or ''
                    if value:
                        for cid in active_comment_ids:
                            comment_text_buffers[cid].append(value)
        
        return text_to_comments

# Path to the document
doc_path = r"C:\vv\URS Coating Machine Rev 1 - GLATT comments 03092025.docx"

print("Analyzing URS Coating Machine document...")
print("=" * 100)

text_to_comments = get_comments_with_text(doc_path)

print(f"\nFound {len(text_to_comments)} commented text segments\n")

for i, (text, comment) in enumerate(list(text_to_comments.items())[:10], 1):
    print(f"\n{i}. COMMENTED TEXT:")
    print(f"   {text[:150]}...")
    print(f"\n   COMMENT:")
    print(f"   Author: {comment['author']}")
    print(f"   Text: {comment['text'][:150]}...")
    print("\n" + "-" * 100)
