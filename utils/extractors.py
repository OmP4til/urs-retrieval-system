
import io
from typing import List, Dict, Any
import pdfplumber
from docx import Document
from lxml import etree
import zipfile
import openpyxl

# Note: The following imports have external dependencies that must be installed.
# For OCR (PDFs with images):
# - pytesseract: pip install pytesseract
# - pdf2image: pip install pdf2image
# - Google Tesseract: Must be installed on the system (e.g., via brew, apt-get)
# - Poppler: Required by pdf2image (e.g., via brew, apt-get)
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    
OOXML_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def get_docx_comments_with_text_mapping(docx_file_bytes: bytes) -> Dict[str, Dict[str, Any]]:
    """Return dict mapping text content to comments for a DOCX file."""
    try:
        with zipfile.ZipFile(io.BytesIO(docx_file_bytes)) as z:
            # Get comments
            if "word/comments.xml" not in z.namelist():
                return {}
            
            comments_xml = z.read("word/comments.xml")
            comments_et = etree.XML(comments_xml)
            
            # Parse all comments first
            comments_dict = {}
            comments = comments_et.xpath("//w:comment", namespaces=OOXML_NS)
            for c in comments:
                cid = c.xpath("@w:id", namespaces=OOXML_NS)[0]
                text = c.xpath("string(.)", namespaces=OOXML_NS)
                author = (c.xpath("@w:author", namespaces=OOXML_NS) or [None])[0]
                initials = (c.xpath("@w:initials", namespaces=OOXML_NS) or [None])[0]
                date = (c.xpath("@w:date", namespaces=OOXML_NS) or [None])[0]
                comments_dict[cid] = {
                    "id": cid,
                    "text": text,
                    "author": author,
                    "initials": initials,
                    "date": date,
                }
            
            # Parse document to map comments to text
            document_xml = z.read("word/document.xml")
            doc_root = etree.fromstring(document_xml)
            
            # Find all comment references and their associated text
            text_to_comments = {}
            comment_refs = doc_root.xpath('.//w:commentReference', namespaces=OOXML_NS)
            
            for ref in comment_refs:
                comment_id = ref.get(f"{{{OOXML_NS['w']}}}id")
                
                if comment_id not in comments_dict:
                    continue
                
                # Get the parent paragraph (this is the key - work at paragraph level)
                parent_para = ref.xpath('ancestor::w:p', namespaces=OOXML_NS)
                
                if parent_para:
                    # Get text content of just this paragraph
                    para_text_nodes = parent_para[0].xpath('.//w:t', namespaces=OOXML_NS)
                    para_text = ''.join([node.text for node in para_text_nodes if node.text]).strip()
                    
                    if para_text:
                        # Also try to get the parent cell content for broader matching
                        parent_tc = ref.xpath('ancestor::w:tc', namespaces=OOXML_NS)
                        
                        # Store both paragraph-level and cell-level mapping
                        if para_text not in text_to_comments:
                            text_to_comments[para_text] = []
                        text_to_comments[para_text].append(comments_dict[comment_id])
                        
                        # If in a table cell, also map to full cell content for fallback
                        if parent_tc:
                            tc_text_nodes = parent_tc[0].xpath('.//w:t', namespaces=OOXML_NS)
                            full_cell_text = ''.join([node.text for node in tc_text_nodes if node.text]).strip()
                            
                            if full_cell_text and full_cell_text != para_text:
                                if full_cell_text not in text_to_comments:
                                    text_to_comments[full_cell_text] = []
                                text_to_comments[full_cell_text].append(comments_dict[comment_id])
            
            return text_to_comments
            
    except Exception:
        # Ignore errors if comments can't be parsed
        pass
    return {}


def get_paragraph_comments(paragraph, comments_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return list of comment dicts attached to a paragraph.

    Each item contains: {id, text, author, initials, date}
    """
    comments: List[Dict[str, Any]] = []
    try:
        # Get the XML element for this paragraph
        p_element = paragraph._element
        
        # Find all comment reference elements using lxml etree
        from lxml import etree
        # Convert paragraph element to lxml element if needed
        if hasattr(p_element, 'xml'):
            # This is a python-docx element, get raw XML
            xml_str = p_element.xml
            element = etree.fromstring(xml_str)
        else:
            element = p_element
            
        refs = element.xpath(".//w:commentReference", namespaces=OOXML_NS)
        for ref in refs:
            cid = ref.get(f"{{{OOXML_NS['w']}}}id")
            if cid and cid in comments_dict:
                comments.append(comments_dict[cid])
    except Exception:
        pass
    return comments


def get_all_comments_fallback(comments_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return all comments as a fallback when paragraph mapping fails."""
    return list(comments_dict.values())


def extract_structured_content(uploaded_file) -> List[Dict[str, Any]]:
    """
    Returns a list of page/chunk dicts for PDF/DOCX/XLSX/TXT.
    """
    try:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    name = getattr(uploaded_file, "name", "").lower()
    pages = []

    try:
        # ---------------- PDF ----------------
        if name.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    # Improved text extraction to handle line breaks better
                    text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    
                    # Fallback to OCR if text is sparse and dependencies are available
                    if len(text.strip()) < 100 and OCR_AVAILABLE:
                        print(f"PDF page {i}: Low text content, attempting OCR.")
                        try:
                            images = convert_from_bytes(
                                file_bytes, first_page=i, last_page=i, dpi=200
                            )
                            if images:
                                ocr_text = pytesseract.image_to_string(images[0]) or ""
                                text += "\n\n" + ocr_text # Append OCR text
                        except Exception as ocr_error:
                            print(f"OCR failed for page {i}: {ocr_error}")
                            print("Ensure Tesseract and Poppler are installed and in your PATH.")

                    # Extract tables separately
                    tables = []
                    try:
                        raw_tables = page.extract_tables()
                        for tbl in raw_tables:
                            rows = [" | ".join(map(str, row)) for row in tbl]
                            tables.append("\n".join(rows))
                    except Exception:
                        pass
                    
                    content = text
                    if tables:
                        content += "\n\n--- TABLES ---\n" + "\n\n".join(tables)

                    pages.append({
                        "page_number": i,
                        "content": content.strip(),
                        "tables": tables,
                        "content_type": "pdf",
                        "comments": [],
                    })

        # ---------------- DOCX ----------------
        elif name.endswith(".docx"):
            doc = Document(io.BytesIO(file_bytes))
            text_to_comments = get_docx_comments_with_text_mapping(file_bytes)
            
            # Process main paragraphs first
            content_buffer = []
            comments_buffer = []
            page_num = 1
            
            # Iterate through document elements (paragraphs and tables) in order
            for element in doc.element.body:
                if element.tag.endswith('p'):
                    # Find the corresponding paragraph object
                    para = None
                    for p in doc.paragraphs:
                        if p._p == element:
                            para = p
                            break
                    
                    if para and para.text.strip():
                        content_buffer.append(para.text)
                        # Look for comments mapped to this specific text
                        para_comments = text_to_comments.get(para.text.strip(), [])
                        comments_buffer.extend(para_comments)
                
                elif element.tag.endswith('tbl'):
                    # Find the corresponding table object
                    table = None
                    for t in doc.tables:
                        if t._tbl == element:
                            table = t
                            break
                    
                    if not table:
                        continue
                    
                    # First, add any preceding text paragraphs as a page
                    if content_buffer:
                        pages.append({
                            "page_number": page_num,
                            "content": "\n".join(content_buffer).strip(),
                            "tables": [],
                            "content_type": "docx",
                            "comments": comments_buffer,
                        })
                        page_num += 1
                        content_buffer, comments_buffer = [], []

                    # Now, process the table row by row to map comments properly
                    table_header = None
                    
                    for row_idx, row in enumerate(table.rows):
                        row_text_cells = []
                        row_comments = []
                        
                        for cell in row.cells:
                            cell_text = cell.text.strip()
                            row_text_cells.append(cell_text)
                            
                            # Look for comments in this specific cell - ULTRA STRICT MATCHING
                            cell_comments = []
                            
                            # 1. Exact match on full cell text (only if cell has substantial content)
                            if cell_text and len(cell_text.strip()) > 5 and cell_text in text_to_comments:
                                cell_comments.extend(text_to_comments[cell_text])
                            
                            # 2. Check individual paragraphs within the cell for exact matches
                            for para in cell.paragraphs:
                                para_text = para.text.strip()
                                if para_text and len(para_text) > 5 and para_text in text_to_comments:
                                    cell_comments.extend(text_to_comments[para_text])
                            
                            # 3. VERY STRICT partial matching - only for substantial, specific content
                            for comment_text, comments in text_to_comments.items():
                                if comment_text and len(comment_text.strip()) > 20:  # Only substantial comment text
                                    # Check if the commented text appears exactly in this cell
                                    if comment_text.strip() in cell_text and len(cell_text) > 50:  # Cell must also be substantial
                                        # Additional validation: make sure it's not a generic phrase
                                        # Skip if the match is too generic (common words/phrases)
                                        generic_phrases = ['the equipment', 'shall be', 'should be', 'requirements', 'specifications', 'as per', 'according to']
                                        if not any(phrase in comment_text.strip().lower() for phrase in generic_phrases):
                                            # Final check: ensure the match is meaningful (not just common words)
                                            if len(set(comment_text.strip().split()) & set(cell_text.split())) >= 3:  # At least 3 words in common
                                                cell_comments.extend(comments)
                            
                            # Add unique cell comments to this row's collection (with safety limit)
                            for comment in cell_comments:
                                comment_key = (comment.get('id'), comment.get('text', ''))
                                if not any((c.get('id'), c.get('text', '')) == comment_key for c in row_comments):
                                    # Safety check: prevent excessive comments per page (likely wrong assignment)
                                    if len(row_comments) < 8:  # Maximum 8 comments per requirement
                                        row_comments.append(comment)
                                    # If we hit the limit, it's likely a wrong assignment, so skip
                        
                        row_content = " | ".join(row_text_cells)
                        
                        # Skip empty rows
                        if not row_content.strip():
                            continue
                        
                        # First row might be header - store it for context
                        if row_idx == 0 and not row_comments:
                            table_header = row_content
                            continue
                        
                        # Create a separate page for each substantial row with content
                        row_display_content = row_content
                        if table_header:
                            row_display_content = f"{table_header}\n{row_content}"
                        
                        if row_content.strip():
                            pages.append({
                                "page_number": f"Table-{page_num}-Row-{row_idx+1}",
                                "content": row_display_content,
                                "tables": [row_display_content],
                                "content_type": "docx-table-row",
                                "comments": row_comments,
                            })
                            page_num += 1

            # Add any remaining text content at the end of the document
            if content_buffer:
                pages.append({
                    "page_number": page_num,
                    "content": "\n".join(content_buffer).strip(),
                    "tables": [],
                    "content_type": "docx",
                    "comments": comments_buffer,
                })

        # ---------------- XLSX ----------------
        elif name.endswith((".xlsx", ".xls")):
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
            for sheetname in wb.sheetnames:
                ws = wb[sheetname]
                rows_text = [" | ".join(str(c) if c is not None else "" for c in row) for row in ws.iter_rows(values_only=True)]
                content = "\n".join(rows_text)
                if content.strip():
                    pages.append({
                        "page_number": sheetname,
                        "content": content.strip(),
                        "tables": [content],
                        "content_type": "xlsx",
                        "comments": [],
                    })

        # ---------------- TXT ----------------
        elif name.endswith(".txt"):
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1", errors="ignore")
            pages.append({
                "page_number": 1,
                "content": text.strip(),
                "tables": [],
                "content_type": "txt",
                "comments": [],
            })

    except Exception as e:
        print(f"[extract_structured_content] error extracting '{name}': {e}")
        return []

    return pages
