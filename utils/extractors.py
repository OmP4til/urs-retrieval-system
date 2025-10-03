import io
from typing import List, Dict, Any
import pdfplumber
from docx import Document
from lxml import etree
import zipfile
import openpyxl
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes

OOXML_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def get_docx_comments_map(docx_file_bytes: bytes) -> Dict[str, str]:
    """Return dict {comment_id: comment_text} for a DOCX file."""
    comments_map = {}
    with zipfile.ZipFile(io.BytesIO(docx_file_bytes)) as z:
        if "word/comments.xml" not in z.namelist():
            return {}
        comments_xml = z.read("word/comments.xml")
        et = etree.XML(comments_xml)
        comments = et.xpath("//w:comment", namespaces=OOXML_NS)
        for c in comments:
            cid = c.xpath("@w:id", namespaces=OOXML_NS)[0]
            text = c.xpath("string(.)", namespaces=OOXML_NS)
            comments_map[cid] = text
    return comments_map


def get_paragraph_comments(paragraph, comments_dict: Dict[str, str]) -> List[str]:
    """Return list of comment texts attached to a paragraph."""
    comments = []
    for run in paragraph.runs:
        refs = run._r.xpath("./w:commentReference")
        if refs:
            cid = refs[0].xpath("@w:id", namespaces=OOXML_NS)[0]
            if cid in comments_dict:
                comments.append(comments_dict[cid])
    return comments


def extract_structured_content(uploaded_file) -> List[Dict[str, Any]]:
    """
    Returns a list of page dicts for PDF/DOCX/XLSX/TXT:
    {
      "page_number": int or sheet name,
      "content": text,
      "tables": [table_text, ...],
      "images_count": int,
      "content_type": "pdf"|"docx"|"xlsx"|"txt",
      "comments": [..]
    }
    """
    try:
        file_bytes = uploaded_file.read()
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    except Exception:
        raise

    name = (getattr(uploaded_file, "name", "") or "").lower()
    pages = []

    try:
        # ---------------- PDF ----------------
        if name.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    tables = []
                    try:
                        raw_tables = page.extract_tables()
                        for tbl in raw_tables:
                            rows = [
                                " | ".join(
                                    [str(cell) if cell is not None else "" for cell in row]
                                )
                                for row in tbl
                            ]
                            tables.append("\n".join(rows))
                    except Exception:
                        pass

                    if not text.strip():
                        try:
                            pil = page.to_image(resolution=200).original
                            text = pytesseract.image_to_string(pil) or ""
                        except Exception:
                            try:
                                images = convert_from_bytes(
                                    file_bytes, first_page=i, last_page=i, dpi=200
                                )
                                if images:
                                    text = pytesseract.image_to_string(images[0]) or ""
                            except Exception:
                                pass

                    content = text
                    if tables:
                        content += "\n\nTABLES:\n" + "\n\n".join(tables)

                    pages.append(
                        {
                            "page_number": i,
                            "content": content.strip(),
                            "tables": tables,
                            "images_count": len(page.images)
                            if hasattr(page, "images")
                            else 0,
                            "content_type": "pdf",
                            "comments": [],
                        }
                    )

        # ---------------- DOCX ----------------
        elif name.endswith(".docx"):
            doc = Document(io.BytesIO(file_bytes))
            comments_dict = get_docx_comments_map(file_bytes)

            # Extract paragraphs with their own comments
            for i, para in enumerate(doc.paragraphs, start=1):
                if not para.text.strip():
                    continue
                para_comments = get_paragraph_comments(para, comments_dict)
                pages.append(
                    {
                        "page_number": i,  # not real pages, but sequence index
                        "content": para.text.strip(),
                        "tables": [],
                        "images_count": 0,
                        "content_type": "docx",
                        "comments": para_comments,
                    }
                )

            # Extract tables as separate blocks
            for t in doc.tables:
                rows = []
                for r in t.rows:
                    cells = [c.text.strip() for c in r.cells]
                    rows.append(" | ".join(cells))
                tbl_text = "\n".join(rows)
                if tbl_text.strip():
                    pages.append(
                        {
                            "page_number": None,
                            "content": tbl_text,
                            "tables": [tbl_text],
                            "images_count": 0,
                            "content_type": "docx-table",
                            "comments": [],
                        }
                    )

        # ---------------- XLSX ----------------
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            wb = openpyxl.load_workbook(
                io.BytesIO(file_bytes), read_only=True, data_only=True
            )
            for sheetname in wb.sheetnames:
                ws = wb[sheetname]
                rows_text = []
                for row in ws.iter_rows(values_only=True):
                    rows_text.append(
                        " | ".join([str(c) if c is not None else "" for c in row])
                    )
                content = "\n".join(rows_text)
                pages.append(
                    {
                        "page_number": sheetname,
                        "content": content.strip(),
                        "tables": [content] if content else [],
                        "images_count": 0,
                        "content_type": "xlsx",
                        "comments": [],  # TODO: Excel cell comments support
                    }
                )

        # ---------------- TXT ----------------
        elif name.endswith(".txt"):
            try:
                text = file_bytes.decode("utf-8")
            except Exception:
                text = file_bytes.decode("latin-1", errors="ignore")
            pages.append(
                {
                    "page_number": 1,
                    "content": text.strip(),
                    "tables": [],
                    "images_count": 0,
                    "content_type": "txt",
                    "comments": [],
                }
            )

        # ---------------- Unknown ----------------
        else:
            try:
                text = file_bytes.decode("utf-8")
            except Exception:
                text = file_bytes.decode("latin-1", errors="ignore")
            pages.append(
                {
                    "page_number": 1,
                    "content": text.strip(),
                    "tables": [],
                    "images_count": 0,
                    "content_type": "unknown",
                    "comments": [],
                }
            )

    except Exception as e:
        print(
            f"[extract_structured_content] error extracting {getattr(uploaded_file,'name', '')}: {e}"
        )
        return []

    return pages
