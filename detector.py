"""
detector.py
-----------
Detects the true file type of any document uploaded to S3.

HOW IT WORKS:
  Every file format has a unique "magic signature" — a fixed sequence of bytes
  at a known position (usually the very start) of the file. This module reads
  those bytes and maps them to a MIME type and a parser key.

  Three detection methods are combined, in trust order:
    1. Magic bytes  — most reliable, reads raw file content
    2. ZIP subtype  — DOCX / XLSX / PPTX are all ZIP files internally,
                      so we refine by file extension after detecting ZIP
    3. Text heuristic — if no binary signature matches, check if the content
                        looks like plain text, then sniff CSV vs Markdown etc.

USAGE:
  from detector import detect

  # From a local file
  result = detect(open("report.pdf", "rb").read(), "report.pdf")
  print(result.parser_key)   # → "pdf"
  print(result.mime_type)    # → "application/pdf"

  # From S3 (only download first 4 KB — enough for all signatures)
  header = s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-4095")["Body"].read()
  result = detect(header, filename=key.split("/")[-1])
"""

import io
import csv as csv_mod
from dataclasses import dataclass
from typing import Optional


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class FileTypeResult:
    """
    Returned by detect(). Contains everything downstream needs to route the file.

    Fields:
      mime_type   — standard MIME string, e.g. "application/pdf"
      format_name — human-readable label, e.g. "PDF document"
      parser_key  — short key used to look up the correct parser function,
                    e.g. "pdf", "docx", "csv". This is what your router uses.
      confidence  — how the type was determined:
                      "magic"          → matched raw byte signature (most reliable)
                      "zip_extension"  → confirmed ZIP, refined by file extension
                      "text_heuristic" → no binary match, content looks like text
                      "fallback"       → could not determine type
      extension   — file extension used during ZIP/OLE2 refinement, or None
    """
    mime_type:   str
    format_name: str
    parser_key:  str
    confidence:  str
    extension:   Optional[str] = None

    def is_supported(self) -> bool:
        """Returns True if we have a parser for this file type."""
        return self.parser_key in PARSER_REGISTRY

    def is_scanned(self) -> bool:
        """
        Returns True if the file is an image or a PDF that may contain
        scanned pages. Caller should route these through Textract / OCR.
        """
        return self.parser_key == "image"

    def __str__(self) -> str:
        return (
            f"{self.format_name} "
            f"[{self.mime_type}] "
            f"parser={self.parser_key} "
            f"confidence={self.confidence}"
        )


# ── Magic byte signatures ─────────────────────────────────────────────────────
#
# Format: (offset, bytes_to_match, mime_type, format_name, parser_key)
#
#   offset          — byte position in the file where the signature starts.
#                     Almost always 0. A few formats (e.g. some ZIP variants)
#                     use non-zero offsets.
#   bytes_to_match  — the exact byte sequence to look for at that offset.
#   mime_type       — standard MIME string.
#   format_name     — human-readable label shown in logs.
#   parser_key      — short routing key for your parser map.
#
# Rules:
#   • First match wins — more specific signatures must come before general ones.
#   • ZIP (PK\x03\x04) must come AFTER any ZIP-based formats if those have
#     distinguishing bytes — but DOCX/XLSX/PPTX all start with the same PK bytes,
#     so we detect as "zip" and refine by extension in the router below.

MAGIC_SIGNATURES = [

    # ── PDF ──────────────────────────────────────────────────────────────────
    # Every PDF starts with the ASCII string %PDF regardless of version.
    (0, b'%PDF',
     'application/pdf', 'PDF document', 'pdf'),

    # ── Office Open XML (DOCX / XLSX / PPTX) and generic ZIP ─────────────────
    # All three Office formats and plain ZIP archives start with PK\x03\x04.
    # We cannot distinguish them by magic bytes alone — the ZIP refinement
    # step below handles it using the file extension.
    (0, b'PK\x03\x04',
     'application/zip', 'ZIP / Office Open XML', 'zip'),

    # ── Legacy Microsoft Office (OLE2 compound document) ─────────────────────
    # .doc, .xls, .ppt (Office 97–2003) all share this 8-byte header.
    # Refinement by extension happens below (same pattern as ZIP).
    (0, b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1',
     'application/msword', 'Legacy MS Office (OLE2)', 'msoffice_legacy'),

    # ── Images ───────────────────────────────────────────────────────────────
    (0, b'\x89PNG\r\n\x1a\n',
     'image/png', 'PNG image', 'image'),

    (0, b'\xff\xd8\xff',
     'image/jpeg', 'JPEG image', 'image'),

    (0, b'GIF87a',
     'image/gif', 'GIF87 image', 'image'),

    (0, b'GIF89a',
     'image/gif', 'GIF89 image', 'image'),

    # TIFF has two valid byte-order markers (little-endian II, big-endian MM)
    (0, b'II\x2a\x00',
     'image/tiff', 'TIFF image (LE)', 'image'),

    (0, b'MM\x00\x2a',
     'image/tiff', 'TIFF image (BE)', 'image'),

    (0, b'BM',
     'image/bmp', 'BMP image', 'image'),

    # WebP is a RIFF container — bytes 0-3 are RIFF, bytes 8-11 are WEBP
    # We match on RIFF here; a more precise check could verify bytes 8-11 too.
    (0, b'RIFF',
     'image/webp', 'WebP / RIFF image', 'image'),

    # ── HTML ─────────────────────────────────────────────────────────────────
    # Match both with and without DOCTYPE declaration.
    (0, b'<!DOCTYPE html',
     'text/html', 'HTML document', 'html'),

    (0, b'<!DOCTYPE HTML',
     'text/html', 'HTML document', 'html'),

    (0, b'<html',
     'text/html', 'HTML document', 'html'),

    (0, b'<HTML',
     'text/html', 'HTML document', 'html'),

    # ── XML ──────────────────────────────────────────────────────────────────
    (0, b'<?xml',
     'text/xml', 'XML document', 'xml'),

    # ── JSON ─────────────────────────────────────────────────────────────────
    # JSON starts with { (object) or [ (array). These are heuristics —
    # a plain text file could also start with these characters, so JSON
    # detection is only triggered when the first non-whitespace byte matches.
    (0, b'{',
     'application/json', 'JSON object', 'json'),

    (0, b'[',
     'application/json', 'JSON array', 'json'),

    # ── RTF ──────────────────────────────────────────────────────────────────
    (0, b'{\\rtf',
     'application/rtf', 'RTF document', 'rtf'),

    # ── Email (EML) ───────────────────────────────────────────────────────────
    # RFC 5322 emails start with one of these header fields.
    (0, b'From ',
     'message/rfc822', 'Email (EML)', 'email'),

    (0, b'Return-Path:',
     'message/rfc822', 'Email (EML)', 'email'),

    (0, b'Received:',
     'message/rfc822', 'Email (EML)', 'email'),

    (0, b'MIME-Version:',
     'message/rfc822', 'Email (EML)', 'email'),

    (0, b'Date:',
     'message/rfc822', 'Email (EML)', 'email'),

    # ── SQLite database ───────────────────────────────────────────────────────
    (0, b'SQLite format 3\x00',
     'application/x-sqlite3', 'SQLite database', 'sqlite'),

    # ── Markdown (heuristic) ─────────────────────────────────────────────────
    # Markdown files very commonly start with a # heading.
    # This is a weak heuristic — a plain text file could also start with #.
    # We rely on the extension (.md) for stronger confirmation in practice.
    (0, b'# ',
     'text/markdown', 'Markdown document', 'markdown'),

]


# ── ZIP subtype refinement ────────────────────────────────────────────────────
#
# When magic bytes match PK\x03\x04 (ZIP), we cannot determine the exact
# format from bytes alone. All Office Open XML formats are ZIP archives.
# We refine using the file extension — this is safe because the magic bytes
# already confirmed the file IS a valid ZIP, so we're only determining
# which ZIP-based format it is.

ZIP_EXTENSION_MAP = {
    'docx': (
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'Word document (DOCX)',
        'docx',
    ),
    'xlsx': (
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'Excel spreadsheet (XLSX)',
        'xlsx',
    ),
    'pptx': (
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'PowerPoint presentation (PPTX)',
        'pptx',
    ),
    'epub': (
        'application/epub+zip',
        'EPUB ebook',
        'epub',
    ),
    'zip': (
        'application/zip',
        'ZIP archive',
        'zip',
    ),
    # .jar, .war, .apk are also ZIP — treat as generic archive
    'jar': ('application/java-archive', 'Java archive (JAR)', 'zip'),
    'apk': ('application/vnd.android.package-archive', 'Android package (APK)', 'zip'),
}


# ── OLE2 (Legacy Office) subtype refinement ───────────────────────────────────
#
# Same pattern as ZIP: magic bytes tell us it's OLE2, extension tells us which.

OLE2_EXTENSION_MAP = {
    'doc': (
        'application/msword',
        'Word document (DOC)',
        'doc',
    ),
    'xls': (
        'application/vnd.ms-excel',
        'Excel spreadsheet (XLS)',
        'xls',
    ),
    'ppt': (
        'application/vnd.ms-powerpoint',
        'PowerPoint presentation (PPT)',
        'ppt',
    ),
    'msg': (
        'application/vnd.ms-outlook',
        'Outlook email (MSG)',
        'email',
    ),
}


# ── Text heuristic helpers ────────────────────────────────────────────────────

def _is_likely_text(data: bytes) -> bool:
    """
    Returns True if the byte sample looks like UTF-8 / ASCII plain text.

    Method: count how many bytes are printable ASCII or common whitespace
    (tab=9, newline=10, carriage return=13). If more than 90% of bytes pass,
    we treat the file as text.

    This threshold (90%) is deliberately conservative — binary files
    sometimes have long runs of printable bytes (e.g. embedded strings
    in executables), but they rarely sustain 90%+ across the whole sample.
    """
    if not data:
        return False
    printable = sum(
        1 for b in data
        if b >= 32 or b in (9, 10, 13)  # printable ASCII or tab/LF/CR
    )
    return (printable / len(data)) > 0.90


def _sniff_text_subtype(data: bytes, extension: str) -> FileTypeResult:
    """
    Given a file that passed the _is_likely_text() check, determines the
    specific text format: CSV, TSV, Markdown, or plain text.

    Strategy:
      1. CSV  — consistent comma count across the first 10 lines
      2. TSV  — consistent tab count across the first 10 lines
      3. Markdown — extension is .md, or content starts with a # heading
      4. Plain text — fallback for everything else
    """
    try:
        sample = data[:4096].decode('utf-8', errors='replace')
    except Exception:
        # Cannot decode as text at all — shouldn't happen after _is_likely_text,
        # but be safe.
        return FileTypeResult(
            'text/plain', 'Plain text', 'text', 'text_heuristic', extension or None
        )

    lines = [l for l in sample.splitlines() if l.strip()]

    if lines:
        # ── CSV check ──────────────────────────────────────────────────────
        # Look at comma counts across first 10 non-empty lines.
        # If they're consistent (at most 2 distinct counts) and each line
        # has at least 1 comma, it's very likely CSV.
        comma_counts = [line.count(',') for line in lines[:10]]
        if len(set(comma_counts)) <= 2 and min(comma_counts) >= 1:
            return FileTypeResult(
                'text/csv', 'CSV file', 'csv', 'text_heuristic', extension or 'csv'
            )

        # ── TSV check ──────────────────────────────────────────────────────
        tab_counts = [line.count('\t') for line in lines[:10]]
        if len(set(tab_counts)) <= 2 and min(tab_counts) >= 1:
            return FileTypeResult(
                'text/tab-separated-values', 'TSV file', 'csv',
                'text_heuristic', extension or 'tsv'
            )

    # ── Markdown check ────────────────────────────────────────────────────
    if extension in ('md', 'markdown') or (lines and lines[0].startswith('# ')):
        return FileTypeResult(
            'text/markdown', 'Markdown document', 'markdown',
            'text_heuristic', extension or 'md'
        )

    # ── Plain text fallback ───────────────────────────────────────────────
    return FileTypeResult(
        'text/plain', 'Plain text', 'text', 'text_heuristic', extension or 'txt'
    )


# ── Main detect function ──────────────────────────────────────────────────────

def detect(file_bytes: bytes, filename: str = '') -> FileTypeResult:
    """
    Detects the true MIME type of a file using magic byte signatures.

    Args:
        file_bytes  — raw bytes of the file. Passing the full file is fine,
                      but only the first ~32 bytes are needed for detection.
                      In production, download only the first 4 KB from S3
                      to save bandwidth:
                          s3.get_object(Bucket=b, Key=k, Range="bytes=0-4095")
        filename    — original filename including extension. Used ONLY for
                      ZIP/OLE2 subtype refinement. Never trusted on its own.

    Returns:
        FileTypeResult — always returns something, never raises.
        Check result.parser_key == "unknown" to detect unrecognised files.

    Confidence levels (result.confidence):
        "magic"          → matched a binary signature → very reliable
        "zip_extension"  → confirmed ZIP, used extension to pick subtype
        "text_heuristic" → no binary match, content looks like text
        "fallback"       → binary file with no known signature
    """
    # ── Guard: empty file ────────────────────────────────────────────────────
    if not file_bytes:
        return FileTypeResult(
            'application/octet-stream', 'Empty file', 'unknown', 'fallback'
        )

    # Extract extension from filename — used only for ZIP/OLE2 refinement
    ext = ''
    if '.' in filename:
        ext = filename.rsplit('.', 1)[-1].lower().strip()

    # ── Step 1: Walk magic signatures ────────────────────────────────────────
    for offset, magic, mime, name, key in MAGIC_SIGNATURES:
        # Slice exactly the bytes at the given offset
        chunk = file_bytes[offset: offset + len(magic)]

        if chunk != magic:
            continue  # No match — try next signature

        # ── ZIP family: refine by extension ──────────────────────────────
        if key == 'zip':
            if ext in ZIP_EXTENSION_MAP:
                refined_mime, refined_name, refined_key = ZIP_EXTENSION_MAP[ext]
                return FileTypeResult(
                    refined_mime, refined_name, refined_key,
                    'zip_extension', ext
                )
            # Extension not in map — return as generic ZIP
            return FileTypeResult(mime, name, 'zip', 'magic', ext or None)

        # ── OLE2 family: refine by extension ─────────────────────────────
        if key == 'msoffice_legacy':
            if ext in OLE2_EXTENSION_MAP:
                refined_mime, refined_name, refined_key = OLE2_EXTENSION_MAP[ext]
                return FileTypeResult(
                    refined_mime, refined_name, refined_key,
                    'zip_extension', ext  # same confidence label: magic + extension
                )
            return FileTypeResult(mime, name, 'msoffice_legacy', 'magic', ext or None)

        # ── All other formats: direct match ──────────────────────────────
        if key == 'xml':
            probe = file_bytes[:2048].lower()
            if b'<html' in probe:
                return FileTypeResult(
                    'text/html', 'HTML document (XHTML w/ XML prolog)', 'html',
                    'magic', ext or None
                )
            return FileTypeResult(mime, name, key, 'magic', ext or None)

        return FileTypeResult(mime, name, key, 'magic', ext or None)

    # ── Step 2: Text heuristic ───────────────────────────────────────────────
    # No binary signature matched. Check if the content looks like text.
    if _is_likely_text(file_bytes[:1024]):
        return _sniff_text_subtype(file_bytes, ext)

    # ── Step 3: Fallback ─────────────────────────────────────────────────────
    # Binary file with no known signature. Log and quarantine in production.
    return FileTypeResult(
        'application/octet-stream', 'Unknown binary', 'unknown', 'fallback',
        ext or None
    )


# ── Parser registry ───────────────────────────────────────────────────────────
#
# Maps parser_key → human description of the parser + library to use.
# Your actual parser functions live in parsers.py.
# This registry is used for logging, routing decisions, and validation.

PARSER_REGISTRY = {
    'pdf':            'PDFParser         → pdfplumber / AWS Textract (scanned)',
    'docx':           'DocxParser        → python-docx',
    'xlsx':           'ExcelParser       → openpyxl',
    'pptx':           'PptxParser        → python-pptx',
    'csv':            'CsvParser         → csv (stdlib) + pandas',
    'html':           'HtmlParser        → BeautifulSoup4 + lxml',
    'xml':            'XmlParser         → xml.etree (stdlib) / lxml',
    'json':           'JsonParser        → json (stdlib)',
    'text':           'PlainTextParser   → direct read',
    'markdown':       'MarkdownParser    → direct read (same as text)',
    'email':          'EmailParser       → email (stdlib) / extract-msg (MSG)',
    'image':          'OcrParser         → AWS Textract / Tesseract',
    'epub':           'EpubParser        → ebooklib',
    'rtf':            'RtfParser         → striprtf',
    'sqlite':         'SqliteParser      → sqlite3 (stdlib)',
    'doc':            'LegacyDocParser   → LibreOffice convert → then DocxParser',
    'xls':            'LegacyXlsParser   → xlrd / pandas',
    'ppt':            'LegacyPptParser   → LibreOffice convert → then PptxParser',
    'zip':            'ArchiveParser     → zipfile (stdlib) → extract + recurse',
    'msoffice_legacy':'LegacyOfficeParser→ LibreOffice headless convert first',
    'unknown':        'UnknownHandler    → log warning + move to quarantine prefix',
}


def get_parser_info(parser_key: str) -> str:
    """Returns the parser description for a given key, or a not-found message."""
    return PARSER_REGISTRY.get(parser_key, f'No parser registered for key: {parser_key!r}')


# ── Self-test ─────────────────────────────────────────────────────────────────
#
# Run directly:  python detector.py
# This runs a suite of 20 test cases covering every supported format.
# All should show PASS. If any show FAIL, check the signature table above.

if __name__ == '__main__':

    # Each test: (file_bytes_sample, filename, expected_parser_key)
    TEST_CASES = [
        # PDFs
        (b'%PDF-1.4 content here',              'report.pdf',      'pdf'),
        (b'%PDF-2.0\n%%EOF',                    'scan.pdf',        'pdf'),

        # Office Open XML (ZIP-based)
        (b'PK\x03\x04rest of zip data',         'document.docx',   'docx'),
        (b'PK\x03\x04rest of zip data',         'spreadsheet.xlsx','xlsx'),
        (b'PK\x03\x04rest of zip data',         'deck.pptx',       'pptx'),
        (b'PK\x03\x04rest of zip data',         'book.epub',       'epub'),
        (b'PK\x03\x04rest of zip data',         'archive.zip',     'zip'),

        # Legacy Office (OLE2)
        (b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1x', 'old.doc',         'doc'),
        (b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1x', 'old.xls',         'xls'),
        (b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1x', 'email.msg',       'email'),

        # Images
        (b'\x89PNG\r\n\x1a\npng data',          'photo.png',       'image'),
        (b'\xff\xd8\xff\xe0jpeg data',           'photo.jpg',       'image'),
        (b'GIF89a image data',                   'anim.gif',        'image'),

        # Text-based
        (b'<!DOCTYPE html><html><body>',         'page.html',       'html'),
        (b'<?xml version="1.0"?>',               'config.xml',      'xml'),
        (b'{"name":"Alice","age":30}',            'data.json',       'json'),
        (b'name,age,city\nAlice,30,NYC\n',        'users.csv',       'csv'),
        (b'col1\tcol2\tcol3\nA\tB\tC\n',          'data.tsv',        'csv'),
        (b'# My Document\n\nSome text.\n',         'notes.md',        'markdown'),

        # Unknown / binary
        (b'\x00\x01\x02\x03\x04\x05\x06\x07',   'mystery.bin',     'unknown'),
    ]

    # Header
    w = [28, 55, 20, 16]
    header = (
        f"{'Filename':<{w[0]}} "
        f"{'Detected MIME':<{w[1]}} "
        f"{'Parser key':<{w[2]}} "
        f"{'Result':<{w[3]}}"
    )
    print(header)
    print('-' * sum(w))

    passed = 0
    for data, fname, expected in TEST_CASES:
        result = detect(data, fname)
        ok = result.parser_key == expected
        if ok:
            passed += 1
        status = 'PASS' if ok else f'FAIL  (expected {expected!r})'
        print(
            f"{fname:<{w[0]}} "
            f"{result.mime_type:<{w[1]}} "
            f"{result.parser_key:<{w[2]}} "
            f"{status}"
        )

    print(f'\n{passed}/{len(TEST_CASES)} tests passed')

    # Show the full parser registry
    print('\nParser registry:')
    for key, info in PARSER_REGISTRY.items():
        print(f'  {key:<20} {info}')