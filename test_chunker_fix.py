from parsers import parse_pdf
from chunker import chunk
from models import BlockType

PDF_PATH = r"C:\Tech\PROD-Grade-POC\AgenticAI\Data-into-RAG\aapl-20240928.pdf"  # same path as before

with open(PDF_PATH, "rb") as f:
    file_bytes = f.read()

blocks = parse_pdf(file_bytes)

# Page 65 specifically — the segment table
page_65_blocks = [b for b in blocks if b.page == 65]

print(f"Page 65 blocks going into chunker: {len(page_65_blocks)}\n")

chunks = chunk(page_65_blocks, fmt="pdf")

print(f"Total chunks produced: {len(chunks)}\n")

for c in chunks:
    print(f"[chunk {c.chunk_index}] {len(c.text)} chars")
    print(f"    {c.text[:150]}")
    print()

# Specific check: does "Rest of Asia Pacific" survive intact as ONE chunk,
# not split mid-row or merged with unrelated rows?
for c in chunks:
    if "Rest of Asia Pacific" in c.text:
        print("=" * 60)
        print("FOUND chunk containing 'Rest of Asia Pacific':")
        print(repr(c.text))
        print(f"Row appears intact (has the $30,658 figure): {'30,658' in c.text}")