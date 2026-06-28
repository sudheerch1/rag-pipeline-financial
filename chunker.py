"""
Chunking strategies for RAG ingestion.

Three strategies, selectable per document type:

  1. recursive   — General purpose. Tries to split on paragraphs → sentences →
                   words. Best for prose (PDF text, DOCX, HTML, Markdown).

  2. semantic    — Groups consecutive blocks by meaning similarity using
                   Cohere Embed v4 (no local ML libraries needed).
                   Splits when cosine similarity between adjacent blocks drops
                   below similarity_threshold. Best for long narrative documents.

  3. table_aware — Keeps table blocks intact (never splits mid-table).
                   Applies recursive to text blocks. Best for mixed docs.

All strategies respect:
  - MAX_TOKENS hard ceiling (no chunk ever exceeds this)
  - OVERLAP_TOKENS to preserve context across chunk boundaries
  - Section / page metadata inheritance
"""

import re
import json
import boto3
import os
import time       # ← add this
from typing import List, Optional

from models import ParsedBlock, Chunk, BlockType
from llm_clients import embed_query, embed_documents


# ── Token counting ────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    words = len(text.split())
    return int(words * 1.3)


# ── Config ────────────────────────────────────────────────────────────────────

MAX_TOKENS     = 448  # 448 content + 64 overlap = 512 max final chunk size
OVERLAP_TOKENS = 64
MIN_TOKENS     = 30

SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


# ── Bedrock client for semantic chunking ──────────────────────────────────────

def get_embedding_bedrock(text: str, bedrock_client=None) -> List[float]:
    """
    Get embedding via Cohere Embed v4 (search_query input_type).
    bedrock_client param kept for backward compatibility but unused.
    Truncates to 512 chars for speed during chunking.
    """
    return embed_query(text[:512])


def get_embedding_with_retry(text: str, bedrock_client=None, max_retries: int = 5) -> List[float]:
    """
    Retry wrapper for Bedrock embeddings with exponential backoff.
    Retries up to max_retries times before raising.
    On final failure: raises → triggers chunk_semantic() fallback to recursive.
    """
    for attempt in range(max_retries):
        try:
            return get_embedding_bedrock(text, bedrock_client)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[semantic chunker] Embedding attempt {attempt + 1} failed, "
                  f"retrying in {wait}s: {e}")
            time.sleep(wait)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk(
    text:        str,
    index:       int,
    block:       ParsedBlock,
    start_block: int,
    end_block:   int,
) -> Optional[Chunk]:
    text = text.strip()
    tokens = count_tokens(text)
    # Table rows are exempt from the prose minimum — a short row like
    # "Operating income $67,656 $60,508 $62,683" is a complete, meaningful
    # retrieval unit even under MIN_TOKENS, unlike a truncated prose fragment.
    if tokens < MIN_TOKENS and block.block_type != BlockType.TABLE:
        return None    

    return Chunk(
        text        = text,
        chunk_index = index,
        token_count = tokens,
        source_key  = block.source_key,
        source_fmt  = block.source_fmt,
        page        = block.page,
        section     = block.section,
        block_type  = block.block_type,
        timestamp   = block.timestamp,
        start_block = start_block,
        end_block   = end_block,
    )


def _split_text_recursive(text: str, max_tokens: int, separators: List[str]) -> List[str]:
    if count_tokens(text) <= max_tokens:
        return [text]

    for sep in separators:
        if sep == "":
            pieces = []
            while text:
                lo, hi = 0, len(text)
                while lo < hi - 1:
                    mid = (lo + hi) // 2
                    if count_tokens(text[:mid]) <= max_tokens:
                        lo = mid
                    else:
                        hi = mid
                pieces.append(text[:lo])
                text = text[lo:]
            return pieces

        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if count_tokens(candidate) <= max_tokens:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    remaining_seps = separators[separators.index(sep) + 1:]
                    if count_tokens(part) > max_tokens and remaining_seps:
                        chunks.extend(_split_text_recursive(part, max_tokens, remaining_seps))
                    else:
                        current = part
            if current:
                chunks.append(current)
            return chunks

    return [text]


def _add_overlap(pieces: List[str], overlap_tokens: int) -> List[str]:
    if len(pieces) <= 1 or overlap_tokens == 0:
        return pieces

    result = [pieces[0]]
    for i in range(1, len(pieces)):
        prev_words = pieces[i - 1].split()
        overlap_words = []
        token_budget = overlap_tokens
        for w in reversed(prev_words):
            if token_budget <= 0:
                break
            overlap_words.insert(0, w)
            token_budget -= 1

        overlap_text = " ".join(overlap_words)
        result.append(overlap_text + " " + pieces[i] if overlap_text else pieces[i])

    return result


# ── Strategy 1: Recursive ─────────────────────────────────────────────────────

def chunk_recursive(
    blocks:        List[ParsedBlock],
    max_tokens:    int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_index = 0

    buffer_text  = ""
    buffer_start = 0
    buffer_block: Optional[ParsedBlock] = None

    def flush(end_idx: int):
        nonlocal buffer_text, buffer_start, buffer_block, chunk_index
        if not buffer_text.strip() or not buffer_block:
            return
        pieces = _split_text_recursive(buffer_text, max_tokens, SEPARATORS)
        pieces = _add_overlap(pieces, overlap_tokens)
        for piece in pieces:
            c = _make_chunk(piece, chunk_index, buffer_block, buffer_start, end_idx)
            if c:
                chunks.append(c)
                chunk_index += 1
        buffer_text  = ""
        buffer_start = end_idx + 1
        buffer_block = None

    for i, block in enumerate(blocks):
        if buffer_block and block.section != buffer_block.section:
            flush(i - 1)

        if block.block_type == BlockType.TABLE:
            flush(i - 1)
            pieces = _split_text_recursive(block.text, max_tokens, ["\n"])
            pieces = _add_overlap(pieces, overlap_tokens)
            for piece in pieces:
                c = _make_chunk(piece, chunk_index, block, i, i)
                if c:
                    chunks.append(c)
                    chunk_index += 1
            continue

        candidate = buffer_text + "\n\n" + block.text if buffer_text else block.text
        if count_tokens(candidate) > max_tokens and buffer_text:
            flush(i - 1)
            buffer_text  = block.text
            buffer_start = i
            buffer_block = block
        else:
            buffer_text  = candidate
            buffer_block = buffer_block or block

    flush(len(blocks) - 1)
    return chunks


# ── Strategy 2: Semantic (Bedrock-powered) ────────────────────────────────────

def chunk_semantic(
    blocks:               List[ParsedBlock],
    max_tokens:           int = MAX_TOKENS,
    overlap_tokens:       int = OVERLAP_TOKENS,
    similarity_threshold: float = 0.75,
) -> List[Chunk]:
    """
    Semantic chunking using Cohere Embed v4.

    How it works:
      1. Embed each block using Bedrock Titan (same model as ingestion)
      2. Compute cosine similarity between adjacent blocks
      3. Split when similarity drops below threshold (topic change detected)
      4. Apply recursive splitting if any group exceeds max_tokens

    Advantages over fixed-size chunking:
      → Keeps related sentences together
      → Splits at natural topic boundaries
      → Better context preservation
      → More relevant chunks retrieved

    Falls back to recursive chunking if Bedrock is unavailable.
    """
    if not blocks:
        return []

    print(f"[semantic chunker] Embedding {len(blocks)} blocks via Bedrock...")

    try:
        # Embed all blocks using Bedrock Titan
        BATCH_SIZE = 90  # Cohere v4 max is 96, stay safely under
        all_texts = [block.text[:512] for block in blocks]
        embeddings = []
        for i in range(0, len(all_texts), BATCH_SIZE):
            batch = all_texts[i:i + BATCH_SIZE]
            print(f"[semantic chunker] Embedding batch {i//BATCH_SIZE + 1} ({len(batch)} blocks)...")
            batch_embeddings = embed_documents(batch)
            embeddings.extend(batch_embeddings)
            if i + BATCH_SIZE < len(all_texts):
                time.sleep(1.0)  # only pause between batches, not between individual calls

        chunks: List[Chunk] = []
        chunk_index = 0
        group_start = 0
        group_blocks = [blocks[0]]

        for i in range(1, len(blocks)):
            sim = cosine_similarity(embeddings[i - 1], embeddings[i])
            group_text = "\n\n".join(b.text for b in group_blocks + [blocks[i]])
            over_budget = count_tokens(group_text) > max_tokens
            section_break = blocks[i].section != blocks[i - 1].section

            # Split if: low similarity (topic change) OR too large OR section break
            if sim < similarity_threshold or over_budget or section_break:
                text = "\n\n".join(b.text for b in group_blocks)
                pieces = _split_text_recursive(text, max_tokens, SEPARATORS)
                pieces = _add_overlap(pieces, overlap_tokens)
                for piece in pieces:
                    c = _make_chunk(piece, chunk_index, group_blocks[0], group_start, i - 1)
                    if c:
                        chunks.append(c)
                        chunk_index += 1
                group_start  = i
                group_blocks = [blocks[i]]
            else:
                group_blocks.append(blocks[i])

        # Flush final group
        text = "\n\n".join(b.text for b in group_blocks)
        pieces = _split_text_recursive(text, max_tokens, SEPARATORS)
        pieces = _add_overlap(pieces, overlap_tokens)
        for piece in pieces:
            c = _make_chunk(piece, chunk_index, group_blocks[0], group_start, len(blocks) - 1)
            if c:
                chunks.append(c)
                chunk_index += 1

        print(f"[semantic chunker] Created {len(chunks)} semantic chunks from {len(blocks)} blocks")
        return chunks

    except Exception as e:
        print(f"[semantic chunker] Failed ({e}), falling back to recursive chunking")
        return chunk_recursive(blocks, max_tokens, overlap_tokens)


# ── Strategy 3: Table-aware ───────────────────────────────────────────────────

def chunk_table_aware(
    blocks:        List[ParsedBlock],
    max_tokens:    int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[Chunk]:
    text_blocks  = [(i, b) for i, b in enumerate(blocks) if b.block_type != BlockType.TABLE]
    table_blocks = [(i, b) for i, b in enumerate(blocks) if b.block_type == BlockType.TABLE]

    result_indexed: List[tuple] = []
    chunk_index = 0

    text_chunks = chunk_recursive([b for _, b in text_blocks], max_tokens, overlap_tokens)
    for c in text_chunks:
        result_indexed.append((text_blocks[c.start_block][0], c))
        c.chunk_index = chunk_index
        chunk_index += 1

    for orig_i, block in table_blocks:
        if count_tokens(block.text) <= max_tokens:
            c = _make_chunk(block.text, chunk_index, block, orig_i, orig_i)
            if c:
                result_indexed.append((orig_i, c))
                chunk_index += 1
        else:
            rows = block.text.split("\n")
            batch: List[str] = []
            for row in rows:
                candidate = "\n".join(batch + [row])
                if count_tokens(candidate) > max_tokens and batch:
                    text = "\n".join(batch)
                    c = _make_chunk(text, chunk_index, block, orig_i, orig_i)
                    if c:
                        result_indexed.append((orig_i, c))
                        chunk_index += 1
                    batch = [row]
                else:
                    batch.append(row)
            if batch:
                text = "\n".join(batch)
                c = _make_chunk(text, chunk_index, block, orig_i, orig_i)
                if c:
                    result_indexed.append((orig_i, c))
                    chunk_index += 1

    result_indexed.sort(key=lambda x: x[0])
    final = []
    for new_i, (_, c) in enumerate(result_indexed):
        c.chunk_index = new_i
        final.append(c)

    return final


# ── Strategy selector ─────────────────────────────────────────────────────────

# Maps parser_key → best chunking strategy
# semantic = uses Bedrock embeddings to detect topic boundaries
STRATEGY_MAP = {
    "pdf":      "table_aware",     # financial PDFs: table rows must stay atomic,
                                 # not split/merged by semantic similarity grouping
    "docx":     "semantic",     # ← upgraded from table_aware to semantic
    "xlsx":     "table_aware",  # tables stay table_aware
    "csv":      "table_aware",
    "pptx":     "recursive",
    "html":     "semantic",     # ← upgraded to semantic
    "json":     "recursive",
    "text":     "semantic",     # ← upgraded to semantic
    "markdown": "semantic",     # ← upgraded to semantic
    "email":    "recursive",
}


def chunk(
    blocks:     List[ParsedBlock],
    fmt:        str,
    strategy:   Optional[str] = None,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[Chunk]:
    """
    Main entry point. Picks a strategy automatically from fmt.
    PDF, DOCX, HTML, text, markdown now use semantic chunking by default.
    Falls back to recursive if Bedrock is unavailable.
    """
    if not blocks:
        return []

    chosen = strategy or STRATEGY_MAP.get(fmt, "recursive")

    if chosen == "semantic":
        return chunk_semantic(blocks, max_tokens, overlap_tokens)
    elif chosen == "table_aware":
        return chunk_table_aware(blocks, max_tokens, overlap_tokens)
    else:
        return chunk_recursive(blocks, max_tokens, overlap_tokens)