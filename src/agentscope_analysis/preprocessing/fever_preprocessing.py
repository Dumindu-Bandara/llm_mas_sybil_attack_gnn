"""
extract_fever_evidence.py

Resolve FEVER claim evidence (Wikipedia_URL + Sentence_ID pairs) to the
exact evidence sentence text(s), using the wiki-pages evidence corpus.

Usage:
    python extract_fever_evidence.py \
        --wiki-dir   /path/to/wiki-pages \
        --claims     /path/to/shared_task_dev.jsonl \
        --out        /path/to/dev_with_evidence.jsonl \
        --index-db   /path/to/wiki_index.sqlite   # built once, reused after

Output: one JSON object per input claim, e.g.

    {
      "id": 137334,
      "claim": "Fox 2000 Pictures released the film Soul Food.",
      "label": "SUPPORTS",
      "evidence_sets": [[["Soul_Food_-LRB-film-RRB-", 0]]],
      "evidence_text": ["Soul Food is a 1997 American drama film ..."]
    }

`evidence_sets` / `evidence_text` are parallel lists: one entry per
independently-sufficient evidence set. A NOT ENOUGH INFO claim gets [] for
both, since it has no citable evidence in the FEVER annotations.
"""

import argparse
import glob
import json
import os
import sqlite3


# ---------------------------------------------------------------------------
# 1. Build a page-id -> `lines` index from the wiki-pages dump
# ---------------------------------------------------------------------------

def build_sqlite_index(wiki_dir: str, db_path: str) -> None:
    """Read every wiki-*.jsonl file once and store id -> (text, lines) in SQLite.
    Mirrors the DrQA-style DocDB used in the official FEVER baseline, but keeps
    memory bounded regardless of corpus size."""
    if os.path.exists(db_path):
        print(f"Index already exists at {db_path}, skipping build.")
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, text TEXT, lines TEXT)")

    files = sorted(glob.glob(os.path.join(wiki_dir, "wiki-*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No wiki-*.jsonl files found in {wiki_dir}")

    total = 0
    for fp in files:
        rows = []
        with open(fp, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                doc = json.loads(raw)
                page_id = doc.get("id", "")
                if not page_id:
                    # the empty-id row that starts each file is a metadata
                    # marker, not a real page - skip it
                    continue
                rows.append((page_id, doc.get("text", ""), doc.get("lines", "")))
        c.executemany("INSERT OR REPLACE INTO documents VALUES (?,?,?)", rows)
        total += len(rows)
        print(f"  indexed {os.path.basename(fp)}  (+{len(rows)}, {total} total)")

    conn.commit()
    conn.close()
    print(f"Done. {total} pages indexed at {db_path}")


class WikiIndex:
    """Looks up a page's raw `lines` blob by id."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)

    def get_lines(self, page_id: str) -> str | None:
        cur = self.conn.execute(
            "SELECT lines FROM documents WHERE id = ?", (page_id,)
        )
        row = cur.fetchone()
        return row[0] if row else None


# ---------------------------------------------------------------------------
# 2. Parse one page's `lines` field into {sentence_id: sentence_text}
# ---------------------------------------------------------------------------

def parse_lines(lines_blob: str) -> dict[int, str]:
    """
    Each row of `lines` looks like:
        "<idx>\t<sentence text>\t<mention>\t<link>\t<mention>\t<link>..."
    or, for an empty sentence:
        "<idx>"

    Key strictly off the leading index in each row (not its row position) -
    that index is the authoritative sentence id used in evidence annotations,
    and relying on position breaks silently if any row is blank/malformed.
    """
    sentences: dict[int, str] = {}
    for row in lines_blob.split("\n"):
        if not row:
            continue
        parts = row.split("\t")
        try:
            idx = int(parts[0])
        except ValueError:
            continue  # malformed row, skip
        sentences[idx] = parts[1] if len(parts) > 1 else ""
    return sentences


# ---------------------------------------------------------------------------
# 3. Resolve one claim's evidence to text
# ---------------------------------------------------------------------------

def get_evidence_sets(claim: dict) -> list[list[tuple[str, int]]]:
    """
    FEVER's `evidence` field is a list of evidence *sets* (multiple
    independently-sufficient justifications can exist per claim). Each set
    is a list of [Annotation_ID, Evidence_ID, Wikipedia_URL, Sentence_ID].
    Returns a list of [(page_id, sentence_id), ...] per set, dropping any
    set with no page (i.e. NOT ENOUGH INFO, where page/sentence are null).
    """
    sets = []
    for group in claim.get("evidence", []):
        pairs = [(ev[2], ev[3]) for ev in group if ev[2] is not None]
        if pairs:
            sets.append(pairs)
    return sets


def evidence_text_for_set(
    pairs: list[tuple[str, int]],
    wiki: WikiIndex,
    cache: dict[str, dict[int, str]],
) -> str:
    """Resolve one evidence set to a single string (sentences joined with a space)."""
    out = []
    for page_id, sent_id in pairs:
        if page_id not in cache:
            lines_blob = wiki.get_lines(page_id)
            cache[page_id] = parse_lines(lines_blob) if lines_blob is not None else {}
        sent = cache[page_id].get(sent_id)
        if sent is None:
            # page missing from the dump, or sentence_id not found on the page
            sent = f"[MISSING: {page_id}#{sent_id}]"
        out.append(sent)
    return " ".join(out)


def process_claim(claim: dict, wiki: WikiIndex, cache: dict[str, dict[int, str]]) -> dict:
    evidence_sets = get_evidence_sets(claim)
    evidence_texts = [evidence_text_for_set(s, wiki, cache) for s in evidence_sets]

    return {
        "id": claim.get("id"),
        "claim": claim.get("claim"),
        "label": claim.get("label"),
        "evidence_sets": evidence_sets,   # raw (page_id, sentence_id) pairs, per set
        "evidence_text": evidence_texts,  # resolved sentence text, one string per set
    }


# ---------------------------------------------------------------------------
# 4. Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki-dir", required=True, help="Directory containing wiki-*.jsonl")
    ap.add_argument("--claims", required=True, help="Path to shared_task_dev.jsonl (or train/test)")
    ap.add_argument("--out", required=True, help="Output jsonl path")
    ap.add_argument("--index-db", default="wiki_index.sqlite", help="Where to build/reuse the SQLite index")
    args = ap.parse_args()

    build_sqlite_index(args.wiki_dir, args.index_db)
    wiki = WikiIndex(args.index_db)
    cache: dict[str, dict[int, str]] = {}

    n = 0
    with open(args.claims, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            claim = json.loads(line)
            result = process_claim(claim, wiki, cache)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            n += 1

    print(f"Resolved evidence for {n} claims -> {args.out}")


if __name__ == "__main__":
    main()