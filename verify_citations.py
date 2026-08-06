"""
Standalone verification for /chat citations + chunks.

Run:
    source .venv/bin/activate
    python verify_citations.py <rag_id> "your question here"

It bypasses the LLM and HTTP layer and directly checks the two things
that decide whether /chat can return chunks/citations:

  1. Does Chroma retrieval return any chunks for this rag_id?  (root cause #1)
  2. Do the citation/chunk builders produce structured output?  (formatting)

If step 1 prints 0 chunks, the problem is retrieval/binding — NOT the API
contract. Check that session_knowledgebase.knowledgebase_id == rag_store.rag_id
and that documents were indexed under this rag_id.
"""

import sys
import json

from chroma_utils import get_chunks_from_chroma, vectorstore


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python verify_citations.py <rag_id> \"question\"")
        raise SystemExit(2)

    rag_id = sys.argv[1]
    question = sys.argv[2]

    # 0) How many chunks exist in Chroma for this rag_id at all?
    try:
        stored = vectorstore.get(where={"rag_id": rag_id})
        stored_count = len(stored.get("ids", []))
    except Exception as exc:  # noqa: BLE001
        stored_count = -1
        print(f"[warn] could not read vectorstore for rag_id={rag_id}: {exc}")

    print(f"[1] Chunks indexed in Chroma for rag_id={rag_id}: {stored_count}")
    if stored_count == 0:
        print("    -> No documents indexed under this rag_id. "
              "This is why /chat returns empty chunks/citations.")

    # 1) Retrieval for this specific question
    chunks = get_chunks_from_chroma(question, rag_id)
    print(f"[2] Retrieved chunks for question: {len(chunks)}")
    for c in chunks[:3]:
        print(f"    - chunk_id={c.get('chunk_id')} file={c.get('filename')} "
              f"score={c.get('score')} preview={ (c.get('content') or '')[:60]!r}")

    if not chunks:
        print("    -> Retrieval returned 0. Fix binding/indexing before checking the API.")
        return

    # 2) Build the exact structures the /chat endpoint returns
    from main import _build_source_chunks, _build_citations

    source_chunks = _build_source_chunks(chunks)
    fake_answer = "This is a grounded sentence. Here is a second grounded sentence."
    citations = _build_citations(fake_answer, source_chunks)

    print(f"[3] Built source_chunks={len(source_chunks)} citations={len(citations)}")
    print("    sample citation:")
    print("   ", json.dumps(citations[0] if citations else {}, indent=2, default=str))
    print("\n[OK] If [2] and [3] are > 0, /chat WILL return chunks + citations.")


if __name__ == "__main__":
    main()
