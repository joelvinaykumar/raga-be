from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import tempfile
import logging
import re
from pathlib import Path
load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

def _get_persist_dir() -> str:
    default = os.path.join(os.getcwd(), "chroma_db")
    try:
        os.makedirs(default, exist_ok=True)
        probe = os.path.join(default, ".write_test")
        with open(probe, "w") as f:
            f.write("")
        os.remove(probe)
        return default
    except OSError:
        return os.path.join(tempfile.gettempdir(), "chroma_db")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def _build_vectorstore():
    api_key = os.getenv("CHROMA_API_KEY")
    if api_key:
        return Chroma(
            embedding_function=embedding_function,
            chroma_cloud_api_key=api_key,
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
        )
    return Chroma(
        embedding_function=embedding_function,
        persist_directory=_get_persist_dir(),
    )

vectorstore = _build_vectorstore()


_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return set(_TOKEN_RE.findall(text.lower()))


def _lexical_overlap_score(query: str, content: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    content_tokens = _tokenize(content)
    if not content_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(content_tokens))
    return overlap / len(query_tokens)

def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = BSHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = loader.load()
    return text_splitter.split_documents(documents)

def index_document_to_chroma(file_path: str, file_id: int, rag_id: str | None = None) -> bool:
    try:
        splits = load_and_split_document(file_path)
        filename = Path(file_path).name

        for idx, split in enumerate(splits, start=1):
            split.metadata['file_id'] = file_id
            split.metadata['filename'] = filename
            split.metadata['chunk_index'] = idx
            split.metadata['chunk_id'] = f"{file_id}:{idx}"
            split.metadata['preview'] = split.page_content[:180]
            if rag_id:
                split.metadata['rag_id'] = rag_id

        vectorstore.add_documents(splits)
        logger.info("Indexed %d chunks for file_id=%s (rag_id=%s)", len(splits), file_id, rag_id)
        return True
    except Exception as e:
        logger.error("Error indexing document %s: %s", file_path, e, exc_info=True)
        return False

def get_chunks_from_chroma(
    query: str,
    knowledgebase_id: str | None = None,
    top_k: int = 8,
    min_relevance_score: float = 0.2,
):
    try:
        top_k = max(1, min(int(top_k), 30))
        fetch_k = max(top_k * 3, 12)
        result = []
        filter_dict = {}
        if knowledgebase_id:
            filter_dict["rag_id"] = knowledgebase_id
        results = vectorstore.similarity_search_with_relevance_scores(
            query, k=fetch_k, filter=filter_dict if filter_dict else None,
        )

        seen = set()
        rescored = []
        for res, score in results:
            vector_score = float(score)
            lexical_score = _lexical_overlap_score(query, res.page_content)
            blended_score = (0.8 * vector_score) + (0.2 * lexical_score)
            if blended_score < float(min_relevance_score):
                continue

            chunk_id = res.metadata.get("chunk_id")
            dedupe_key = chunk_id or res.page_content[:180]
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            rescored.append({
                "score": round(blended_score, 6),
                "vector_score": round(vector_score, 6),
                "lexical_score": round(lexical_score, 6),
                "content": res.page_content,
                "source": res.metadata.get("source"),
                "file_id": res.metadata.get("file_id"),
                "rag_id": res.metadata.get("rag_id"),
                "chunk_id": chunk_id,
                "chunk_index": res.metadata.get("chunk_index"),
                "filename": res.metadata.get("filename"),
                "preview": res.metadata.get("preview") or res.page_content[:180],
                "url": res.metadata.get("source"),
            })

        rescored.sort(key=lambda item: item.get("score", 0), reverse=True)
        result.extend(rescored[:top_k])

        logger.info(
            "Retrieved %d/%d chunks for rag_id=%s top_k=%d min_score=%s",
            len(result), len(results), knowledgebase_id, top_k, min_relevance_score,
        )
        return result
    except Exception as e:
        logger.error("Error in getting chunks from chroma: %s", e, exc_info=True)
        return []

def delete_doc_from_chroma(file_id: int):
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
        
        vectorstore._collection.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")
        
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False

def delete_rag_from_chroma(rag_id: str):
    try:
        vectorstore._collection.delete(where={"rag_id": rag_id})
        print(f"Deleted all chunks with rag_id {rag_id} from Chroma")
        return True
    except Exception as e:
        print(f"Error deleting RAG {rag_id} from Chroma: {str(e)}")
        return False