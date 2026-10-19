from rag import SalonRAG
from pathlib import Path

# Initialize RAG once (lazy loading)
_rag_instance = None

def _get_rag():
    global _rag_instance
    if _rag_instance is None:
        from pathlib import Path
        import sys
        # Get business name from sys.argv or default
        business = sys.argv[1] if len(sys.argv) > 1 else "salon"
        faq_path = Path(__file__).parent / "businesses" / business / "faq.md"
        _rag_instance = SalonRAG(faq_path)
    return _rag_instance


def search_faq(question: str) -> str:
    """Search the business FAQ for answers to common questions."""
    try:
        rag = _get_rag()
        result = rag.query(question)
        if not result.strip():
            return "NO_MATCH: no FAQ entry found. Answer from your own knowledge or hand off."
        return result
    except Exception as e:
        return f"ERROR: RAG failed ({e}). Answer from your own knowledge."