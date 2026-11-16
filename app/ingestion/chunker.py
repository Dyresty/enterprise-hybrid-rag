import uuid
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.document import ChunkMetadata, DocumentChunk

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Parent-child chunking: small children give precise retrieval,
# their larger parent gives the LLM enough surrounding context.
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400, separators=SEPARATORS)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, separators=SEPARATORS)


def chunk_documents(pages, document_id, filename, source_type="pdf", title=None, url=None):
    """pages: list of {"page": int, "text": str}. Returns child chunks as dicts, each carrying its parent text."""
    children = []
    chunk_id = 0
    today = datetime.now().strftime("%Y-%m-%d")

    for page in pages:
        for parent_text in parent_splitter.split_text(page["text"]):
            parent_id = str(uuid.uuid4())
            for child_text in child_splitter.split_text(parent_text):
                children.append(
                    DocumentChunk(
                        text=child_text,
                        parent_text=parent_text,
                        metadata=ChunkMetadata(
                            document_id=document_id,
                            filename=filename,
                            page=page["page"],
                            chunk_id=chunk_id,
                            parent_id=parent_id,
                            created_at=today,
                            source_type=source_type,
                            title=title,
                            url=url,
                        ),
                    ).model_dump()
                )
                chunk_id += 1
    return children
