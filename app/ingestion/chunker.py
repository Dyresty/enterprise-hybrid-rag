import uuid
from datetime import datetime

from app.models.document import (
    DocumentChunk,
    ChunkMetadata
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(
    pages,
    document_id,
    filename
):
    # Parent chunks
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=500,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
    # Child chunks
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
    children = []
    chunk_id = 0
    for page in pages:
        parents = parent_splitter.split_text(
            page["text"]
        )
        for parent_text in parents:
            parent_id = str(
                uuid.uuid4()
            )
            child_chunks = child_splitter.split_text(
                parent_text
            )
            for child_text in child_chunks:
                document_chunk = DocumentChunk(
                    text=child_text,
                    metadata=ChunkMetadata(
                        document_id=document_id,
                        filename=filename,
                        page=page["page"],
                        chunk_id=chunk_id,
                        parent_id=parent_id,
                        section=None,
                        created_at=datetime.now()
                        .strftime("%Y-%m-%d"),
                        source_type="pdf"
                    )
                )
                children.append(
                    document_chunk.model_dump()
                )
                chunk_id += 1
    return children