from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    pages,
    document_id,
    filename
):

    """
    Split extracted PDF pages into smaller chunks.

    Output:
        [
            {
                document_id,
                filename,
                chunk_id,
                page,
                text
            }
        ]
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )


    chunks = []

    chunk_id = 0


    for page in pages:

        page_chunks = splitter.split_text(
            page["text"]
        )


        for chunk in page_chunks:

            chunks.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": chunk
                }
            )

            chunk_id += 1


    return chunks