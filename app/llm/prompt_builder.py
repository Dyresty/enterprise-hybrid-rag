def create_rag_prompt(
    question,
    context
):
    prompt=f"""
You are an enterprise document assistant.

Answer the question using only the provided context.

If the answer is not present, say:
"I don't know based on the provided documents."

Always cite the source page when possible.

Context:

{context}


Question:

{question}


Answer:
"""
    return prompt