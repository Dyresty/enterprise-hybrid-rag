class ContextBuilder:
    def build_context(
        self,
        documents
    ):
        context = ""
        for doc in documents:
            context += f"""
Source:
{doc['filename']}
Page:
{doc['page']}
Content:
{doc['text']}

-------------------
"""
        return context