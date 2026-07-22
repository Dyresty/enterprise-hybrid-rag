class ParentStore:
    def __init__(self):
        self.parents = {}

    def add(
        self,
        parent_id,
        text
    ):
        self.parents[parent_id] = text

    def get(
        self,
        parent_id
    ):
        return self.parents.get(
            parent_id,
            ""
        )