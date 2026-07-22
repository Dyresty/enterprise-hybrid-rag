from fastapi import APIRouter

from app.services.container import rag_service


router = APIRouter()


@router.post("/chat")
async def chat(
    query: str
):

    response = (
        rag_service
        .generate_answer(query)
    )

    return response