from fastapi import APIRouter, HTTPException

from app.models.document import FeedbackRequest
from app.services.container import get_container

router = APIRouter()


@router.post("/feedback")
async def feedback(request: FeedbackRequest):
    """A user rates an answer (thumbs up/down); down-rated answers enter the review queue."""
    try:
        get_container().feedback.add_feedback(request.answer_id, request.rating, request.comment)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown answer_id")
    return {"status": "recorded"}


@router.get("/review")
async def review_queue():
    """Answers a curator should check: flagged by automatic validation or rated down."""
    return get_container().feedback.review_queue()
