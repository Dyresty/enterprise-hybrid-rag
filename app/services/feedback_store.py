"""Human-in-the-loop storage: every answer is logged, users rate answers, and flagged or
down-rated answers form a review queue for curators. Reviewed cases can later become
new golden-set questions or fine-tuning examples."""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class FeedbackStore:
    def __init__(self, folder="data/feedback"):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.answers_file = self.folder / "answers.jsonl"
        self.feedback_file = self.folder / "feedback.jsonl"
        self.lock = threading.Lock()

    def _append(self, path, record):
        record = {**record, "timestamp": datetime.now(timezone.utc).isoformat()}
        with self.lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read(self, path):
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def log_answer(self, response):
        self._append(self.answers_file, response)

    def add_feedback(self, answer_id, rating, comment=None):
        if not any(a["answer_id"] == answer_id for a in self._read(self.answers_file)):
            raise KeyError(answer_id)
        self._append(self.feedback_file, {"answer_id": answer_id, "rating": rating, "comment": comment})

    def review_queue(self):
        """Answers flagged by the automatic checks or rated down by a user."""
        down = {f["answer_id"]: f for f in self._read(self.feedback_file) if f["rating"] == "down"}
        queue = []
        for answer in self._read(self.answers_file):
            if answer.get("needs_review") or answer["answer_id"] in down:
                reasons = list(answer.get("review_reasons", []))
                if answer["answer_id"] in down:
                    reasons.append(f"rated down: {down[answer['answer_id']].get('comment') or 'no comment'}")
                queue.append({**answer, "review_reasons": reasons})
        return queue
