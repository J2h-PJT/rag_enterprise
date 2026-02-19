from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    history: str = ""
    file_id: str | None = None

def register_routes(app, container):
    @app.post("/ask")
    def ask(req: QueryRequest):
        filters = None
        if req.file_id:
            filters = {"file_id": req.file_id}

        stream = container["qa_service"].answer(
            req.question,
            history=req.history,
            selected_ids=req.file_id
        )
        return {"answer": "".join(stream)}
