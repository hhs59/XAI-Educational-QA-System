from fastapi import FastAPI
from pydantic import BaseModel, Field
from solver import  solve


app = FastAPI()

class PredictRequest(BaseModel):
    question: str
    type: str | None = None
    premises_NL: list[str] | None = Field(default=None, alias='premises-NL')

@app.post('/predict')
def predict(request: PredictRequest):
    return solve(question=request.question,
                query_type=request.type,
                premises_nl=request.premises_NL)
