from fastapi import FastAPI
from pydantic import BaseModel
from solver import  solve
app = FastAPI()

class PredictRequest(BaseModel):
    question: str

@app.post('/predict')
def predict(request: PredictRequest):
    return solve(request.question)
