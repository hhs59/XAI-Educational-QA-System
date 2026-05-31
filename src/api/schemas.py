from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    question: str
    type: str | None = None
    premises_NL: list[str] | None = Field(default=None, alias='premises-NL')
