from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class GenerateRequest(BaseModel):
    context: str
    question: str


class GenerateResponse(BaseModel):
    output: str


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    # Here, you can implement logic to generate text from the LLM based on request.context and request.question.
    # For demonstration purposes, let's assume a simple concatenation of context and question.
    generated_text = f"SELECT first_name FROM actor"
    return GenerateResponse(output=generated_text)
