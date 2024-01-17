from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    question: str
    context: str


app = FastAPI()
templates = Jinja2Templates(directory="templates")
model_name = "deepset/roberta-base-squad2"
classifier = pipeline('question-answering', model=model_name, tokenizer=model_name)


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(text: str = Form(...), question: str = Form(...)):
    answer = classifier({
        'question': question,
        'context': text,
    })
    return {"text": text, "question": question, "answer": answer['answer']}
