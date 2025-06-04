from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from datetime import datetime
import time
import os
import asyncio
import psutil
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"[{datetime.utcnow().isoformat()}] TOTAL {request.method} {request.url.path} - {duration:.3f}s")
    return response

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER")
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.0"))

if not GROQ_API_KEY:
    raise RuntimeError("The environment variable 'GROQ_API_KEY' is required.")

class ClassificationRequest(BaseModel):
    user_input: str
    intents: dict
    entities: dict

class Classification(BaseModel):
    intents: list
    entities: dict
    language: str

llm = init_chat_model(
    MODEL_NAME,
    model_provider=MODEL_PROVIDER,
    temperature=TEMPERATURE,
    api_key=GROQ_API_KEY
).with_structured_output(Classification)

async def classify_input(user_input: str, intents: dict, entities: dict):
    t0 = time.time()
    prompt = (
        "Extract the desired information from the following passage.\n"
        "Use the following list of possible intents for classification:\n"
        + "\n".join(f"- {k}: {v}" for k, v in intents.items()) + "\n"
        "Use the following list of possible entities to detect:\n"
        + "\n".join(f"- {k}: {v}" for k, v in entities.items()) + "\n"
        f"User input:\n{user_input}"
    )
    t1 = time.time()

    try:
        t2 = time.time()
        result = await asyncio.to_thread(lambda: llm.invoke(prompt))
        t3 = time.time()
        output = result.model_dump()
        t4 = time.time()
    except Exception:
        raise HTTPException(status_code=500, detail="Error processing the input with the model.")

    latency_total = t4 - t0
    time_prompt_prep = t1 - t0
    time_model_call = t3 - t2
    time_output_process = t4 - t3

    mem = psutil.Process().memory_info().rss / 1024**2

    print(f"[{datetime.utcnow().isoformat()}] "
          f"TOTAL_INFER {latency_total:.3f}s | "
          f"PROMPT_PREP {time_prompt_prep:.3f}s | "
          f"MODEL_CALL {time_model_call:.3f}s | "
          f"OUTPUT_PROC {time_output_process:.3f}s | "
          f"MEM {mem:.2f} MB")

    return output, latency_total

@app.post("/classify")
async def classify(req: ClassificationRequest):
    result, latency = await classify_input(req.user_input, req.intents, req.entities)
    return {"result": result, "response_time": f"{latency:.2f} seconds"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
