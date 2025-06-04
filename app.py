import os
import asyncio
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import asyncpg 

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "groq")
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.0"))
DATABASE_URL = os.getenv("DATABASE_URL")  

if not GROQ_API_KEY:
    raise RuntimeError("The environment variable 'GROQ_API_KEY' is required.")

if not DATABASE_URL:
    raise RuntimeError("The environment variable 'DATABASE_URL' is required.")

app = FastAPI()

class ClassificationRequest(BaseModel):
    user_input: str = Field(..., description="Text provided by the user for classification.")
    intents: dict = Field(..., description="Dictionary of possible intents with their descriptions.")
    entities: dict = Field(..., description="Dictionary of possible entities with their descriptions.")

class Classification(BaseModel):
    intents: list = Field(..., description="List of intents detected in the user's input.")
    entities: dict = Field(..., description="Dictionary of extracted entities and their values.")
    language: str = Field(..., description="Language code (ISO 639-1) of the input, e.g., 'en' or 'es'.")

llm = init_chat_model(
    MODEL_NAME,
    model_provider=MODEL_PROVIDER,
    temperature=TEMPERATURE,
    api_key=GROQ_API_KEY
).with_structured_output(Classification)

pool: asyncpg.Pool = None

@app.on_event("startup")
async def startup():
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL)

@app.on_event("shutdown")
async def shutdown():
    await pool.close()

async def insert_log(
    ip: str,
    date: str,
    hour: str,
    input_text: str,
    input_intents: dict,
    input_entities: dict,
    response_intents: list,
    response_entities: dict,
    response_language: str,
    infer_time: float,
    model_name: str,
    model_provider: str,
    temperature: float,
):
    query = """
    INSERT INTO dynamic-classifier (
        ip, date, hour, input_text, input_intents, input_entities, 
        response_intents, response_entities, response_language, 
        infer_time, model_name, model_provider, temperature
    ) VALUES (
        $1, $2, $3, $4, $5::json, $6::json, $7::json, $8::json, $9, $10, $11, $12, $13
    )
    """
    await pool.execute(
        query,
        ip,
        date,
        hour,
        input_text,
        input_intents,
        input_entities,
        response_intents,
        response_entities,
        response_language,
        infer_time,
        model_name,
        model_provider,
        temperature,
    )

async def classify_input(user_input: str, intents: dict, entities: dict):
    prompt = (
        "Extract the desired information from the following passage.\n"
        "Use the following list of possible intents for classification:\n"
        + "\n".join(f"- {k}: {v}" for k, v in intents.items()) + "\n"
        "Use the following list of possible entities to detect:\n"
        + "\n".join(f"- {k}: {v}" for k, v in entities.items()) + "\n"
        f"User input:\n{user_input}"
    )
    try:
        start = time.perf_counter()
        result = await asyncio.to_thread(lambda: llm.invoke(prompt))
        infer_time = time.perf_counter() - start
        output = result.model_dump()
    except Exception:
        raise HTTPException(status_code=500, detail="Error processing the input with the model.")
    return output, infer_time

@app.post("/classify")
async def classify(req: ClassificationRequest, request: Request):
    result, infer_time = await classify_input(req.user_input, req.intents, req.entities)

    now = datetime.utcnow()
    date_str = now.date().isoformat()  
    hour_str = now.time().strftime("%H:%M:%S")  

    ip = request.client.host

    asyncio.create_task(insert_log(
        ip=ip,
        date=date_str,
        hour=hour_str,
        input_text=req.user_input,
        input_intents=req.intents,
        input_entities=req.entities,
        response_intents=result["intents"],
        response_entities=result["entities"],
        response_language=result["language"],
        infer_time=infer_time,
        model_name=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        temperature=TEMPERATURE,
    ))

    return {"result": result, "infer_time": f"{infer_time:.3f} seconds"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
