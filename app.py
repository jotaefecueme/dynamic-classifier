from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from datetime import datetime
import time
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import json
import base64
from concurrent.futures import ThreadPoolExecutor
import asyncio

load_dotenv()

app = FastAPI()

sheet_url = os.getenv("SHEET_URL")
creds_base64 = os.getenv("CREDS")  
groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")
model_provider = os.getenv("MODEL_PROVIDER")
temperature = float(os.getenv("MODEL_TEMPERATURE", "0.0"))

if not sheet_url or not creds_base64 or not groq_api_key:
    raise ValueError("The environment variables 'SHEET_URL', 'CREDS', and 'GROQ_API_KEY' are required.")

creds_dict = json.loads(base64.b64decode(creds_base64).decode('utf-8'))
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
sheet = client.open_by_url(sheet_url).sheet1

llm = init_chat_model(
    model_name,
    model_provider=model_provider,
    temperature=temperature,
    api_key=groq_api_key
).with_structured_output(BaseModel)

db_executor = ThreadPoolExecutor(max_workers=2)

class ClassificationRequest(BaseModel):
    user_input: str = Field(..., description="Text provided by the user for classification.")
    intents: dict = Field(..., description="Dictionary of possible intents with their descriptions.")
    entities: dict = Field(..., description="Dictionary of possible entities with their descriptions.")

class Classification(BaseModel):
    intents: list = Field(..., description="List of intents detected in the user's input.")
    entities: dict = Field(..., description="Dictionary of extracted entities and their values.")
    explanation: str = Field(..., description="Explanation of how the intents and entities were identified.")
    language: str = Field(..., description="Language code (ISO 639-1) of the input, e.g., 'en' or 'es'.")

async def classify_input(user_input: str, intents: dict, entities: dict):
    # Construir prompt en string simple
    intents_desc = "\n".join(f"- {k}: {v}" for k, v in intents.items())
    entities_desc = "\n".join(f"- {k}: {v}" for k, v in entities.items())
    prompt = f"""
Extract the desired information from the following passage.
Use the following list of possible intents for classification:
{intents_desc}
Use the following list of possible entities to detect:
{entities_desc}
User input:
{user_input}
"""

    start = time.time()
    try:
        result = await asyncio.to_thread(lambda: llm.invoke(prompt))
        output = result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing the input with the model.")
    latency = time.time() - start
    return output, latency

async def log_to_gsheet(ip: str, req: ClassificationRequest, result: dict, response_time: float):
    now = datetime.utcnow()
    row = [
        ip,
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        req.user_input,
        json.dumps(req.intents, ensure_ascii=False),
        json.dumps(req.entities, ensure_ascii=False),
        json.dumps(result.get("intents", []), ensure_ascii=False),
        json.dumps(result.get("entities", {}), ensure_ascii=False),
        result.get("explanation", ""),
        result.get("language", ""),
        f"{response_time:.2f}",
        model_name,
        model_provider,
        temperature
    ]
    try:
        sheet.append_row(row)
    except Exception:
        pass  

@app.post("/classify", response_model=dict)
async def classify_via_api(req: ClassificationRequest, request: Request):
    ip = request.client.host if request.client else ""
    result, response_time = await classify_input(req.user_input, req.intents, req.entities)
    asyncio.create_task(log_to_gsheet(ip, req, result, response_time))
    return {"result": result, "response_time": f"{response_time:.2f} seconds"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
