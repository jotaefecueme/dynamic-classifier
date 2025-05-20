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
import logging
import asyncio

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI()

# Cargar configuración
sheet_url = os.getenv("SHEET_URL")
creds_base64 = os.getenv("CREDS")  
groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")
model_provider = os.getenv("MODEL_PROVIDER")
temperature = float(os.getenv("MODEL_TEMPERATURE", "0.0"))

if not sheet_url or not creds_base64 or not groq_api_key:
    raise RuntimeError("Faltan variables de entorno obligatorias: SHEET_URL, CREDS, GROQ_API_KEY")

data = json.loads(base64.b64decode(creds_base64))
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(data, scope)
client = gspread.authorize(creds)
sheet = client.open_by_url(sheet_url).sheet1

tpl_llm = init_chat_model(model_name, model_provider=model_provider, temperature=temperature, api_key=groq_api_key)
llm = tpl_llm.with_structured_output(
    BaseModel.construct(
        __fields__={
            'intents': Field(...),
            'entities': Field(...),
            'explanation': Field(...),
            'language': Field(...)
        }
    )
)

class ClassificationRequest(BaseModel):
    user_input: str = Field(..., description="Texto para clasificación.")
    intents: dict = Field(..., description="Intents posibles con descripciones.")
    entities: dict = Field(..., description="Entidades posibles con descripciones.")

class ClassificationResponse(BaseModel):
    result: dict
    response_time: str

async def classify_input(user_input: str, intents: dict, entities: dict):
    intents_desc = '\n'.join(f"- {k}: {v}" for k, v in intents.items())
    entities_desc = '\n'.join(f"- {k}: {v}" for k, v in entities.items())
    prompt = (
        "Extract the desired information from the following passage.\n"
        f"Use the following list of possible intents for classification:\n{intents_desc}\n"
        f"Use the following list of possible entities to detect:\n{entities_desc}\n"
        f"User input:\n{user_input}"
    )
    start = time.time()
    try:
        response = llm.invoke(prompt)
        data = response.model_dump()
    except Exception as e:
        logging.error(f"Error procesando modelo: {e}")
        raise HTTPException(status_code=500, detail="Error processing the input with the model.")
    latency = time.time() - start
    return data, latency

async def log_to_gsheet(ip: str, req: ClassificationRequest, data: dict, latency: float):
    now = datetime.utcnow()
    row = [
        ip,
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        req.user_input,
        json.dumps(req.intents, ensure_ascii=False),
        json.dumps(req.entities, ensure_ascii=False),
        json.dumps(data.get("intents", []), ensure_ascii=False),
        json.dumps(data.get("entities", {}), ensure_ascii=False),
        data.get("explanation", ""),
        data.get("language", ""),
        f"{latency:.2f}",
        model_name,
        model_provider,
        temperature
    ]
    try:
        await asyncio.to_thread(sheet.append_row, row)
    except Exception as e:
        logging.warning(f"Error logging to Google Sheets: {e}")

@app.post("/classify", response_model=ClassificationResponse)
async def classify_via_api(req: ClassificationRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    data, latency = await classify_input(req.user_input, req.intents, req.entities)
    asyncio.create_task(log_to_gsheet(ip, req, data, latency))
    return {"result": data, "response_time": f"{latency:.2f} seconds"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
