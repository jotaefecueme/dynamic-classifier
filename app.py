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
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI()

REQUIRED = ["SHEET_URL", "CREDS", "GROQ_API_KEY", "MODEL_NAME", "MODEL_PROVIDER", "MODEL_TEMPERATURE"]
for var in REQUIRED:
    if not os.getenv(var):
        raise RuntimeError(f"Falta la variable de entorno: {var}")

def init_sheet():
    data = json.loads(base64.b64decode(os.getenv("CREDS")).decode())
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(data, scope)
    return gspread.authorize(creds).open_by_url(os.getenv("SHEET_URL")).sheet1
sheet = init_sheet()

class ClassificationRequest(BaseModel):
    user_input: str = Field(..., description="Texto para clasificación")
    intents: dict = Field(..., description="Intents posibles con descripciones")
    entities: dict = Field(..., description="Entidades posibles con descripciones")

class Classification(BaseModel):
    intents: list = Field(..., description="Intents detectados en la entrada")
    entities: dict = Field(..., description="Entidades extraídas y sus valores")
    explanation: str = Field(..., description="Explicación de la clasificación")
    language: str = Field(..., description="Código de idioma ISO 639-1")

model_name = os.getenv("MODEL_NAME")
model_provider = os.getenv("MODEL_PROVIDER")
temperature = float(os.getenv("MODEL_TEMPERATURE"))
llm = init_chat_model(model_name, model_provider=model_provider, temperature=temperature, api_key=os.getenv("GROQ_API_KEY"))
llm_typed = llm.with_structured_output(Classification)

executor = ThreadPoolExecutor(max_workers=2)

async def classify_input(user_input: str, intents: dict, entities: dict):
    prompt = (
        "Extract the desired information from the following passage.\n"
        "Use the following list of possible intents for classification:\n" +
        "\n".join(f"- {k}: {v}" for k, v in intents.items()) +
        "\nUse the following list of possible entities to detect:\n" +
        "\n".join(f"- {k}: {v}" for k, v in entities.items()) +
        "\nUser input:\n" + user_input
    )
    start = time.time()
    try:
        response = llm_typed.invoke(prompt)
    except Exception as e:
        logging.error(f"LLM invocation error: {e}")
        raise HTTPException(status_code=500, detail="Error procesando el modelo")
    return response.model_dump(), time.time() - start

def log_to_sheet(row):
    try:
        sheet.append_row(row)
    except Exception as e:
        logging.warning(f"Error registrando en Google Sheets: {e}")

@app.post("/classify", response_model=dict)
async def classify_via_api(req: ClassificationRequest, request: Request):
    result, latency = await classify_input(req.user_input, req.intents, req.entities)
    ip = request.client.host
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
        f"{latency:.2f}",
        model_name,
        model_provider,
        temperature
    ]
    executor.submit(log_to_sheet, row)
    return {"result": result, "response_time": f"{latency:.2f}s"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
