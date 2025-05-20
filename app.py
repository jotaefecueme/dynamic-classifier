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

# Variables de entorno
sheet_url       = os.getenv("SHEET_URL")
creds_base64    = os.getenv("CREDS")
groq_api_key    = os.getenv("GROQ_API_KEY")
model_name      = os.getenv("MODEL_NAME")
model_provider  = os.getenv("MODEL_PROVIDER")
temperature     = float(os.getenv("MODEL_TEMPERATURE", "0.0"))

if not sheet_url or not creds_base64 or not groq_api_key:
    raise ValueError("Faltan SHEET_URL, CREDS o GROQ_API_KEY")

# Inicialización de Google Sheets sin escribir archivos
creds_dict = json.loads(base64.b64decode(creds_base64))
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
sheet = gspread.authorize(creds).open_by_url(sheet_url).sheet1

class ClassificationRequest(BaseModel):
    user_input: str = Field(..., description="Text provided by the user for classification.")
    intents: dict = Field(..., description="Dictionary of possible intents with their descriptions.")
    entities: dict = Field(..., description="Dictionary of possible entities with their descriptions.")

class Classification(BaseModel):
    intents: list = Field(..., description="List of intents detected in the user's input.")
    entities: dict = Field(..., description="Dictionary of extracted entities and their values. Only include entities mentioned in the input.")
    explanation: str = Field(..., description="Explanation of how the intents and entities were identified.")
    language: str = Field(..., description="Language code (ISO 639-1) of the input, e.g., 'en' or 'es'.")

llm = init_chat_model(
    model_name,
    model_provider=model_provider,
    temperature=temperature,
    api_key=groq_api_key
).with_structured_output(Classification)

executor = ThreadPoolExecutor(max_workers=2)

def classify_input(user_input: str, intents: dict, entities: dict):
    intents_desc   = "\n".join(f"- {k}: {v}" for k, v in intents.items())
    entities_desc  = "\n".join(f"- {k}: {v}" for k, v in entities.items())
    prompt = (
        "Extract the desired information from the following passage.\n"
        "Possible intents:\n" + intents_desc + "\n"
        "Possible entities:\n" + entities_desc + "\n"
        "User input:\n" + user_input
    )

    start = time.time()
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        logging.error(f"LLM invocation error: {e}")
        raise HTTPException(status_code=500, detail="Error processing the input with the model.")
    latency = time.time() - start
    return response.model_dump(), latency

def log_to_gsheet(ip: str, req: ClassificationRequest, result: dict, latency: float):
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
    try:
        sheet.append_row(row)
    except Exception as e:
        logging.warning(f"Error logging to Google Sheets: {e}")

@app.post("/classify", response_model=dict)
async def classify_via_api(req: ClassificationRequest, request: Request):
    ip = request.client.host or "unknown"
    result, latency = classify_input(req.user_input, req.intents, req.entities)
    executor.submit(log_to_gsheet, ip, req, result, latency)
    return {
        "result": result,
        "response_time": f"{latency:.2f} seconds"
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
