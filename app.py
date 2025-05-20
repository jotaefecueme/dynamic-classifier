```python
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

# Carga de configuración
REQUIRED_VARS = ["SHEET_URL", "CREDS", "GROQ_API_KEY", "MODEL_NAME", "MODEL_PROVIDER", "MODEL_TEMPERATURE"]
for var in REQUIRED_VARS:
    if not os.getenv(var):
        raise RuntimeError(f"Falta la variable de entorno: {var}")

sheet_url = os.getenv("SHEET_URL")
creds_base64 = os.getenv("CREDS")
groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")
model_provider = os.getenv("MODEL_PROVIDER")
temperature = float(os.getenv("MODEL_TEMPERATURE"))

# Inicialización de Google Sheets
data = json.loads(base64.b64decode(creds_base64).decode('utf-8'))
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(data, scope)
client = gspread.authorize(creds)
sheet = client.open_by_url(sheet_url).sheet1

# Modelos de datos
class ClassificationRequest(BaseModel):
    user_input: str = Field(..., description="Texto para clasificación")
    intents: dict = Field(..., description="Intents posibles con descripciones")
    entities: dict = Field(..., description="Entidades posibles con descripciones")

class ClassificationOutput(BaseModel):
    intents: list = Field(default_factory=list)
    entities: dict = Field(default_factory=dict)
    explanation: str = ""
    language: str = ""

# Inicialización del LLM solo una vez
tmp_llm = init_chat_model(
    model_name,
    model_provider=model_provider,
    temperature=temperature,
    api_key=groq_api_key
)

async def classify_input(user_input: str, intents: dict, entities: dict) -> tuple[dict, float]:
    """Invoca el LLM y sanea la respuesta garantizando tipos correctos."""
    # Construir prompt simple
    prompt = (
        "Extract the desired information from the following passage.\n"
        "Possible intents:\n" + "\n".join(f"- {k}: {v}" for k, v in intents.items()) +
        "\nPossible entities:\n" + "\n".join(f"- {k}: {v}" for k, v in entities.items()) +
        "\nUser input:\n" + user_input
    )
    start = time.time()
    try:
        # Llamada al LLM
        response = tmp_llm.invoke(prompt)
        # Obtener JSON raw
        raw = response.content if hasattr(response, 'content') else response
        data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        logging.error("Error invoking LLM", exc_info=True)
        raise HTTPException(status_code=500, detail="Error procesando el modelo")
    finally:
        elapsed = time.time() - start

    # Saneamiento robusto
    intents_out = data.get("intents")
    entities_out = data.get("entities")
    if not isinstance(intents_out, list):
        logging.warning("LLM returned non-list for intents, coercing to empty list")
        intents_out = []
    if not isinstance(entities_out, dict):
        logging.warning("LLM returned non-dict for entities, coercing to empty dict")
        entities_out = {}
    explanation = data.get("explanation", "")
    language = data.get("language", "")

    result = {
        "intents": intents_out,
        "entities": entities_out,
        "explanation": explanation,
        "language": language
    }
    # Validar estructura
    try:
        ClassificationOutput(**result)
    except Exception as e:
        logging.error("Validation error on output schema", exc_info=True)
        # Forzar valores mínimos
        result = ClassificationOutput().dict()

    return result, elapsed

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
        await asyncio.to_thread(sheet.append_row, row)
    except Exception:
        logging.warning("Failed to log to Google Sheets", exc_info=True)

@app.post("/classify", response_model=dict)
async def classify_via_api(req: ClassificationRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    result, latency = await classify_input(req.user_input, req.intents, req.entities)
    # Logging en background
    asyncio.create_task(log_to_gsheet(ip, req, result, latency))
    return {"result": result, "response_time": f"{latency:.2f}s"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
