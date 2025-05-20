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
from typing import Dict, Any, Tuple

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI()

REQUIRED = ["SHEET_URL", "CREDS", "GROQ_API_KEY", "MODEL_NAME", "MODEL_PROVIDER", "MODEL_TEMPERATURE"]

for var in REQUIRED:
    val = os.getenv(var)
    if val is None or val.strip() == "":
        raise RuntimeError(f"Falta la variable de entorno obligatoria: {var}")

def init_sheet():
    try:
        creds_b64 = os.getenv("CREDS")
        creds_json = base64.b64decode(creds_b64).decode()
        data = json.loads(creds_json)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(data, scope)
        gc = gspread.authorize(creds)
        sheet = gc.open_by_url(os.getenv("SHEET_URL")).sheet1
        logging.info("Google Sheets inicializado correctamente")
        return sheet
    except Exception as e:
        logging.error(f"Error inicializando Google Sheets: {e}")
        return None

sheet = init_sheet()

class ClassificationRequest(BaseModel):
    user_input: str = Field(..., description="Texto para clasificación")
    intents: Dict[str, str] = Field(..., description="Intents posibles con descripciones")
    entities: Dict[str, str] = Field(..., description="Entidades posibles con descripciones")

class Classification(BaseModel):
    intents: list = Field(..., description="Intents detectados en la entrada")
    entities: Dict[str, Any] = Field(..., description="Entidades extraídas y sus valores")
    explanation: str = Field(..., description="Explicación de la clasificación")
    language: str = Field(..., description="Código de idioma ISO 639-1")

model_name = os.getenv("MODEL_NAME")
model_provider = os.getenv("MODEL_PROVIDER")
try:
    temperature = float(os.getenv("MODEL_TEMPERATURE"))
except ValueError:
    raise RuntimeError("La variable MODEL_TEMPERATURE debe ser un número válido")

llm = init_chat_model(model_name, model_provider=model_provider, temperature=temperature, api_key=os.getenv("GROQ_API_KEY"))
llm_typed = llm.with_structured_output(Classification)

executor = ThreadPoolExecutor(max_workers=2)

async def classify_input(user_input: str, intents: Dict[str, str], entities: Dict[str, str]) -> Tuple[Dict[str, Any], float]:
    """
    Invoca el modelo para clasificar la entrada del usuario.
    Devuelve la respuesta y el tiempo de latencia.
    """
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
        # Dependiendo de si invoke es async o no, se adapta:
        if callable(getattr(llm_typed, "ainvoke", None)):
            response_raw = await llm_typed.ainvoke(prompt)
        else:
            response_raw = llm_typed.invoke(prompt)
        # response_raw puede ser dict o Pydantic model
        output = response_raw if isinstance(response_raw, dict) else response_raw.dict()
        # Sanear campos
        if not isinstance(output.get("intents", []), list):
            output["intents"] = []
        if not isinstance(output.get("entities", {}), dict):
            output["entities"] = {}
        if "explanation" not in output or not isinstance(output["explanation"], str):
            output["explanation"] = ""
        if "language" not in output or not isinstance(output["language"], str):
            output["language"] = ""
        response = Classification(**output)
    except Exception as e:
        logging.error(f"Error en invocación del modelo: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error procesando el modelo")
    latency = time.time() - start
    return response.model_dump(), latency

def log_to_sheet(row: list):
    """
    Registra una fila en Google Sheets.
    Maneja excepciones para evitar fallos críticos.
    """
    if sheet is None:
        logging.warning("No se puede registrar en Google Sheets: hoja no inicializada")
        return
    try:
        sheet.append_row(row)
    except Exception as e:
        logging.warning(f"Error registrando en Google Sheets: {e}")

@app.post("/classify", response_model=dict)
async def classify_via_api(req: ClassificationRequest, request: Request):
    result, latency = await classify_input(req.user_input, req.intents, req.entities)
    ip = request.client.host if request.client else "unknown"
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
        executor.submit(log_to_sheet, row)
    except Exception as e:
        logging.warning(f"Error al enviar tarea de logging a executor: {e}")
    return {"result": result, "response_time": f"{latency:.2f}s"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
