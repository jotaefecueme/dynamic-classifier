from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import time
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import json
import base64
import logging

load_dotenv()

app = FastAPI()

sheet_url = os.getenv("SHEET_URL")
creds_base64 = os.getenv("CREDS")
groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")  
model_provider = os.getenv("MODEL_PROVIDER")  
try:
    temperature = float(os.getenv("MODEL_TEMPERATURE", 0.7))
except ValueError:
    temperature = 0.7

if not sheet_url or not creds_base64 or not groq_api_key:
    raise ValueError("Las variables 'SHEET_URL', 'CREDS' y 'GROQ_API_KEY' son obligatorias.")

try:
    creds_json = base64.b64decode(creds_base64).decode('utf-8')
    creds_dict = json.loads(creds_json) 
    
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url).sheet1
except Exception as e:
    logging.error(f"Error inicializando cliente Google Sheets: {e}")
    raise RuntimeError("No se pudo inicializar Google Sheets.")

llm = init_chat_model(
    model_name,
    model_provider=model_provider,
    temperature=temperature,
    api_key=groq_api_key
).with_structured_output(None) 

class ClassificationRequest(BaseModel):
    user_input: str = Field(..., description="Texto del usuario para clasificación")
    intents: dict = Field(..., description="Diccionario de intents con descripciones")
    entities: dict = Field(..., description="Diccionario de entidades con descripciones")

class Classification(BaseModel):
    intents: list = Field(..., description="Intents detectados")
    entities: dict = Field(..., description="Entidades extraídas")
    explanation: str = Field(..., description="Explicación de la clasificación")
    language: str = Field(..., description="Código idioma ISO 639-1")

def classify_input(user_input: str, intents: dict, entities: dict):
    intents_with_desc = "\n".join(f"- {intent}: {desc}" for intent, desc in intents.items())
    entities_with_desc = "\n".join(f"- {entity}: {desc}" for entity, desc in entities.items())

    prompt_text = (
        "Extract the desired information from the following passage.\n"
        "Use the following list of possible intents for classification:\n"
        f"{intents_with_desc}\n"
        "Use the following list of possible entities to detect:\n"
        f"{entities_with_desc}\n"
        "User input:\n"
        f"{user_input}"
    )

    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    typed_llm = llm.with_structured_output(Classification)

    start = time.time()
    try:
        response = typed_llm.invoke(prompt_text)
    except Exception as e:
        logging.error(f"Error en invoke LLM: {e}")
        raise HTTPException(status_code=500, detail="Error procesando la entrada con el modelo.")
    end = time.time()

    return response.model_dump(), end - start

def log_to_gsheet(ip: str, req: ClassificationRequest, result: dict, response_time: float):
    now = datetime.now()
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
        temperature,
    ]
    try:
        sheet.append_row(row)
    except Exception as e:
        logging.warning(f"Error al loguear en Google Sheets: {e}")

@app.post("/classify", response_model=dict)
async def classify_via_api(req: ClassificationRequest, request: Request):
    ip = request.client.host
    try:
        result, response_time = classify_input(req.user_input, req.intents, req.entities)
        log_to_gsheet(ip, req, result, response_time)
        return {"result": result, "response_time": f"{response_time:.2f} seconds"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error interno: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
