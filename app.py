from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import os
import json
import base64
import time

load_dotenv()

app = FastAPI()

# === CARGA DE VARIABLES DE ENTORNO ===
sheet_url = os.getenv("SHEET_URL")
creds_base64 = os.getenv("CREDS")
groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")
model_provider = os.getenv("MODEL_PROVIDER")
temperature = float(os.getenv("MODEL_TEMPERATURE"))

if not sheet_url or not creds_base64 or not groq_api_key:
    raise ValueError("Las variables de entorno necesarias no están definidas.")

# === CONFIGURACIÓN DE GOOGLE SHEETS UNA VEZ ===
creds_json = base64.b64decode(creds_base64).decode('utf-8')
creds_dict = json.loads(creds_json)
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
sheet = client.open_by_url(sheet_url).sheet1

# === CONFIGURACIÓN DEL MODELO UNA VEZ ===
llm = init_chat_model(model_name, model_provider=model_provider, temperature=temperature, api_key=groq_api_key).with_structured_output("Classification")

# === DEFINICIONES DE MODELOS Pydantic ===
class ClassificationRequest(BaseModel):
    user_input: str = Field(..., description="Texto del usuario.")
    intents: dict = Field(..., description="Intenciones posibles.")
    entities: dict = Field(..., description="Entidades posibles.")

class Classification(BaseModel):
    intents: list = Field(..., description="Intenciones detectadas.")
    entities: dict = Field(..., description="Entidades extraídas.")
    explanation: str = Field(..., description="Explicación del análisis.")
    language: str = Field(..., description="Código de idioma ISO 639-1.")

# === PLANTILLA DE PROMPT ===
PROMPT_TEMPLATE = """
Extract the desired information from the following passage.
Use the following list of possible intents for classification:
{intents}
Use the following list of possible entities to detect:
{entities}
User input:
{user_input}
"""
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# === CLASIFICADOR ===
def classify_input(user_input: str, intents: dict, entities: dict):
    intents_with_desc = "\n".join(f"- {intent}: {desc}" for intent, desc in intents.items())
    entities_with_desc = "\n".join(f"- {entity}: {desc}" for entity, desc in entities.items())

    prompt = prompt_template.format(
        user_input=user_input,
        intents=intents_with_desc,
        entities=entities_with_desc
    )

    start = time.time()
    try:
        response = llm.invoke(prompt)
    except Exception:
        raise HTTPException(status_code=500, detail="Error procesando la entrada con el modelo.")
    end = time.time()

    return response.model_dump(), end - start

# === LOG A GOOGLE SHEETS ===
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
        temperature
    ]
    sheet.append_row(row)

# === RUTA PRINCIPAL ===
@app.post("/classify", response_model=dict)
async def classify_via_api(req: ClassificationRequest, request: Request):
    ip = request.client.host
    try:
        result, response_time = classify_input(req.user_input, req.intents, req.entities)
        log_to_gsheet(ip, req, result, response_time)
        return {"result": result, "response_time": f"{response_time:.2f} seconds"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
