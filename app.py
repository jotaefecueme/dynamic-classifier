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

sheet_url = os.getenv("SHEET_URL")
creds_base64 = os.getenv("CREDS")
groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")
model_provider = os.getenv("MODEL_PROVIDER")
temperature = float(os.getenv("MODEL_TEMPERATURE", 0.0))

for var in ["SHEET_URL", "CREDS", "GROQ_API_KEY", "MODEL_NAME", "MODEL_PROVIDER", "MODEL_TEMPERATURE"]:
    if not os.getenv(var):
        raise RuntimeError(f"Falta la variable de entorno: {var}")

creds_dict = json.loads(base64.b64decode(creds_base64).decode("utf-8"))
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
sheet = client.open_by_url(sheet_url).sheet1

llm = init_chat_model(
    model_name,
    model_provider=model_provider,
    temperature=temperature,
    api_key=groq_api_key
).with_structured_output(
    type("Classification", (BaseModel,), {
        "__annotations__": {
            "intents": list,
            "entities": dict,
            "explanation": str,
            "language": str
        }
    })
)

class ClassificationRequest(BaseModel):
    user_input: str = Field(..., description="Text for classification")
    intents: dict = Field(..., description="Possible intents with descriptions")
    entities: dict = Field(..., description="Possible entities with descriptions")

class ClassificationOutput(BaseModel):
    intents: list
    entities: dict
    explanation: str
    language: str

async def classify_input(user_input: str, intents: dict, entities: dict) -> tuple[dict, float]:
    prompt = (
        "Extract the desired information from the following passage.\n"
        "Possible intents:\n" +
        "\n".join(f"- {k}: {v}" for k, v in intents.items()) +
        "\nPossible entities:\n" +
        "\n".join(f"- {k}: {v}" for k, v in entities.items()) +
        "\nUser input:\n" + user_input
    )
    start = time.time()
    try:
        response = llm.invoke(prompt)
        data = response.model_dump() if hasattr(response, "model_dump") else response
        data["intents"] = data.get("intents") or []
        data["entities"] = data.get("entities") or {}
        data["explanation"] = data.get("explanation", "")
        data["language"] = data.get("language", "")
        ClassificationOutput(**data)
    except Exception as e:
        logging.error("Error invoking LLM", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing the input with the model.")
    elapsed = time.time() - start
    return data, elapsed

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
        logging.warning("Failed to log to Google Sheets", exc_info=True)

@app.post("/classify", response_model=dict)
async def classify_via_api(req: ClassificationRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    data, latency = await classify_input(req.user_input, req.intents, req.entities)
    asyncio.create_task(log_to_gsheet(ip, req, data, latency))
    return {"result": data, "response_time": f"{latency:.2f}s"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
