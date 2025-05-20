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

# FastAPI app
tmp_app = FastAPI()
app = tmp_app

# Load environment variables
sheet_url = os.getenv("SHEET_URL")
creds_base64 = os.getenv("CREDS")
groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")
model_provider = os.getenv("MODEL_PROVIDER")
temperature = float(os.getenv("MODEL_TEMPERATURE", "0.0"))

if not sheet_url or not creds_base64 or not groq_api_key:
    raise RuntimeError("Missing required environment variables: SHEET_URL, CREDS, GROQ_API_KEY")

# Initialize Google Sheets client (reuse in memory, no file write)
try:
    creds_dict = json.loads(base64.b64decode(creds_base64))
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    gc = gspread.authorize(ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope))
    sheet = gc.open_by_url(sheet_url).sheet1
    logging.info("Google Sheets client initialized")
except Exception as e:
    logging.error(f"Error initializing Google Sheets: {e}")
    sheet = None

# Define request/response models
class ClassificationRequest(BaseModel):
    user_input: str = Field(..., description="Text provided by the user for classification.")
    intents: dict = Field(..., description="Dictionary of possible intents with their descriptions.")
    entities: dict = Field(..., description="Dictionary of possible entities with their descriptions.")

class ClassificationOutput(BaseModel):
    intents: list = Field(..., description="List of intents detected in the user's input.")
    entities: dict = Field(..., description="Dictionary of extracted entities and their values.")
    explanation: str = Field(..., description="Explanation of how the intents and entities were identified.")
    language: str = Field(..., description="Language code (ISO 639-1) of the input.")

try:
    llm = init_chat_model(model_name, model_provider=model_provider, temperature=temperature, api_key=groq_api_key)
    llm = llm.with_structured_output(ClassificationOutput)
    logging.info("LLM initialized")
except Exception as e:
    logging.error(f"Error initializing LLM: {e}")
    raise RuntimeError("LLM initialization failed")

async def classify_input(user_input: str, intents: dict, entities: dict):
    prompt = (
        "Extract the desired information from the following passage.\n"
        "Possible intents:\n" + "\n".join(f"- {k}: {v}" for k, v in intents.items()) + "\n"
        "Possible entities:\n" + "\n".join(f"- {k}: {v}" for k, v in entities.items()) + "\n"
        "User input:\n" + user_input
    )
    start = time.time()
    try:
        response = llm.invoke(prompt)
        output = response.model_dump() if hasattr(response, 'model_dump') else response
        if not isinstance(output.get("entities"), dict):
            output["entities"] = {}
        if not isinstance(output.get("intents"), list):
            output["intents"] = []
        output.setdefault("explanation", "")
        output.setdefault("language", "")
        result = ClassificationOutput(**output)
    except Exception as e:
        logging.error(f"LLM invocation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing the input with the model.")
    latency = time.time() - start
    return result.dict(), latency

async def log_to_sheet(ip: str, req: ClassificationRequest, result: dict, latency: float):
    if not sheet:
        logging.warning("Sheet client not initialized, skipping log")
        return
    now = datetime.utcnow()
    row = [
        ip,
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        req.user_input,
        json.dumps(req.intents, ensure_ascii=False),
        json.dumps(req.entities, ensure_ascii=False),
        json.dumps(result.get("intents"), ensure_ascii=False),
        json.dumps(result.get("entities"), ensure_ascii=False),
        result.get("explanation"),
        result.get("language"),
        f"{latency:.2f}",
        model_name,
        model_provider,
        temperature
    ]
    try:
        await asyncio.to_thread(sheet.append_row, row)
    except Exception as e:
        logging.warning(f"Failed to log to sheet: {e}")

@app.post("/classify", response_model=dict)
async def classify_via_api(req: ClassificationRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    result, latency = await classify_input(req.user_input, req.intents, req.entities)
a    asyncio.create_task(log_to_sheet(ip, req, result, latency))
    return {"result": result, "response_time": f"{latency:.2f}s"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
