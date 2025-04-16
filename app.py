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

load_dotenv()

app = FastAPI()

sheet_url = os.getenv("SHEET_URL")
creds_base64 = os.getenv("CREDS")  
groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")
model_provider = os.getenv("MODEL_PROVIDER")
temperature = float(os.getenv("MODEL_TEMPERATURE"))

if not sheet_url or not creds_base64 or not groq_api_key:
    raise ValueError("The environment variables 'SHEET_URL', 'GOOGLE_CREDS_BASE64', and 'GROQ_API_KEY' are required.")

# Decodificar las credenciales desde base64 y guardarlas como un archivo temporal
creds_json = base64.b64decode(creds_base64).decode('utf-8')
with open("google_creds.json", "w") as f:
    f.write(creds_json)

# Google Sheets configuration
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("google_creds.json", scope)
client = gspread.authorize(creds)
sheet = client.open_by_url(sheet_url).sheet1

# Models
class ClassificationRequest(BaseModel):
    user_input: str = Field(..., description="Text provided by the user for classification.")
    intents: dict = Field(..., description="Dictionary of possible intents with their descriptions.")
    entities: dict = Field(..., description="Dictionary of possible entities with their descriptions.")

class Classification(BaseModel):
    intents: list = Field(..., description="List of intents detected in the user's input.")
    entities: dict = Field(..., description="Dictionary of extracted entities and their values. Only include entities mentioned in the input.")
    explanation: str = Field(..., description="Explanation of how the intents and entities were identified.")
    language: str = Field(..., description="Language code (ISO 639-1) of the input, e.g., 'en' or 'es'.")

# Classification function
def classify_input(user_input: str, intents: dict, entities: dict):
    intents_with_desc = "\n".join(f"- {intent}: {desc}" for intent, desc in intents.items())
    entities_with_desc = "\n".join(f"- {entity}: {desc}" for entity, desc in entities.items())

    prompt = ChatPromptTemplate.from_template(
        f"""
        Extract the desired information from the following passage.
        Use the following list of possible intents for classification:
        {intents_with_desc}
        Use the following list of possible entities to detect:
        {entities_with_desc}
        User input:
        {user_input}
        """
    )

    llm = init_chat_model(model_name, model_provider=model_provider, temperature=temperature, api_key=groq_api_key).with_structured_output(Classification)

    start = time.time()
    try:
        response = llm.invoke(prompt.format(user_input=user_input))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing the input with the model.")
    end = time.time()

    return response.model_dump(), end - start

# Function to log classification results to Google Sheets
def log_to_gsheet(ip: str, req: ClassificationRequest, result: dict, response_time: float):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_of_day = now.strftime("%H:%M:%S")

    input_text = req.user_input
    input_intent = json.dumps(req.intents, ensure_ascii=False)
    input_entity = json.dumps(req.entities, ensure_ascii=False)

    response_intent = json.dumps(result.get("intents", []), ensure_ascii=False)
    response_entity = json.dumps(result.get("entities", {}), ensure_ascii=False)
    response_explanation = result.get("explanation", "")
    response_language = result.get("language", "")

    row = [
        ip, date, time_of_day, input_text, input_intent, input_entity,
        response_intent, response_entity, response_explanation, response_language,
        f"{response_time:.2f}"
    ]
    sheet.append_row(row)

# API endpoint to classify input text
@app.post("/classify", response_model=dict)
async def classify_via_api(req: ClassificationRequest, request: Request):
    ip = request.client.host
    try:
        result, response_time = classify_input(req.user_input, req.intents, req.entities)
        log_to_gsheet(ip, req, result, response_time)

        # Eliminar el archivo de credenciales temporal después de usarlo
        if os.path.exists("google_creds.json"):
            os.remove("google_creds.json")

        return {"result": result, "response_time": f"{response_time:.2f} seconds"}
    except HTTPException as e:
        raise e
    except Exception as e:
        # Para evitar exponer información sensible del error
        error_details = "Internal server error"
        return HTTPException(status_code=500, detail=error_details)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
