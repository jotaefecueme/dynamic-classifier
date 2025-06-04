import os
import time
import asyncio
import psutil
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z"
)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "groq")
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.0"))

if not GROQ_API_KEY:
    raise RuntimeError("The environment variable 'GROQ_API_KEY' is required.")

app = FastAPI()

@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    logger.info(f"{request.method} {request.url.path} - {duration:.3f}s - {response.status_code}")
    return response

class ClassificationRequest(BaseModel):
    user_input: str = Field(..., description="Text provided by the user for classification.")
    intents: dict = Field(..., description="Dictionary of possible intents with their descriptions.")
    entities: dict = Field(..., description="Dictionary of possible entities with their descriptions.")

class Classification(BaseModel):
    intents: list = Field(..., description="List of intents detected in the user's input.")
    entities: dict = Field(..., description="Dictionary of extracted entities and their values.")
    language: str = Field(..., description="Language code (ISO 639-1) of the input, e.g., 'en' or 'es'.")

llm = init_chat_model(
    MODEL_NAME,
    model_provider=MODEL_PROVIDER,
    temperature=TEMPERATURE,
    api_key=GROQ_API_KEY
).with_structured_output(Classification)

class Profiler:
    def __init__(self):
        self.times = {}
        self._start = None
        self._current_label = None

    def start(self, label: str):
        if self._current_label is not None:
            raise RuntimeError(f"Profiler is already timing '{self._current_label}'")
        self._current_label = label
        self._start = time.perf_counter()

    def stop(self):
        if self._current_label is None:
            raise RuntimeError("Profiler was not started")
        elapsed = time.perf_counter() - self._start
        self.times[self._current_label] = self.times.get(self._current_label, 0) + elapsed
        self._current_label = None

    def time(self, label: str):
        class TimerContext:
            def __init__(self, profiler, label):
                self.profiler = profiler
                self.label = label
            def __enter__(self):
                self.profiler.start(self.label)
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.profiler.stop()
        return TimerContext(self, label)

    def results(self):
        return dict(self.times)

    def total(self):
        return sum(self.times.values())

def log_inference_profiling(profiler: Profiler):
    mem = psutil.Process().memory_info().rss / 1024**2
    stats = profiler.results()
    t_total = profiler.total()
    t_prompt = stats.get("prompt_prep", 0)
    t_model = stats.get("model_call", 0)
    t_output = stats.get("output_proc", 0)

    logger.info(
        f"TOTAL_INFER {t_total:.3f}s | PROMPT_PREP {t_prompt:.3f}s | "
        f"MODEL_CALL {t_model:.3f}s | OUTPUT_PROC {t_output:.3f}s | MEM {mem:.2f} MB"
    )
    return t_total

async def classify_input(user_input: str, intents: dict, entities: dict):
    profiler = Profiler()

    with profiler.time("prompt_prep"):
        prompt = (
            "Extract the desired information from the following passage.\n"
            "Use the following list of possible intents for classification:\n"
            + "\n".join(f"- {k}: {v}" for k, v in intents.items()) + "\n"
            "Use the following list of possible entities to detect:\n"
            + "\n".join(f"- {k}: {v}" for k, v in entities.items()) + "\n"
            f"User input:\n{user_input}"
        )

    try:
        with profiler.time("model_call"):
            result = await asyncio.to_thread(lambda: llm.invoke(prompt))
        with profiler.time("output_proc"):
            output = result.model_dump()
    except Exception:
        logger.exception("Error invoking the model")
        raise HTTPException(status_code=500, detail="Error processing the input with the model.")

    latency_total = log_inference_profiling(profiler)
    return output, latency_total

@app.post("/classify")
async def classify(req: ClassificationRequest):
    result, latency = await classify_input(req.user_input, req.intents, req.entities)
    return {"result": result, "response_time": f"{latency:.2f} seconds"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
