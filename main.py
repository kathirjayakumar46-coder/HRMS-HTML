import os
import json
import uuid
import time
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

from models import ProcessRequest
from rag import rag_index
from vertex_client import stream_generate
from utils import validate_and_sanitize_json, build_prompt, extract_json_from_text, Timer
from logger_config import setup_logger

load_dotenv()

logger = setup_logger("HRMS")

app = FastAPI(title="HRMS AI Ticket Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


# ─────────────────────────────
# STARTUP
# ─────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("SYSTEM STARTING")
    with Timer() as t:
        rag_index.build()
    logger.info(f"RAG READY in {t.elapsed:.3f}s")


# ─────────────────────────────
# FRONTEND
# ─────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    with open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────
# MAIN API
# ─────────────────────────────
@app.post("/process")
async def process(req: ProcessRequest):

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[REQ-{request_id}] START")

    start_time = time.perf_counter()

    try:
        clean_json = validate_and_sanitize_json(req.json_data)
        query = req.query.strip()
        if not query:
            raise ValueError("Query cannot be empty")

    except Exception as e:
        raise HTTPException(400, str(e))

    # RAG retrieval
    with Timer() as t:
        rag_docs = rag_index.retrieve(query)
    logger.info(f"[REQ-{request_id}] RAG found {len(rag_docs)} docs")

    prompt = build_prompt(clean_json, query, rag_docs)

    async def stream() -> AsyncGenerator[str, None]:

        loop = asyncio.get_running_loop()
        full_output = []

        def run_llm():
            return stream_generate(prompt)

        try:
            generator = await loop.run_in_executor(None, run_llm)

            for token in generator:
                full_output.append(token)
                yield f"data:{json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0.001)

            raw = "".join(full_output)

            try:
                cleaned = extract_json_from_text(raw)
                parsed = json.loads(cleaned)
                if not isinstance(parsed, list):
                    parsed = [parsed]
            except:
                parsed = [{"message": "There is no information about this."}]

            elapsed = round(time.perf_counter() - start_time, 3)
            logger.info(f"[REQ-{request_id}] DONE {elapsed}s")

            yield f"data:{json.dumps({'done': True, 'result': parsed})}\n\n"

        except Exception as e:
            logger.exception("STREAM ERROR")
            yield f"data:{json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────
# HEALTH
# ─────────────────────────────
@app.get("/health")
def health():
    return {"status": "running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host ="172.17.200.39" , port=8000, reload=True)