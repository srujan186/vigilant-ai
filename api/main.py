"""
SmartGuard - FastAPI Gateway
POST /classify  → classify a prompt
POST /chat      → classify + forward safe prompts to Groq LLM
GET  /health    → health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os, sys, time, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
from classifier.guard import SmartGuard
from classifier.output_guard import OutputGuard    

# ── Groq client ───────────────────────────────────────────────
from groq import Groq

app = FastAPI(title="Vigilant AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton guard (loaded once at startup)
guard: SmartGuard = None
output_guard: OutputGuard = None
groq_client: Groq = None

LOG_PATH = Path(__file__).parent / "results" / "api_log.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup():
    global guard,output_guard, groq_client
    guard = SmartGuard(threshold=float(os.getenv("GUARD_THRESHOLD", "0.5")))
    output_guard = OutputGuard(strict_mode=False)
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("✅ Vigilant AI API ready (inut+output guard   )")


# ── Request / Response models ─────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str
    threshold: Optional[float] = None

class ChatRequest(BaseModel):
    prompt: str
    threshold: Optional[float] = None
    system_prompt: Optional[str] = "You are a helpful assistant."

class ClassifyResponse(BaseModel):
    verdict: str
    category: str
    confidence: float
    latency_ms: float

class ChatResponse(BaseModel):
    verdict: str
    category: str
    confidence: float
    guard_latency_ms: float
    llm_response: Optional[str] = None
    llm_latency_ms: Optional[float] = None
    blocked: bool
    blocked_by: Optional[str] = None
    output_check: Optional[str] = None

# ── Helpers ───────────────────────────────────────────────────

def log_event(event: dict):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "threshold": guard.threshold}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if req.threshold is not None:
        guard.set_threshold(req.threshold)
    result = guard.classify(req.text)
    log_event({
        "type": "classify",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt_snippet": req.text[:100],
        "verdict": result.verdict,
        "category": result.category,
        "confidence": result.confidence,
        "latency_ms": result.latency_ms
    })
    return ClassifyResponse(
        verdict=result.verdict,
        category=result.category,
        confidence=result.confidence,
        latency_ms=result.latency_ms
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1. Guard check
    if req.threshold is not None:
        guard.set_threshold(req.threshold)
    result = guard.classify(req.prompt)

    blocked = result.verdict == "unsafe"
    llm_response = None
    llm_latency = None

    # 2. Only forward to LLM if safe
    if not blocked:
        t0 = time.perf_counter()
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": req.system_prompt},
                    {"role": "user", "content": req.prompt}
                ],
                max_tokens=512,
            )
            llm_response = completion.choices[0].message.content
            llm_latency = round((time.perf_counter() - t0) * 1000, 2)
        except Exception as e:
            llm_response = f"[LLM error: {str(e)}]"

        # 3. Output guard
        output_result = output_guard.validate(llm_response, req.prompt)
        if output_result.verdict == "unsafe":
            blocked = True
            llm_response = f" Output blocked — {output_result.check_failed} detected in LLM response"
            output_check_failed = output_result.check_failed
        else:
            output_check_failed = None
    else:
        output_check_failed = None

    log_event({
        "type": "chat",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt_snippet": req.prompt[:100],
        "verdict": result.verdict,
        "category": result.category,
        "confidence": result.confidence,
        "guard_latency_ms": result.latency_ms,
        "blocked": blocked,
        "llm_latency_ms": llm_latency
    })

    return ChatResponse(
        verdict=result.verdict,
        category=result.category,
        confidence=result.confidence,
        guard_latency_ms=result.latency_ms,
        llm_response=llm_response if not blocked else (
            f"🚫 Input blocked — {result.category} detected (confidence: {result.confidence})"
            if result.verdict == "unsafe"
            else llm_response
        ),
        llm_latency_ms=llm_latency,
        blocked=blocked,
        blocked_by="input_guard" if result.verdict == "unsafe"
                   else ("output_guard" if output_check_failed else None),
        output_check=output_check_failed
    )


@app.post("/threshold")
def set_threshold(value: float):
    if not 0.0 <= value <= 1.0:
        raise HTTPException(400, "Threshold must be between 0.0 and 1.0")
    guard.set_threshold(value)
    return {"threshold": guard.threshold}
