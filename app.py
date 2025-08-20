from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
import prompt_engineer_ai  # your refinement logic

app = FastAPI(title="Prompt Engineer AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[  # keep specific local origins here if you use them
        "http://localhost:3000",
        "http://127.0.0.1:5500",
    ],
    # 👇 allow every Vercel preview/prod domain (both UI and any previews)
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class PromptRequest(BaseModel):
    text: str
    mode: Literal["chatgpt", "image", "video", "coding"] = "chatgpt"   # options: chatgpt, image, video, coding
    profile: str = "speed"  # options: speed, balanced, max

@app.post("/refine")
async def refine_prompt(request: PromptRequest):
    """
    Refine a raw Uzbek/Russian/English prompt into a structured prompt.
    Defaults to profile="speed" (qwen2.5:7b).
    """
    refined = prompt_engineer_ai.refine(
        text=request.text,
        mode=request.mode,
        profile=request.profile or "speed"
    )
    return {
        "input": request.text,
        "mode": request.mode,
        "profile": request.profile,
        "refined": refined
    }

@app.get("/")
def home():
    """Health check"""
    return {"message": "AI Prompt Engineer is running 🚀"}
