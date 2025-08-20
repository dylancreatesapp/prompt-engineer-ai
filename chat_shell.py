import os, sys, json, yaml, requests
from datetime import datetime
from typing import List, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from prompt_engineer_ai import build_refiner, SUPPORTED_MODES

SYSTEM_PROMPT = (
  "You are a helpful, concise assistant. "
  "Answer in the user's language. "
  "Do NOT repeat the user's message. "
  "Provide a direct, useful answer."
)

HELP = """Commands:
  /mode raw                - normal assistant chat (default, uses gpt-oss:20b)
  /mode engineer [m]       - prompt-engineering (m in {coding,image,video,chatgpt}) on gpt-oss:20b
  /reset                   - clear memory
  /save [path]             - save transcript to JSONL
  /help                    - show commands
  /exit                    - quit
"""

def _load_cfg():
  try:
    with open("config.yaml","r",encoding="utf-8") as f:
      return yaml.safe_load(f) or {}
  except Exception:
    return {}

def _prewarm(host: str, model: str, keep_alive: str = "30m"):
  """Keep the big model resident in RAM so first token is fast."""
  try:
    url = (host or "http://localhost:11434").rstrip("/") + "/api/generate"
    requests.post(url, json={"model": model, "prompt": " ", "stream": False, "keep_alive": keep_alive}, timeout=2)
  except Exception:
    pass

class ChatShell:
  def __init__(self):
    cfg = _load_cfg()
    oll = (cfg.get("ollama") or {})
    self.model = oll.get("model","gpt-oss:20b")
    self.host  = os.getenv("OLLAMA_HOST")  # None -> http://localhost:11434
    self.num_ctx = int(oll.get("num_ctx", 4096))
    self.keep_alive = str(oll.get("keep_alive","30m"))

    # RAW chat client (20B)
    self.chat = ChatOllama(
      model=self.model,
      base_url=self.host,
      temperature=float(oll.get("temperature",0.4)),
      top_p=float(oll.get("top_p",0.9)),
      num_ctx=self.num_ctx,
      num_predict=512,
    )

    # Prompt-engineer pipeline (also forced to 20B)
    self.refiner = build_refiner()

    self.history: List = [SystemMessage(content=SYSTEM_PROMPT)]
    self.engineer_mode = False
    self.engineer_submode: Literal["coding", "image", "video", "chatgpt"] = "chatgpt"

    # Prewarm the big model so the first token doesn’t “hang”
    _prewarm(self.host or "http://localhost:11434", self.model, self.keep_alive)

  def run(self):
    print(f"[model: {self.model} | num_ctx: {self.num_ctx} | host: {self.host or 'http://localhost:11434'}]")
    print(HELP)
    while True:
      try:
        user = input("you › ").strip()
        if not user:
          continue
        if user.startswith("/"):
          if self._handle_cmd(user):  # handled command
            continue
          else:
            break

        if self.engineer_mode:
          # Always engineer with 20B
          text = self.refiner.refine(user, mode=self.engineer_submode,
                                     model=self.model, num_ctx=self.num_ctx, stream=False)
          print("\n" + text + "\n")
          self.history.append(HumanMessage(content=user))
          self.history.append(AIMessage(content=text))
        else:
          # RAW chat on 20B
          msgs = self.history + [HumanMessage(content=user)]
          print("assistant › ", end="", flush=True)
          chunks = []
          try:
            for chunk in self.chat.stream(msgs):
              t = getattr(chunk, "content", str(chunk))
              chunks.append(t)
              sys.stdout.write(t)
              sys.stdout.flush()
            print()
            self.history.append(HumanMessage(content=user))
            self.history.append(AIMessage(content="".join(chunks)))
          except Exception as e:
            print(f"\n[chat error] {e}")
      except (KeyboardInterrupt, EOFError):
        print("\nbye!")
        break

  def _handle_cmd(self, s: str) -> bool:
    parts = s.split()
    cmd = parts[0].lower()

    if cmd == "/help":
      print(HELP); return True

    if cmd == "/reset":
      self.history = [SystemMessage(content=SYSTEM_PROMPT)]
      print("[memory cleared]"); return True

    if cmd == "/mode":
      if len(parts) < 2:
        print("usage: /mode raw | /mode engineer [submode]"); return True
      if parts[1] == "raw":
        self.engineer_mode = False
        print("[mode=raw: gpt-oss:20b]"); return True
      if parts[1] == "engineer":
        self.engineer_mode = True
        if len(parts) > 2 and parts[2] in SUPPORTED_MODES:
          self.engineer_submode = parts[2]
        print(f"[mode=engineer:{self.engineer_submode} on gpt-oss:20b]"); return True
      print("unknown mode"); return True

    if cmd == "/save":
      path = parts[1] if len(parts) > 1 else None
      if not path:
        os.makedirs("data/logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"data/logs/chat_{ts}.jsonl"
      with open(path, "w", encoding="utf-8") as f:
        for m in self.history:
          role = "system" if isinstance(m, SystemMessage) else ("user" if isinstance(m, HumanMessage) else "assistant")
          f.write(json.dumps({"role": role, "content": m.content}, ensure_ascii=False) + "\n")
      print(f"[saved → {path}]"); return True

    if cmd == "/exit":
      return False

    print("unknown command; /help"); return True

if __name__ == "__main__":
  ChatShell().run()
