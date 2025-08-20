# Prompt Engineer AI (UZ/RU) — VS Code project

## Quickstart
1. Unzip this folder and open it in VS Code (`File > Open Folder`).
2. Create a virtual env and activate it:
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux:**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
3. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
4. Pull the model (terminal outside Python venv is fine):
   ```bash
   ollama pull gpt-oss:20b
   ```
5. (Optional) If using remote Ollama, set `.env`:
   ```
   OLLAMA_HOST=http://YOUR_SERVER_IP:11434
   ```
6. Run examples:
   ```bash
   python prompt_engineer_ai.py "rasm chiz: qizil mashina" --mode image
   python prompt_engineer_ai.py "сделай промпт для телеграм-бота" --mode chatgpt
   ```

## Notes
- Output language follows input (UZ or RU).
- Modes: `coding`, `image`, `video`, `chatgpt`.
- Tweak model/params in `config.yaml`.
