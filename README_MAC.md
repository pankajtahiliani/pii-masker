# PII Shield — Mac Setup Guide

## Quick Start (2 minutes)

### Step 1 — Copy to your Mac
Transfer the `pii_masker_mac` folder to your Mac (AirDrop, USB, or download).

### Step 2 — Run the startup script
Open **Terminal** and run:
```bash
cd ~/Downloads/pii_masker_mac      # or wherever you saved it
chmod +x start_mac.sh
./start_mac.sh
```
The script automatically:
- Installs Homebrew (if missing)
- Installs Ollama (if missing)
- Detects Apple Silicon vs Intel and picks the right model
- Downloads the AI model (one-time only)
- Opens the app in your browser

### Step 3 — Use the app
Browser opens at **http://localhost:5000** automatically.

---

## Apple Silicon vs Intel

| | Apple Silicon (M1/M2/M3) | Intel Mac |
|--|--|--|
| Model used | gemma3:4b | gemma3:1b |
| RAM needed | ~2.5 GB | ~600 MB |
| Speed | Very fast (Neural Engine) | Moderate |
| Download size | ~2.5 GB | ~800 MB |

---

## Manual Setup (if script fails)

```bash
# Install Ollama
brew install ollama

# Start Ollama
ollama serve &

# Pull model (Apple Silicon)
ollama pull gemma3:4b

# Pull model (Intel Mac)
ollama pull gemma3:1b

# Install Python deps
pip3 install -r requirements.txt

# Run app
python3 app.py
```

Then open: http://localhost:5000

---

## Stopping the App
Press **Ctrl+C** in Terminal — the script also stops Ollama automatically.
