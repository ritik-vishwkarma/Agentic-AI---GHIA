# GHIA - Gramin Health Intake Assistant
## AI-Powered Multi-Agent Healthcare System for Rural India

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple.svg)

### 🎯 Problem Statement
Rural India faces severe healthcare access challenges. This project provides an AI-powered voice-based health intake system that:
- Understands **Hindi/Hinglish** spoken by patients
- Uses **Multi-Agent AI** for intelligent symptom extraction and triage
- Provides **bilingual summaries** for doctors
- Works in **low-resource environments**

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GHIA - Agentic Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Patient Voice ──► ASR (Whisper) ──► ORCHESTRATOR               │
│                                           │                     │
│                    ┌──────────────────────┼──────────────────┐  │
│                    ▼                      ▼                  ▼  │
│           ┌───────────────┐      ┌────────────┐     ┌──────────┐│
│           │   MEDICAL     │      │ INTERROGA- │     │  TRIAGE  ││
│           │  EXTRACTOR    │◄────►│    TOR     │◄───►│  AGENT   ││
│           │    AGENT      │      │   AGENT    │     │          ││
│           └───────────────┘      └────────────┘     └──────────┘│
│                    │                    │                  │    │
│                    └────────────────────┴──────────────────┘    │
│                                         │                       │
│                                         ▼                       │
│                              ┌──────────────────┐               │
│                              │   OUTPUT AGENT   │               │
│                              │  (Bilingual)     │               │
│                              └──────────────────┘               │
│                                         │                       │
│                    ┌────────────────────┴────────────────┐      │
│                    ▼                                     ▼      │
│            Doctor Dashboard                      Patient TTS    │
└─────────────────────────────────────────────────────────────────┘
```

### 🚀 Quick Start

#### 1. Get a Free Groq API Key
Go to [console.groq.com](https://console.groq.com/) and get a free API key.

#### 2. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Configure Environment
```bash
# Copy example env file
copy .env.example .env

# Edit .env and add your Groq API key
# GROQ_API_KEY=your_actual_key_here
```

#### 4. Run the Backend
```bash
cd d:\code\antigravity\MH_agents
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 5. Run the Doctor Dashboard
```bash
# In a new terminal
streamlit run dashboard/app.py
```

#### 6. Open the Patient Frontend
Open `frontend/index.html` in a browser.

### 📁 Project Structure
```
MH_agents/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Configuration settings
│   ├── agents/
│   │   ├── orchestrator.py  # LangGraph multi-agent system
│   ├── services/
│   │   ├── asr.py           # Whisper ASR service
│   │   └── medical_kb.py    # Hinglish-to-Medical mapping
│   ├── routes/
│   │   ├── intake.py        # Audio/text intake API
│   │   └── dashboard.py     # Doctor dashboard API
│   ├── db/
│   │   └── database.py      # SQLite database
│   └── schemas/
│       └── intake.py        # Pydantic models
├── dashboard/
│   └── app.py               # Streamlit doctor UI
├── frontend/
│   └── index.html           # Patient voice input UI
├── requirements.txt
└── README.md
```

### 🤖 Multi-Agent System

The system uses **LangGraph** for multi-agent orchestration:

| Agent | Role | Agentic Behavior |
|-------|------|------------------|
| **Medical Extractor** | Extracts symptoms, duration, severity | Uses LLM for intelligent extraction |
| **Interrogator** | Generates follow-up questions | Autonomously identifies missing info |
| **Triage Agent** | Classifies urgency (urgent/moderate/routine) | Makes autonomous decisions |
| **Output Agent** | Generates bilingual summaries | Creates doctor-ready reports |

### 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/intake/audio` | POST | Upload audio for processing |
| `/api/intake/text` | POST | Submit text for processing |
| `/api/intake/demo` | POST | Run demo with sample input |
| `/api/dashboard/intakes` | GET | List all intakes |
| `/api/dashboard/stats` | GET | Get summary statistics |

### 📊 Demo

Test the system with sample Hindi input:
```bash
curl -X POST "http://localhost:8000/api/intake/demo"
```

Or via text:
```bash
curl -X POST "http://localhost:8000/api/intake/text" \
  -F "text=मुझे कमर में बहुत दर्द है तीन दिन से" \
  -F "language=hi"
```

### 🛠️ Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Agents**: LangGraph + LangChain
- **LLM**: Groq (Llama 3.1 70B - FREE & Fast)
- **ASR**: Faster-Whisper (GPU accelerated)
- **Database**: SQLite
- **Dashboard**: Streamlit
- **Frontend**: Vanilla HTML/JS

### 🎯 Key Features

1. **Truly Agentic**: Not just a pipeline - agents make autonomous decisions
2. **Hinglish Support**: 100+ medical phrase mappings
3. **Red Flag Detection**: Automatic urgent case identification
4. **Bilingual Output**: English + Hindi summaries
5. **Agent Trace**: See exactly what each agent decided
6. **Privacy First**: All data stored locally

### 📝 License
MIT License - Built for Hackathon

### 👥 Team
Built for HealthTech Hackathon 2025
# Agentic-AI---GHIA
