---
title: ClinicalTrialEnv
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# 🏥 ClinicalTrialEnv

**AI Agent Environment for Clinical Research Coordination**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://openenv.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-Powered-red)](https://pytorch.org)
[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-blue)](https://huggingface.co/spaces/Sanju562586/ClinicalTrialEnv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **OpenEnv-compatible** reinforcement learning environment that simulates core clinical research coordination tasks. It is grounded in the **FDA 21 CFR 312.32** and **ICH E6 GCP** regulatory frameworks, testing an agent's ability to handle complex medical logic with high precision.

---

## 🎯 Key Features

- **Regulatory Grounded**: All tasks are mapped to real FDA and ICH safety reporting and data integrity rules.
- **PyTorch Scored**: Uses a custom `TorchConfidenceScorer` (Softmax-calibrated) in the core reward pipeline to evaluate agent reasoning.
- **Offline Ready**: Fully compatible with **Ollama** for private, free local development.
- **OpenEnv Spec**: Adheres strictly to the Meta PyTorch OpenEnv technical manifest.

---

## 📋 Tasks

| Task | Difficulty | Description | Regulatory Framework |
|------|-----------|-------------|-----------------------|
| `eligibility_screening` | 🟢 Easy | Screen patient records against trial inclusion/exclusion criteria | Trial Protocol |
| `adverse_event_triage` | 🟡 Medium | Rank adverse event reports by FDA reporting urgency | FDA 21 CFR 312.32 |
| `deviation_assessment` | 🔴 Hard | Classify and respond to protocol deviations | ICH E6 GCP |

---

## 🛠️ Technical Architecture

### Core Reward Pipeline (PyTorch)
Unlike heuristic-based environments, **ClinicalTrialEnv** uses PyTorch to calibrate agent rewards. The `TorchConfidenceScorer` calculates a probability distribution over valid regulatory categories based on the agent's provided rationale, ensuring that agents are rewarded for *why* they made a decision, not just the decision itself.

### Folder Structure
```
ClinicalTrialEnv/
├── openenv.yaml              # OpenEnv Manifest
├── inference.py              # Main Agent Loop
├── clinical_trial_env/       # Client Library
├── server/
│   ├── app.py                # FastAPI Server (REST/WS)
│   ├── environment.py        # RL Environment Logic
│   └── graders/              # PyTorch-based Grading Modules
├── data/                     # HIPAA-synthetic Clinical Datasets
└── Dockerfile                # Deployment Configuration
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have `uv` installed for fast dependency management:
```powershell
pip install uv openenv-core
```

### 2. Install Project
Clone the repository and install dependencies:
```powershell
git clone https://github.com/Sanju562586/ClinicalTrialEnv.git
cd ClinicalTrialEnv
uv pip install -e .
```

### 3. Configure Environment
Copy the example environment file and add your keys (if using cloud):
```powershell
cp .env.example .env
```

---

## 🚦 Usage (Local Offline Mode)

### Step 1: Start Server
Launch the environment server in your first terminal:
```powershell
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Step 2: Run Agent
In a second terminal, execute the inference script:
```powershell
python inference.py
```
*By default, this uses **Ollama** (llama3.2). Ensure Ollama is running locally.*

---

## 📤 Submission & Deployment

To submit this environment to the leaderboard, follow the standard 6-step workflow:

1. **Application Form**: Choose the "Clinical Research AI" problem statement.
2. **Scaffold**: Project is already scaffolded in this repository.
3. **Build**: Define logic (Done in `server/environment.py`).
4. **Test locally**: Verify logic using `python inference.py`.
5. **Deploy**: Push to Hugging Face:
   ```powershell
   uv run openenv push --repo-id your-username/ClinicalTrialEnv
   ```
6. **Submit**: Paste your HF Spaces URL into the [Hackathon Dashboard](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard).

---

## 📋 Regulatory References

- **FDA 21 CFR 312.32** — IND Safety Reporting requirements
- **ICH E6(R2) GCP** — Good Clinical Practice guidelines
- **CTCAE v5.0** — Common Terminology Criteria for Adverse Events

## 📄 License

MIT
