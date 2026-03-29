# 🏥 ClinicalTrialEnv

**AI Agent Environment for Clinical Research Coordination**

An OpenEnv-compatible reinforcement learning environment that simulates three core clinical research coordination tasks, grounded in **FDA 21 CFR 312.32** and **ICH E6 GCP** regulatory frameworks. 

*This environment is designed to be fully runnable offline using local LLMs via Ollama, ensuring privacy for clinical datasets and unlimited free inference testing!*

##  Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `eligibility_screening` | 🟢 Easy | Screen patient records against trial inclusion/exclusion criteria |
| `adverse_event_triage` | 🟡 Medium | Rank adverse event reports by FDA reporting urgency (IND Safety Reports) |
| `deviation_assessment` | 🔴 Hard | Classify and respond to protocol deviations per ICH E6 GCP guidelines |

## 🏗️ Architecture

```
ClinicalTrialEnv/
├── openenv.yaml              # Environment manifest
├── inference.py              # Agent inference script (root level)
├── clinical_trial_env/       # Client package
│   ├── client.py             # HTTPEnvClient wrapper
│   └── models.py             # Typed Action/Observation/State models
├── server/
│   ├── app.py                # FastAPI app (WebSocket + REST)
│   ├── environment.py        # Core environment logic
│   └── graders/              # Task-specific grading functions
├── data/                     # Pre-baked clinical datasets
├── scripts/                  # Data generation utilities
└── Dockerfile                # HF Spaces deployment
```

##  Quick Start (Local Offline Mode)

### 1. Install Dependencies

You will need `uv` and `openenv-core` to run the server:
```bash
pip install openenv-core uv
```

### 2. Install & Start Ollama 

This environment is configured by default to use **Ollama** for all inference and data generation, keeping your tests completely free.

- **Windows:** `winget install Ollama.Ollama` (or download from ollama.com)
- **Mac/Linux:** `curl -fsSL https://ollama.ai/install.sh | sh`

Pull the required models:
```bash
ollama pull llama3.2        # smaller, faster model for inference testing
ollama pull llama3.1:8b     # slightly larger model for robust data generation
```

### 3. Configure `.env`
Copy the `.env.example` file to create your local `.env`:
```bash
cp .env.example .env
```
Ensure your variables point to your local Ollama server (`http://localhost:11434/v1`).

### 4. Start the Server

In your first terminal, start the local environment server:
```bash
uv run uvicorn server.app:app --port 7860
```

### 5. Run the Evaluation Agent

In a second terminal, execute the inference script:
```bash
python inference.py
```
*The agent will immediately start connecting to Ollama, parsing your tasks, submitting its responses to the server, and printing out its scores in `[0.0, 1.0]` increments for every decision.*


## 🌐 Deployment (HuggingFace Serverless Router)

When you are ready to deploy your agent or submit it to an OpenEnv leaderboard, you can instantly switch back to the cloud-hosted HuggingFace serverless router.

1. Ensure you have a free [HuggingFace Token](https://huggingface.co/settings/tokens).
2. Edit your `.env` file to disable Ollama and enable the cloud router:
   ```env
   HF_TOKEN=hf_your_actual_token_here
   API_BASE_URL=https://router.huggingface.co/v1
   MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
   ```
3. Run `python inference.py`. Your agent will now execute using HuggingFace's cloud inference without needing any code changes.

##  Grading

All graders return scores in `[0.0, 1.0]`:

- **Eligibility Screening** `[0.950+]`  
  Correct inclusion/exclusion decisions with high-quality reasoning.

- **Adverse Event Triage** `[0.900+]`  
  Accurate urgency classification along with correct FDA reporting timelines.

- **Deviation Assessment** `[0.850–0.950]`  
  Proper severity classification with appropriate corrective actions.

## Model Setup & Evaluation

### 🖥️ Local Models

- **Llama 3.1 (8B)** `[Baseline]`  
  Lightweight and fast. Good for initial testing and quick iterations.
  ```bash
  ollama pull llama3.1:8b
  ```

- **Qwen 2.5 (14B)** `[0.850+]`  
  Balanced performance with better reasoning than smaller models.
  ```bash
  ollama pull qwen2.5:14b
  ```

- **Phi-4** `[0.800–0.900]`  
  Efficient and compact. Performs well on structured tasks but may lack depth.
  ```bash
  ollama pull phi4
  ```

- **Mistral Nemo** `[0.850–0.920]`  
  Strong general-purpose model with good reasoning and speed tradeoff.
  ```bash
  ollama pull mistral-nemo
  ```
---

### ☁️ Cloud Models (Hugging Face)

- **Llama 3.3 (70B Instruct)** `[0.950+]`  
  High-quality reasoning and instruction following. Near top-tier performance.
  ```bash
  meta-llama/Llama-3.3-70B-Instruct
  ```

- **Qwen 2.5 (72B Instruct)** `[0.950+]`  
  Excellent logical reasoning and consistency across complex tasks.
  ```bash
  Qwen/Qwen2.5-72B-Instruct
  ```

- **Mixtral 8x22B (Instruct v0.1)** `[0.900–0.960]`  
  Mixture-of-experts model with strong performance and efficiency.
  ```bash
  mistralai/Mixtral-8x22B-Instruct-v0.1
  ```
---

## Goal

Evaluate all models across:
- Eligibility Screening  
- Adverse Event Triage  
- Deviation Assessment  

👉 Select the **best-performing model** based on:
- Accuracy  
- Reasoning quality  
- Consistency  
- Latency (optional)

---

## 🔥 Notes

- Local models → Faster, cheaper, good for prototyping  
- Cloud models → More powerful, better for final submissions  
- Use the same evaluation pipeline for fair comparison


## 📋 Regulatory Grounding

- **FDA 21 CFR 312.32** — IND Safety Reporting requirements
- **ICH E6(R2) GCP** — Good Clinical Practice guidelines
- **ICH E2A/E2B** — Clinical safety data management standards
- **CTCAE v5.0** — Common Terminology Criteria for Adverse Events

## 📄 License

MIT
