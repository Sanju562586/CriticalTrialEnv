"""
server/app.py — FastAPI application for ClinicalTrialEnv.

Exposes WebSocket and REST endpoints:
  /ws          — WebSocket for persistent agent sessions
  /reset       — POST: reset environment for a task
  /step        — POST: submit action, receive reward
  /state       — GET: current episode metadata
  /tasks       — GET: list available tasks
  /health      — GET: health check
"""

import json
import dataclasses
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .environment import ClinicalTrialEnvironment


# ---- Pydantic request/response models for REST endpoints ----

class ResetRequest(BaseModel):
    task: str = "eligibility_screening"


class ActionRequest(BaseModel):
    decision: str = ""
    reasoning: str = ""
    criteria_cited: list[str] = []
    urgency_classification: str = ""
    reporting_timeline: str = ""
    classification: str = ""
    corrective_action: str = ""
    rationale: str = ""
    confidence: float = 0.5


# ---- App setup ----

env = ClinicalTrialEnvironment()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    print("🏥 ClinicalTrialEnv server starting...")
    yield
    print("🏥 ClinicalTrialEnv server shutting down.")


app = FastAPI(
    title="ClinicalTrialEnv",
    description="AI Agent Environment for Clinical Research Coordination",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_dict(obj):
    """Convert dataclass to dict, handling nested dataclasses."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return obj


# ---- REST Endpoints ----

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ClinicalTrialEnv", "version": "1.0.0"}


@app.get("/tasks")
async def get_tasks():
    return {"tasks": env.get_tasks()}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    if request is None:
        request = ResetRequest()
    result = env.reset(task=request.task)
    return {
        "observation": _to_dict(result.observation),
        "info": result.info,
    }


@app.post("/step")
async def step(request: Optional[ActionRequest] = None):
    if request is None:
        request = ActionRequest()
    action_dict = request.model_dump()
    result = env.step(action_dict)
    return {
        "observation": _to_dict(result.observation),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
async def get_state():
    return _to_dict(env.state)


# ---- WebSocket Endpoint ----

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for persistent agent sessions.

    Protocol:
    - Client sends JSON messages with 'type' field: 'reset', 'step', 'state'
    - Server responds with JSON containing the result

    Example messages:
    {"type": "reset", "task": "eligibility_screening"}
    {"type": "step", "action": {"decision": "eligible", "reasoning": "..."}}
    {"type": "state"}
    """
    await ws.accept()
    ws_env = ClinicalTrialEnvironment()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "reset":
                task = msg.get("task", "eligibility_screening")
                result = ws_env.reset(task=task)
                await ws.send_json({
                    "type": "reset_result",
                    "observation": _to_dict(result.observation),
                    "info": result.info,
                })

            elif msg_type == "step":
                action = msg.get("action", {})
                result = ws_env.step(action)
                await ws.send_json({
                    "type": "step_result",
                    "observation": _to_dict(result.observation),
                    "reward": result.reward,
                    "done": result.done,
                    "info": result.info,
                })

            elif msg_type == "state":
                await ws.send_json({
                    "type": "state_result",
                    **_to_dict(ws_env.state),
                })

            elif msg_type == "tasks":
                await ws.send_json({
                    "type": "tasks_result",
                    "tasks": ws_env.get_tasks(),
                })

            else:
                await ws.send_json({
                    "type": "error",
                    "message": f"Unknown message type: '{msg_type}'. "
                               f"Valid types: reset, step, state, tasks",
                })

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
