import logging
import sys
from collections import deque
from datetime import datetime
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException

from app.dummy_agent import DummyAgent
from app.types import AgentAction, AgentCreate, AgentStep

# Initiate module with setting up a server
app = FastAPI(
    title="Agents Bar - Dummy Agent",
    description="Dummy agent compatible with Agents Bar's Agent entity",
    docs_url="/docs",
)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

agent: Optional[DummyAgent] = None
last_active = datetime.utcnow()
last_metrics_time = datetime.utcnow()
metrics_buffer = deque(maxlen=20)


@app.get("/ping")
def ping():
    return {"msg": "All good"}


@app.post("/agent", status_code=201)
def api_post_agent(agent_create: AgentCreate):
    global agent

    agent = DummyAgent(obs_space=agent_create.obs_space, action_space=agent_create.action_space)
    return {"response": "Successfully created a new agent"}


@app.delete("/agent/{agent_name}", status_code=204)
def api_delete_agent(agent_name: str):
    """Assumption is that the service contains only one agent. Otherwise... something else."""
    global agent
    if agent is None:
        return {"response": "Agent doesn't exist."}

    if agent_name == "dummy":
        agent = None
        return {"response": "Deleted successfully."}

    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    

@app.get("/agent/state")
def api_get_agent_state():
    if agent is None:
        raise HTTPException(status_code=404, detail="No agent found")
    return {}


@app.get("/agent/")
def api_get_agent_info():
    if agent is None:
        raise HTTPException(status_code=404, detail="No agent found")
    return agent


@app.get("/agent/last_active", response_model=datetime)
def api_get_agent_last_active():
    if agent is None:
        raise HTTPException(status_code=404, detail="No agent found")
    return last_active


@app.get("/agent/hparams")
def api_get_agent_hparasm():
    if agent is None:
        raise HTTPException(status_code=404, detail="No agent found")
    return {"hyperparameters": {}}


@app.get("/agent/loss")
def api_get_agent_loss(last_samples: int = 1):
    if agent is None:
        raise HTTPException(status_code=404, detail="No agent found")
    return []


@app.post("/agent/step", status_code=200)
def api_post_agent_step(step: AgentStep):
    # TODO: Agent should have a property whether it's discrete
    assert agent, "Agent needs to be initiated before it can be used"
    global last_active
    last_active = datetime.utcnow()

    agent.step(
        obs=step.obs,
        action=step.action,
        reward=step.reward,
        next_obs=step.next_obs,
        done=step.done
    )

    return {"response": "Stepping"}


@app.post("/agent/act", response_model=AgentAction)
def api_post_agent_act(state: List[Any], noise: float=0.):
    assert agent, "Agent needs to be initiated before it can be used"
    global last_active
    last_active = datetime.utcnow()
    try:
        action = agent.act(state=state, noise=noise)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sorry :(\n{e}")

    print(action)
    return AgentAction(action=action)
    
