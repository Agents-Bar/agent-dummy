from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, StrictInt, StrictFloat


class NumericalList(BaseModel):
    __root__: Union[StrictInt, StrictFloat, List[NumericalList]]

# Unroll NumericalList
NumericalList.update_forward_refs()

ActionType = NumericalList
ObservationType = NumericalList
RewardType = Union[float, List[float]]
DoneType = Union[bool, List[bool]]
AgentLoss = Dict[str, Optional[float]]


class AgentAction(BaseModel):
    action: ActionType


class AgentStep(BaseModel):
    obs: ObservationType
    action: ActionType
    next_obs: ObservationType
    reward: RewardType
    done: DoneType


class AgentStateJSON(BaseModel):
    model: str
    state_space: int
    action_space: int
    encoded_config: str
    encoded_network: str
    encoded_buffer: str


class AgentInfo(BaseModel):
    model: str
    hyperparameters: Dict[str, Any]
    last_active: datetime
    discret: bool


class AgentCreate(BaseModel):
    model_type: str
    obs_space: Dict[str, Any]
    action_space: Dict[str, Any]
    model_config: Optional[Dict[str, Any]]
    network_state: Optional[bytes] = None
    buffer_state: Optional[bytes] = None
