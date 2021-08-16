from gym.spaces import Discrete, MultiDiscrete
from gym.spaces import Box


def create_space(dtype, shape, **kwargs):
    if dtype.startswith('int'):
        if len(shape) > 1:
            return MultiDiscrete(nvec=shape)
        return Discrete(shape[0])
    else:
        low = kwargs.get("low", -1)  # TODO:
        high = kwargs.get("high", 1)  # TODO:
        assert low is not None
        assert high is not None
        return Box(low=low, high=high, shape=shape, dtype=dtype)


class DummyAgent:
    def __init__(self, obs_space, action_space):
        self.obs_space = create_space(**obs_space)
        self.action_space = create_space(**action_space)

    def step(self, **kwargs):
        obs = kwargs.get("obs")
        assert obs is not None
        # assert self.obs_space.contains(obs)

        next_obs = kwargs.get("next_obs")
        assert next_obs is not None
        # assert self.obs_space.contains(next_obs)

        action = kwargs.get('action')
        assert action is not None
        # assert self.action_space.contains(action)

        reward = kwargs.get('reward')
        assert reward is not None

        done = kwargs.get('done')
        assert done is not None
        # assert isinstance(done, bool)

    def act(self, **kwargs):
        state = kwargs.get("state")
        assert state, state
        # assert self.obs_space.contains(state), f"State ({state}) is not container within obs space (obs_space: {self.obs_space})"
        return self.action_space.sample().tolist()
