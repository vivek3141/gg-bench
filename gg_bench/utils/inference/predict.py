from typing import List, Optional, Union

import numpy as np
from sbx import DQN, PPO

# Internal Imports
from gg_bench.utils.inference.activation import softmax


def get_prediction(
    agent: Union[PPO, DQN],
    obs: np.ndarray,
    valid_moves: Optional[List[int]] = None,
    epsilon: float = float("-inf"),
    deterministic: bool = False,
) -> int:
    """
    Get a prediction from the agent based on the given observation.
    Can also filter to predict from a subset of valid moves.

    Args:
        agent (Union[PPO, DQN]): The agent (either PPO or DQN) to use for prediction.
        obs (np.ndarray): The observation array.
        valid_moves (Optional[List[int]]): List of valid moves. Defaults to None.
        epsilon (float): Epsilon value for epsilon-greedy exploration. Defaults to negative infinity.
        deterministic (bool): Whether to use deterministic action selection. Defaults to False.

    Returns:
        int: The predicted action.

    Raises:
        AssertionError: If valid_moves is not provided when epsilon is not negative infinity.
        ValueError: If the agent type is not supported.
    """
    if np.random.random() < epsilon:
        assert (
            valid_moves is not None
        ), "valid_moves must be provided when epsilon is not -inf"
        return np.random.choice(valid_moves)

    if isinstance(agent, PPO):
        actor_state = agent.policy.actor_state
        params = actor_state.params
        logits = actor_state.apply_fn(params, obs.reshape(1, -1)).logits
    elif isinstance(agent, DQN):
        qf_state = agent.policy.qf_state
        logits = qf_state.apply_fn(qf_state.params, obs.reshape(1, -1))
    else:
        raise ValueError(f"Unsupported agent type: {type(agent)}")

    if valid_moves is not None:
        logits = logits[:, valid_moves].ravel()

    if deterministic:
        action = int(np.argmax(logits))
    else:
        probs = softmax(logits)
        action = int(np.random.choice(len(logits), p=probs))

    if valid_moves is not None:
        action = valid_moves[action]

    return action
