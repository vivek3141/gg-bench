from copy import deepcopy
from typing import Any, Callable, Optional

import numpy as np
from gym import Env


def get_valid_moves(env: Env) -> list:
    if hasattr(env, "valid_moves"):
        return env.valid_moves()  # type: ignore[attr-defined]
    elif hasattr(env, "action_space") and hasattr(env.action_space, "n"):
        return list(range(env.action_space.n))  # type: ignore[attr-defined]
    else:
        raise AttributeError("Env needs'valid_moves' fn or 'action_space' attribute")


def minimax(
    state: Any,
    env: Env,
    depth: int = 0,
    is_maximizing: bool = True,
    max_depth: int = 3,
    evaluate_state: Optional[Callable[[Any], float]] = None,
    accumulated_reward: float = 0,
) -> float:
    """
    Implements the minimax algorithm for game-playing AI.

    Args:
        state (Any): The current state of the game.
        env (Env): The environment object representing the game.
        depth (int): The current depth in the game tree.
        is_maximizing (bool): Whether it's the maximizing player's turn.
        max_depth (int): The maximum depth to search in the game tree.
        evaluate_state (Optional[Callable[[Any], float]]): A function to evaluate non-terminal states.
        accumulated_reward (float): The accumulated reward up to the current state.

    Returns:
        float: The best possible score for the current player.
    """
    # Base case: terminal state or depth limit
    if depth == max_depth:
        return evaluate_state(state) if evaluate_state else accumulated_reward

    # Check if the game is done
    if is_maximizing:
        max_eval = -np.inf
        for action in get_valid_moves(env):
            env_copy = deepcopy(env)
            new_state, reward, done, _, _ = env_copy.step(action)
            if done:
                return (
                    evaluate_state(new_state)
                    if evaluate_state
                    else (accumulated_reward + reward)
                )
            eval = minimax(
                new_state,
                env_copy,
                depth + 1,
                False,
                max_depth,
                evaluate_state,
                accumulated_reward + reward,
            )
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = np.inf
        for action in get_valid_moves(env):
            env_copy = deepcopy(env)
            new_state, reward, done, _, _ = env_copy.step(action)
            if done:
                return (
                    evaluate_state(new_state)
                    if evaluate_state
                    else (accumulated_reward + reward)
                )
            eval = minimax(
                new_state,
                env_copy,
                depth + 1,
                True,
                max_depth,
                evaluate_state,
                accumulated_reward + reward,
            )
            min_eval = min(min_eval, eval)
        return min_eval


def pick_minimax_action(
    state: Any,
    env: Env,
    max_depth: int = 3,
    evaluate_state: Optional[Callable[[Any], float]] = None,
) -> int:
    """
    Picks the best action for the current state using the minimax algorithm.

    Args:
        state (Any): The current state of the game.
        env (Env): The environment object representing the game.
        max_depth (int): The maximum depth to search in the game tree.
        evaluate_state (Optional[Callable[[Any], float]]): A function to evaluate non-terminal states.

    Returns:
        int: The best action to take.
    """
    best_action = get_valid_moves(env)[0]
    best_score = -np.inf

    for action in get_valid_moves(env):
        env_copy = deepcopy(env)
        new_state, reward, done, _, _ = env_copy.step(action)
        if done:
            score = evaluate_state(new_state) if evaluate_state else reward
        else:
            score = minimax(
                new_state,
                env_copy,
                depth=1,
                is_maximizing=False,
                max_depth=max_depth,
                evaluate_state=evaluate_state,
                accumulated_reward=reward,
            )

        if score > best_score:
            best_score = score
            best_action = action

    return best_action
