import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete actions representing numbers 2 to 20 (indices 0 to 18)
        self.action_space = spaces.Discrete(19)

        # Observation space:
        # - First element: normalized last number in chain (value between 0 and 1)
        # - Next 19 elements: number pool status (1.0 if available, 0.0 if used)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(20,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.chain = [1]
        self.number_pool = np.ones(19, dtype=np.float32)  # Numbers 2 to 20 available
        self.current_player = 1
        self.done = False

        self.last_number = 1
        last_number_normalized = (
            self.last_number - 1
        ) / 19.0  # Normalize between 0 and 1
        observation = np.concatenate(
            ([last_number_normalized], self.number_pool.copy())
        )  # Observation is last number + number pool status
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            last_number_normalized = (self.last_number - 1) / 19.0
            observation = np.concatenate(
                ([last_number_normalized], self.number_pool.copy())
            )
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # Observ., reward, terminated, truncated, info

        if action < 0 or action >= 19:
            # Invalid action index
            last_number_normalized = (self.last_number - 1) / 19.0
            observation = np.concatenate(
                ([last_number_normalized], self.number_pool.copy())
            )
            return observation, -10, True, False, {}
        number = action + 2  # Map action index to number (2 to 20)

        if self.number_pool[action] == 0.0:
            # Number already used
            last_number_normalized = (self.last_number - 1) / 19.0
            observation = np.concatenate(
                ([last_number_normalized], self.number_pool.copy())
            )
            return observation, -10, True, False, {}

        last_number = self.chain[-1]

        # Check if the move is valid (number is factor or multiple of last number)
        if last_number % number != 0 and number % last_number != 0:
            # Invalid move
            last_number_normalized = (last_number - 1) / 19.0
            observation = np.concatenate(
                ([last_number_normalized], self.number_pool.copy())
            )
            return observation, -10, True, False, {}

        # Valid move
        self.chain.append(number)
        self.number_pool[action] = 0.0  # Mark number as used
        self.last_number = number
        self.current_player *= -1  # Switch player

        # Check if next player has valid moves
        has_valid_move = False
        for idx in range(19):
            if self.number_pool[idx] == 1.0:
                next_number = idx + 2
                if number % next_number == 0 or next_number % number == 0:
                    has_valid_move = True
                    break

        if not has_valid_move:
            # Current player wins
            self.done = True
            last_number_normalized = (number - 1) / 19.0
            observation = np.concatenate(
                ([last_number_normalized], self.number_pool.copy())
            )
            return observation, 1.0, True, False, {}

        # Game continues
        last_number_normalized = (number - 1) / 19.0
        observation = np.concatenate(
            ([last_number_normalized], self.number_pool.copy())
        )
        return observation, 0.0, False, False, {}

    def render(self):
        chain_str = "Chain: " + " -> ".join(map(str, self.chain))
        available_numbers = [
            str(i + 2) for i in range(19) if self.number_pool[i] == 1.0
        ]
        number_pool_str = "Available Numbers: " + ", ".join(available_numbers)
        current_player_str = (
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"
        )
        return f"{chain_str}\n{number_pool_str}\n{current_player_str}"

    def valid_moves(self):
        last_number = self.chain[-1]
        valid_actions = []
        for idx in range(19):
            if self.number_pool[idx] == 1.0:
                number = idx + 2
                if last_number % number == 0 or number % last_number == 0:
                    valid_actions.append(idx)
        return valid_actions
