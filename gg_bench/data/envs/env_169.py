import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(20)  # Actions correspond to numbers 1 to 20
        # Observation space consists of 21 elements:
        # - 20 elements for the availability of numbers 1 to 20 (1 if available, 0 if selected)
        # - 1 element for the opponent's last selected number (0 if none)
        self.observation_space = spaces.Box(
            low=np.zeros(21, dtype=np.int32),
            high=np.concatenate((np.ones(20, dtype=np.int32), [20])),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(
            20, dtype=np.int32
        )  # Numbers 1 to 20 are available
        self.opponent_last_number = 0  # No number selected by opponent yet
        self.done = False
        self.first_move = True  # Indicates if it's the first move of the game
        return self._get_observation(), {}  # Observation and info

    def step(self, action):
        info = {}
        if self.done:
            return self._get_observation(), 0, True, False, info

        selected_number = (
            action + 1
        )  # Action corresponds to selecting number action + 1 (1-20)

        # Check if selected number is within valid range
        if selected_number < 1 or selected_number > 20:
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, info

        # Check if the selected number is available
        if self.available_numbers[action] == 0:
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, info

        # Validate move based on game rules
        if not self.first_move:
            if not self._is_valid_move(selected_number, self.opponent_last_number):
                reward = -10
                self.done = True
                return self._get_observation(), reward, True, False, info

        # Valid move by the agent
        self.available_numbers[action] = 0  # Mark the number as selected
        self.first_move = False  # After the first move
        agent_last_number = selected_number  # Update agent's last selected number

        # Check if opponent can make a valid move
        opponent_valid_moves = self._get_valid_moves(agent_last_number)
        if not opponent_valid_moves:
            reward = 1  # Agent wins
            self.done = True
            return self._get_observation(), reward, True, False, info

        # Opponent makes a move (randomly select from valid moves)
        opponent_action = (
            np.random.choice(opponent_valid_moves) - 1
        )  # Convert to 0-based index
        self.available_numbers[opponent_action] = 0  # Mark the number as selected
        self.opponent_last_number = (
            opponent_action + 1
        )  # Update opponent's last selected number

        # Check if agent can make a valid move for the next turn
        agent_valid_moves = self._get_valid_moves(self.opponent_last_number)
        if not agent_valid_moves:
            reward = -1  # Agent loses
            self.done = True
            return self._get_observation(), reward, True, False, info

        # Continue the game
        reward = 0
        return self._get_observation(), reward, False, False, info

    def render(self):
        output = "Numbers Available:\n"
        for i in range(20):
            num = i + 1
            status = "Available" if self.available_numbers[i] == 1 else "Selected"
            output += f"{num}: {status}\n"
        output += f"Opponent's Last Number: {self.opponent_last_number}\n"
        return output

    def valid_moves(self):
        if self.first_move:
            return [i for i in range(20) if self.available_numbers[i] == 1]
        else:
            return [
                i
                for i in range(20)
                if self.available_numbers[i] == 1
                and self._is_valid_move(i + 1, self.opponent_last_number)
            ]

    def _get_observation(self):
        observation = np.concatenate(
            (self.available_numbers.copy(), [self.opponent_last_number])
        )
        return observation

    def _is_valid_move(self, selected_number, last_number):
        if self.available_numbers[selected_number - 1] == 0:
            return False
        if selected_number == last_number:
            return False
        if selected_number % last_number == 0 or last_number % selected_number == 0:
            return True
        return False

    def _get_valid_moves(self, last_number):
        valid_moves = []
        for i in range(20):
            if self.available_numbers[i] == 1 and self._is_valid_move(
                i + 1, last_number
            ):
                valid_moves.append(i + 1)
        return valid_moves
