import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(36) for combinations of numbers (1-9) and operations (+, -, *, /)
        self.action_space = spaces.Discrete(36)

        # Observation space:
        # - Number pool: counts of numbers 1-9 (each can be 0, 1, or 2)
        # - Current player's score: 0 to 50
        # - Opponent's score: 0 to 50
        # Total observation space shape: (11,)
        self.observation_space = spaces.Box(
            low=np.array([0] * 9 + [0, 0], dtype=np.float32),
            high=np.array([2] * 9 + [50, 50], dtype=np.float32),
            dtype=np.float32,
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize number pool: counts of numbers 1-9, each with count 2
        self.number_pool = {i: 2 for i in range(1, 10)}  # Numbers 1-9
        # Players' scores
        self.player_scores = {1: 0, -1: 0}
        # Current player: start with player 1
        self.current_player = 1
        # Game done flag
        self.done = False
        # Return initial observation and info
        observation = self._get_observation()
        return observation, {}

    def _get_observation(self):
        # Construct the observation array
        number_counts = [
            self.number_pool[i] for i in range(1, 10)
        ]  # Counts for numbers 1-9
        current_player_score = self.player_scores[self.current_player]
        opponent_player_score = self.player_scores[-self.current_player]
        observation = np.array(
            number_counts + [current_player_score, opponent_player_score],
            dtype=np.float32,
        )
        return observation

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action index to (number, operation)
        number_index = action // 4  # 0-8 for numbers 1-9
        operation_index = action % 4  # 0-3 for operations '+', '-', '*', '/'
        number = number_index + 1  # Map to numbers 1-9
        operation_map = {0: "+", 1: "-", 2: "*", 3: "/"}
        operation = operation_map[operation_index]

        # Check if the selected number is available in the number pool
        if self.number_pool.get(number, 0) <= 0:
            # Invalid action: number not available
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Perform the operation
        current_score = self.player_scores[self.current_player]
        selected_number = number

        try:
            if operation == "+":
                new_score = current_score + selected_number
            elif operation == "-":
                new_score = current_score - selected_number
            elif operation == "*":
                new_score = current_score * selected_number
            elif operation == "/":
                if selected_number == 0:
                    # Division by zero not allowed
                    raise ZeroDivisionError
                new_score = current_score // selected_number  # Integer division
            else:
                # Invalid operation
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, {}

            # Check if the new score is a valid non-negative integer
            if new_score < 0:
                # Invalid move: result is negative
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, {}
        except Exception:
            # Invalid move due to exception (e.g., division by zero)
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Valid move: update score and number pool
        self.player_scores[self.current_player] = new_score
        self.number_pool[number] -= 1

        # Check for victory or loss conditions
        if new_score == 50:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}
        elif new_score > 50:
            # Current player loses by overshooting
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}
        else:
            # Continue the game: switch players
            self.current_player *= -1
            reward = 0
            return self._get_observation(), reward, False, False, {}

    def render(self):
        # Return a visual representation of the environment state as a string
        out = "Number Pool:\n"
        for num in range(1, 10):
            out += f"Number {num}: {self.number_pool[num]} remaining\n"
        out += f"\nPlayer {1 if self.current_player == 1 else 2}'s Turn\n"
        out += f"Player Scores:\nPlayer 1: {self.player_scores[1]}\nPlayer 2: {self.player_scores[-1]}\n"
        return out

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        current_score = self.player_scores[self.current_player]
        for action in range(36):
            number_index = action // 4  # 0-8
            operation_index = action % 4  # 0-3
            number = number_index + 1  # Map to numbers 1-9
            operation_map = {0: "+", 1: "-", 2: "*", 3: "/"}
            operation = operation_map[operation_index]
            # Check if the number is available
            if self.number_pool.get(number, 0) > 0:
                try:
                    if operation == "+":
                        new_score = current_score + number
                    elif operation == "-":
                        new_score = current_score - number
                    elif operation == "*":
                        new_score = current_score * number
                    elif operation == "/":
                        if number == 0:
                            continue  # Cannot divide by zero
                        new_score = current_score // number
                    # Check if the new score is valid
                    if new_score >= 0 and isinstance(new_score, int):
                        valid_actions.append(action)
                except Exception:
                    continue  # Skip invalid actions due to exceptions
        return valid_actions
