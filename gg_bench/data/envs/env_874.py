import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            9
        )  # Numbers 2 to 10 correspond to actions 0 to 8

        # Observation space includes:
        # - [0]: Current player's score (0 to 50)
        # - [1]: Opponent's score (0 to 50)
        # - [2-10]: Availability of numbers 2 to 10 (1 if available, 0 if not)
        self.observation_space = spaces.Box(
            low=np.array([0, 0] + [0] * 9, dtype=np.int32),
            high=np.array([50, 50] + [1] * 9, dtype=np.int32),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize game state
        self.player_scores = [0, 0]  # Player 1 and Player 2 scores
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.available_numbers = [1] * 9  # Numbers 2 to 10 are available
        self.done = False
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if action < 0 or action >= 9 or not self.available_numbers[action]:
            # Invalid action
            return self._get_observation(), -10, True, False, {}
        # Valid action
        selected_number = action + 2  # Map actions 0-8 to numbers 2-10
        self.available_numbers[action] = 0  # Remove number from availability

        prime_numbers = [2, 3, 5, 7]
        is_prime = selected_number in prime_numbers

        current_player = self.current_player
        opponent_player = 1 - self.current_player

        current_score = self.player_scores[current_player]
        tentative_new_score = current_score + selected_number

        # Over 50 Rule
        if tentative_new_score > 50:
            # Score remains the same
            new_score = current_score
        else:
            new_score = tentative_new_score

            # Apply effects to opponent if prime number
            if is_prime:
                opponent_score = self.player_scores[opponent_player]
                new_opponent_score = max(0, opponent_score - selected_number)
                self.player_scores[opponent_player] = new_opponent_score

        self.player_scores[current_player] = new_score

        # Check for win condition
        if new_score == 50:
            reward = 1
            terminated = True
        else:
            reward = 0
            terminated = False

        if not terminated:
            # Switch to the next player
            self.current_player = opponent_player

        observation = self._get_observation()
        return observation, reward, terminated, False, {}

    def render(self):
        # Generate a string representation of the game state
        available_nums = [str(i + 2) for i in range(9) if self.available_numbers[i]]
        available_nums_str = ", ".join(available_nums) if available_nums else "None"

        output = "--------------------------------------------------\n"
        output += f"Player {self.current_player + 1}'s Turn\n"
        output += f"Available Numbers: {available_nums_str}\n"
        output += f"Your Score: {self.player_scores[self.current_player]}\n"
        output += f"Opponent's Score: {self.player_scores[1 - self.current_player]}\n"
        return output

    def valid_moves(self):
        # Return a list of valid action indices
        return [i for i, available in enumerate(self.available_numbers) if available]

    def _get_observation(self):
        # Observation includes current player's score, opponent's score, and number availability
        obs = np.zeros(11, dtype=np.int32)
        obs[0] = self.player_scores[self.current_player]
        obs[1] = self.player_scores[1 - self.current_player]
        obs[2:] = self.available_numbers
        return obs
