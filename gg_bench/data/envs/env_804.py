import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: integers from 0 to 9 corresponding to numbers 1 to 10
        self.action_space = spaces.Discrete(10)

        # Observation space: player_total, opponent_total, opponent_last_number
        # opponent_last_number is 0 if there's no previous number
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), high=np.array([50, 50, 10]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_totals = [0, 0]  # Player 1 and Player 2 totals
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.opponent_last_number = None
        self.done = False

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            # Game is over
            observation = self._get_observation()
            return observation, 0, self.done, False, {}

        chosen_number = action + 1  # Map action index to number between 1 and 10

        # Validate the move
        if (
            self.opponent_last_number is not None
            and chosen_number == self.opponent_last_number
        ):
            # Invalid move: same as opponent's last number
            self.done = True
            reward = -10
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

        # Check if the move exceeds 50
        new_total = self.player_totals[self.current_player] + chosen_number
        if new_total > 50:
            # Player loses by exceeding 50
            self.done = True
            reward = -10
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

        # Update player's total sum
        self.player_totals[self.current_player] = new_total

        # Check for a win
        if new_total == 50:
            # Player wins by reaching exactly 50
            self.done = True
            reward = 1
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

        # Continue the game
        reward = 0
        self.opponent_last_number = chosen_number
        self.current_player = 1 - self.current_player  # Switch player

        # Check if the next player has any valid moves
        if not self.valid_moves():
            # Next player cannot make a valid move; current player wins
            self.done = True
            self.current_player = (
                1 - self.current_player
            )  # Switch back to current player
            reward = 1
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

        # Return the observation and reward
        observation = self._get_observation()
        return observation, reward, self.done, False, {}

    def render(self):
        output = "--- Sum Trap Game State ---\n"
        output += f"Player {self.current_player + 1}'s Turn\n"
        output += f"Your Total: {self.player_totals[self.current_player]}\n"
        output += f"Opponent's Total: {self.player_totals[1 - self.current_player]}\n"
        if self.opponent_last_number:
            output += f"Opponent's Last Number: {self.opponent_last_number}\n"
        else:
            output += "No previous opponent number\n"
        output += "---------------------------\n"
        return output

    def valid_moves(self):
        # Returns a list of valid action indices
        valid_actions = []
        for action in range(10):
            chosen_number = action + 1
            if (
                self.opponent_last_number is not None
                and chosen_number == self.opponent_last_number
            ):
                continue  # Cannot choose opponent's last number
            if self.player_totals[self.current_player] + chosen_number > 50:
                continue  # Move exceeds 50
            valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Constructs the observation
        player_total = self.player_totals[self.current_player]
        opponent_total = self.player_totals[1 - self.current_player]
        opponent_last_number = (
            self.opponent_last_number if self.opponent_last_number else 0
        )
        return np.array(
            [player_total, opponent_total, opponent_last_number], dtype=np.int32
        )
