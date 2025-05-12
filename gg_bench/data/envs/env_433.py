import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces

        # Action space: numbers from 1 to 10 (indices 0 to 9)
        self.action_space = spaces.Discrete(10)

        # Observation space:
        # - Positions 0-9: Available numbers (1 if available, 0 if not)
        # - Position 10: Cumulative score of Player 1
        # - Position 11: Cumulative score of Player 2
        # Maximum cumulative score can be up to 55 (sum of numbers 1 to 10)
        self.observation_space = spaces.Box(
            low=0,
            high=55,
            shape=(12,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(10, dtype=np.float32)
        self.scores = {1: 0, -1: 0}
        self.current_player = 1  # Player 1 starts
        self.bids = {1: None, -1: None}
        self.done = False

        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        if action < 0 or action >= 10:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {"invalid_action": True}

        if self.available_numbers[action] == 0:
            # Invalid action: number not available
            self.done = True
            return self._get_observation(), -10, True, False, {"invalid_action": True}

        # Valid action
        bid_number = action + 1  # Convert action index to bid number
        self.bids[self.current_player] = bid_number

        reward = 0
        info = {}

        # Check if both bids have been made
        if self.bids[1] is not None and self.bids[-1] is not None:
            # Resolve the round
            bid_player1 = self.bids[1]
            bid_player2 = self.bids[-1]

            if bid_player1 > bid_player2:
                # Player 1 wins the round
                self.scores[1] += bid_player1 + bid_player2
                winner = 1
            elif bid_player2 > bid_player1:
                # Player 2 wins the round
                self.scores[-1] += bid_player1 + bid_player2
                winner = -1
            else:
                # Tie: No points awarded
                winner = 0

            # Remove the bid numbers from the pool
            self.available_numbers[bid_player1 - 1] = 0
            self.available_numbers[bid_player2 - 1] = 0

            # Assign reward to the current player if they won the round
            if winner == self.current_player:
                reward = 1
            else:
                reward = 0

            # Clear bids for the next round
            self.bids = {1: None, -1: None}

            # Check if the game has ended
            if np.sum(self.available_numbers) == 0:
                self.done = True

        else:
            # Waiting for the next player's bid
            reward = 0

        # Switch to the next player
        self.current_player *= -1

        observation = self._get_observation()
        return observation, reward, self.done, False, info

    def _get_observation(self):
        # Construct the observation array
        obs = np.zeros(12, dtype=np.float32)
        obs[0:10] = self.available_numbers
        obs[10] = self.scores[1]
        obs[11] = self.scores[-1]
        return obs

    def render(self):
        # Create a visual representation of the game state
        output = "Available Numbers: "
        output += ", ".join(
            [str(i + 1) for i in range(10) if self.available_numbers[i] == 1]
        )
        output += "\n"
        output += f"Scores - Player 1: {self.scores[1]}, Player 2: {self.scores[-1]}\n"
        output += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return output

    def valid_moves(self):
        # Return a list of valid action indices (available numbers)
        return [i for i in range(10) if self.available_numbers[i] == 1]
