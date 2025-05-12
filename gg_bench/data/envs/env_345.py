import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions are numbers from 1 to 9 (indices 0 to 8)
        self.action_space = spaces.Discrete(9)

        # Observation space includes:
        # - Player 1 Life Points (0 to 15)
        # - Player 2 Life Points (0 to 15)
        # - Available Numbers (1 if available, 0 if used), for numbers 1 to 9
        # Total observation space shape: (11,)
        self.observation_space = spaces.Box(
            low=0, high=15, shape=(11,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize life points
        self.player1_life = 15
        self.player2_life = 15

        # Numbers 1 to 9 are available at the start (index 0 to 8)
        self.available_numbers = np.ones(9, dtype=np.int8)

        # Game not done
        self.done = False

        # Initialize current player (1 for Player 1)
        self.current_player = 1

        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Map action index to number (1 to 9)
        player_number = action + 1

        # Check if the selected number is available
        if self.available_numbers[action] == 0:
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Remove the selected number from available numbers
        self.available_numbers[action] = 0

        # Opponent (Player 2) selects a number randomly from available numbers
        opponent_available_actions = np.where(self.available_numbers == 1)[0]
        if len(opponent_available_actions) == 0:
            # No numbers left; determine the winner
            self.done = True
            if self.player1_life > self.player2_life:
                # Player 1 wins
                return self._get_obs(), 1, True, False, {}
            elif self.player1_life < self.player2_life:
                # Player 2 wins
                return self._get_obs(), -1, True, False, {}
            else:
                # Draw (should not occur as per game rules)
                return self._get_obs(), 0, True, False, {}

        opponent_action = self.np_random.choice(opponent_available_actions)
        opponent_number = opponent_action + 1
        # Remove the opponent's selected number from available numbers
        self.available_numbers[opponent_action] = 0

        # Resolve combat
        if player_number > opponent_number:
            damage = player_number - opponent_number
            self.player2_life -= damage
            self.player2_life = max(self.player2_life, 0)
        elif opponent_number > player_number:
            damage = opponent_number - player_number
            self.player1_life -= damage
            self.player1_life = max(self.player1_life, 0)
        # If numbers are equal, no damage is dealt

        # Check for win condition
        if self.player1_life == 0 or self.player2_life == 0:
            self.done = True
            if self.player1_life > self.player2_life:
                # Player 1 wins
                return self._get_obs(), 1, True, False, {}
            elif self.player1_life < self.player2_life:
                # Player 2 wins
                return self._get_obs(), -1, True, False, {}
            else:
                # Draw (should not occur as per game rules)
                return self._get_obs(), 0, True, False, {}

        # If no numbers left, determine the winner
        if np.sum(self.available_numbers) == 0:
            self.done = True
            if self.player1_life > self.player2_life:
                # Player 1 wins
                return self._get_obs(), 1, True, False, {}
            elif self.player1_life < self.player2_life:
                # Player 2 wins
                return self._get_obs(), -1, True, False, {}
            else:
                # Draw (should not occur as per game rules)
                return self._get_obs(), 0, True, False, {}

        # Continue the game
        return self._get_obs(), 0, False, False, {}

    def render(self):
        # Display the current game state
        available_numbers_list = [
            str(i + 1) for i in range(9) if self.available_numbers[i] == 1
        ]
        board_str = f"Player 1 Life Points: {self.player1_life}\n"
        board_str += f"Player 2 Life Points: {self.player2_life}\n"
        board_str += f"Available Numbers: {', '.join(available_numbers_list)}\n"
        return board_str

    def valid_moves(self):
        # Return a list of valid actions (indices of available numbers)
        return [i for i in range(9) if self.available_numbers[i] == 1]

    def _get_obs(self):
        # Observation includes:
        # - Player 1 Life Points
        # - Player 2 Life Points
        # - Available Numbers (binary indicators)
        obs = np.zeros(11, dtype=np.float32)
        obs[0] = self.player1_life
        obs[1] = self.player2_life
        obs[2:] = self.available_numbers
        return obs
