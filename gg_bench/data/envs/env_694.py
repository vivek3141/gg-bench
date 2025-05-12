import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Numbers from 1 to 50 (actions 0 to 49), action 50 is 'pass'
        self.action_space = spaces.Discrete(51)

        # Observation space is an array of size 53
        # Indices 0-49: number pool availability (1 for available, 0 for selected)
        # Index 50: current player's score
        # Index 51: opponent's score
        # Index 52: opponent's last selected number (or -1 if none)
        self.observation_space = spaces.Box(
            low=np.array([0] * 50 + [0.0, 0.0, -1.0]),
            high=np.array([1] * 50 + [200.0, 200.0, 50.0]),
            shape=(53,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the number pool
        self.number_pool = np.ones(
            50, dtype=np.float32
        )  # 1 for available, 0 for selected

        # Initialize player scores
        self.player_scores = [0, 0]

        # Initialize last selected numbers
        self.last_selected_numbers = [-1, -1]  # -1 indicates no number selected yet

        # Set current player: 0 or 1
        self.current_player = 0

        # Keep track of the last player who picked a number
        self.last_player_to_pick = -1  # -1 indicates no player has picked yet

        self.done = False

        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self._get_obs(), -10.0, True, False, {}

        opponent_index = 1 - self.current_player

        if action == 50:
            # Player chooses to pass
            # Switch to opponent
            self.current_player = opponent_index

            # Check endgame condition
            reward = self._check_endgame()
            if reward is not None:
                return self._get_obs(), reward, True, False, {}

            return self._get_obs(), 0.0, False, False, {}

        # Player selects a number
        selected_number = action + 1  # Numbers from 1 to 50

        # Remove number from pool
        self.number_pool[action] = 0

        # Add number to player's score
        self.player_scores[self.current_player] += selected_number

        # Update last selected number
        self.last_selected_numbers[self.current_player] = selected_number

        # Update last player to pick
        self.last_player_to_pick = self.current_player

        # Check for winning condition
        if self.player_scores[self.current_player] >= 100:
            # Current player wins
            self.done = True
            return self._get_obs(), 1.0, True, False, {}

        # Switch to the other player
        self.current_player = opponent_index

        # Check endgame condition
        reward = self._check_endgame()
        if reward is not None:
            return self._get_obs(), reward, True, False, {}

        return self._get_obs(), 0.0, False, False, {}

    def render(self):
        # Return a string representation of the game state
        pool_str = "Available Numbers: " + ", ".join(
            [str(i + 1) for i in range(50) if self.number_pool[i] == 1]
        )
        scores_str = f"Player 1 Score: {self.player_scores[0]}, Last Number: {self.last_selected_numbers[0]}"
        scores_str += f"\nPlayer 2 Score: {self.player_scores[1]}, Last Number: {self.last_selected_numbers[1]}"
        current_player_str = (
            f"Current Player: {'1' if self.current_player == 0 else '2'}"
        )
        return f"{pool_str}\n{scores_str}\n{current_player_str}"

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        return self._get_valid_actions_for_player(self.current_player)

    def _get_valid_actions_for_player(self, player_index):
        # Returns list of valid action indices for given player
        opponent_index = 1 - player_index
        opponent_last_number = self.last_selected_numbers[opponent_index]
        valid_actions = []
        available_numbers = np.where(self.number_pool == 1)[
            0
        ]  # Indices of available numbers

        if opponent_last_number == -1:
            # First turn, any available number is valid
            valid_actions = available_numbers.tolist()
        else:
            required_last_digit = opponent_last_number % 10
            for i in available_numbers:
                if (i + 1) % 10 == required_last_digit:
                    valid_actions.append(i)

        if not valid_actions:
            # No valid moves, player can pass
            valid_actions = [50]  # Action index for pass
        return valid_actions

    def _check_endgame(self):
        # Return reward if game has ended, otherwise return None
        opponent_index = 1 - self.current_player
        current_valid_actions = self._get_valid_actions_for_player(self.current_player)
        opponent_valid_actions = self._get_valid_actions_for_player(opponent_index)

        if current_valid_actions == [50] and opponent_valid_actions == [50]:
            # Neither player can make a move
            # Determine winner
            if (
                self.player_scores[self.current_player]
                > self.player_scores[opponent_index]
            ):
                # Current player wins
                self.done = True
                return 1.0
            elif (
                self.player_scores[self.current_player]
                < self.player_scores[opponent_index]
            ):
                # Opponent wins
                self.done = True
                return -1.0
            else:
                # Scores are equal; last player to make a selection wins
                if self.last_player_to_pick == self.current_player:
                    self.done = True
                    return 1.0
                else:
                    self.done = True
                    return -1.0
        else:
            return None  # Game not ended

    def _get_obs(self):
        # Build the observation array
        obs = np.zeros(53, dtype=np.float32)
        obs[0:50] = self.number_pool
        obs[50] = self.player_scores[self.current_player]
        obs[51] = self.player_scores[1 - self.current_player]
        obs[52] = self.last_selected_numbers[
            1 - self.current_player
        ]  # Opponent's last selected number
        return obs
