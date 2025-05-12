import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Numbers from 2 to 20 inclusive
        self.numbers_list = list(range(2, 21))
        self.num_numbers = len(self.numbers_list)  # Should be 19

        # Define action and observation space
        self.action_space = spaces.Discrete(self.num_numbers)

        # Observation space:
        # 19 for number pool status
        # 1 for last opponent's chosen number (normalized)
        # 2 for players' scores (normalized)
        # 1 for current player indicator (0 or 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(23,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the number pool (1 means available, 0 means picked)
        self.number_pool_status = np.ones(self.num_numbers, dtype=np.float32)

        # Initialize players' scores
        self.players_scores = [0.0, 0.0]

        # Initialize last numbers chosen by each player
        # -1 indicates no number has been chosen yet
        self.last_numbers = [-1, -1]

        # Player indices: 0 and 1
        # Current player: 0 (Player 1)
        self.current_player = 0

        # Game not terminated
        self.done = False

        # Build initial observation
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        if action < 0 or action >= self.num_numbers:
            # Invalid action index
            return self._get_observation(), -10.0, True, False, {}

        num = self.numbers_list[action]

        # Check if the number is available in the pool
        if self.number_pool_status[action] == 0:
            # Number already picked
            return self._get_observation(), -10.0, True, False, {}

        # Check if the action is valid
        opponent_last_num = self.last_numbers[1 - self.current_player]
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            return self._get_observation(), -10.0, True, False, {}

        # Valid move, update game state
        self.number_pool_status[action] = 0  # Remove number from pool
        self.players_scores[self.current_player] += num  # Update score
        self.last_numbers[self.current_player] = num  # Update last chosen number

        # Check for win condition
        if self.players_scores[self.current_player] >= 50:
            self.done = True
            return self._get_observation(), 1.0, True, False, {}

        # Check if both players cannot move
        next_player_valid_moves = self._check_next_player_valid_moves()
        if not next_player_valid_moves and not self.valid_moves():
            # Neither player can move, determine winner
            if self.players_scores[0] > self.players_scores[1]:
                # Player 1 wins
                self.done = True
                return (
                    self._get_observation(),
                    1.0 if self.current_player == 0 else 0.0,
                    True,
                    False,
                    {},
                )
            elif self.players_scores[1] > self.players_scores[0]:
                # Player 2 wins
                self.done = True
                return (
                    self._get_observation(),
                    1.0 if self.current_player == 1 else 0.0,
                    True,
                    False,
                    {},
                )
            else:
                # Tie game continues
                pass

        # Switch current player
        self.current_player = 1 - self.current_player

        # Return observation with zero reward
        return self._get_observation(), 0.0, False, False, {}

    def render(self):
        # Return a string representation of the current game state
        pool_numbers = [
            str(self.numbers_list[i])
            for i in range(self.num_numbers)
            if self.number_pool_status[i] == 1
        ]
        pool_str = ", ".join(pool_numbers)
        player1_score = self.players_scores[0]
        player2_score = self.players_scores[1]
        last_num_p1 = self.last_numbers[0]
        last_num_p2 = self.last_numbers[1]
        current_player = self.current_player + 1  # For display purposes
        render_str = (
            f"Current Scores - Player 1: {player1_score}, Player 2: {player2_score}\n"
        )
        render_str += f"Available Numbers: [{pool_str}]\n"
        render_str += f"Last Number Chosen by Player 1: {last_num_p1 if last_num_p1 != -1 else 'None'}\n"
        render_str += f"Last Number Chosen by Player 2: {last_num_p2 if last_num_p2 != -1 else 'None'}\n"
        render_str += f"Player {current_player}'s turn.\n"
        return render_str

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        opponent_last_num = self.last_numbers[1 - self.current_player]
        valid_actions = []

        # If it's the first move or opponent hasn't chosen a number, any available number is valid
        if opponent_last_num == -1:
            valid_actions = [
                i for i in range(self.num_numbers) if self.number_pool_status[i] == 1
            ]
        else:
            for i in range(self.num_numbers):
                if self.number_pool_status[i] == 1:
                    num = self.numbers_list[i]
                    if num % opponent_last_num == 0 or opponent_last_num % num == 0:
                        valid_actions.append(i)
        return valid_actions

    def _check_next_player_valid_moves(self):
        # Check if the next player has valid moves
        next_player = 1 - self.current_player
        opponent_last_num = self.last_numbers[self.current_player]
        valid_actions = []

        # If opponent hasn't chosen a number, any available number is valid
        if opponent_last_num == -1:
            valid_actions = [
                i for i in range(self.num_numbers) if self.number_pool_status[i] == 1
            ]
        else:
            for i in range(self.num_numbers):
                if self.number_pool_status[i] == 1:
                    num = self.numbers_list[i]
                    if num % opponent_last_num == 0 or opponent_last_num % num == 0:
                        valid_actions.append(i)
        return valid_actions

    def _get_observation(self):
        # Construct the observation vector
        # Number pool status (19 elements)
        obs = self.number_pool_status.copy()

        # Last opponent's chosen number (normalized between 0 and 1)
        opponent_last_num = self.last_numbers[1 - self.current_player]
        if opponent_last_num == -1:
            last_num_normalized = 0.0
        else:
            last_num_normalized = opponent_last_num / 20.0  # Normalize between 0 and 1
        obs = np.append(obs, [last_num_normalized])

        # Players' scores (normalized between 0 and 1)
        max_score = 70.0  # Maximum possible score (arbitrary upper bound)
        p1_score_normalized = self.players_scores[0] / max_score
        p2_score_normalized = self.players_scores[1] / max_score
        obs = np.append(obs, [p1_score_normalized, p2_score_normalized])

        # Current player indicator (0 or 1)
        obs = np.append(obs, [self.current_player])

        return obs.astype(np.float32)
