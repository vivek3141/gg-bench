import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is Discrete(9) representing numbers 1 to 9
        self.action_space = spaces.Discrete(9)

        # The observation space is a Box space representing:
        # - Numbers availability: values -1 (opponent), 0 (available), 1 (current player) for numbers 1 to 9
        # - Current player's total score
        # - Opponent's total score
        # The observation is an array of shape (11,)
        self.observation_space = spaces.Box(
            low=-1, high=45, shape=(11,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.numbers = np.zeros(
            9, dtype=np.int32
        )  # 0: available, 1: current player, -1: opponent
        self.player_scores = {1: 0, -1: 0}  # Players' total scores
        self.current_player = 1  # 1 for Player A, -1 for Player B
        self.done = False
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # Check if action is valid
        if action not in self.valid_moves():
            self.done = True
            reward = -10  # Penalty for invalid move
            observation = self._get_observation()
            return observation, reward, self.done, False, {}
        else:
            # Update the game state
            self.numbers[action] = self.current_player
            number_selected = action + 1  # Numbers are from 1 to 9
            self.player_scores[self.current_player] += number_selected

            # Check for win condition
            if self.player_scores[self.current_player] in [10, 20, 30]:
                self.done = True
                reward = 1  # Reward for winning
                observation = self._get_observation()
                return observation, reward, self.done, False, {}

            # Check if all numbers have been taken
            if np.all(self.numbers != 0):
                # Game ends, determine the winner
                current_score = self.player_scores[self.current_player]
                opponent_score = self.player_scores[-self.current_player]

                current_multiple = (current_score // 10) * 10
                opponent_multiple = (opponent_score // 10) * 10

                if current_multiple > opponent_multiple:
                    winner = self.current_player
                elif current_multiple < opponent_multiple:
                    winner = -self.current_player
                else:
                    winner = (
                        self.current_player
                    )  # Last player to have taken a turn wins

                self.done = True
                if winner == self.current_player:
                    reward = 1  # Reward for winning
                else:
                    reward = 0  # No reward if the opponent wins
                observation = self._get_observation()
                return observation, reward, self.done, False, {}

            # Continue the game
            reward = -10  # Penalty for each valid move to encourage faster wins
            self.current_player *= -1  # Switch players
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

    def render(self):
        output = "Available Numbers: "
        available_numbers = [str(i + 1) for i in range(9) if self.numbers[i] == 0]
        output += ", ".join(available_numbers) + "\n"
        output += f"Player A Total: {self.player_scores[1]}\n"
        output += f"Player B Total: {self.player_scores[-1]}\n"
        output += f"Current Player: {'A' if self.current_player == 1 else 'B'}\n"
        return output

    def valid_moves(self):
        return [i for i in range(9) if self.numbers[i] == 0]

    def _get_observation(self):
        # Observation from the perspective of the current player
        observation = np.zeros(11, dtype=np.int32)
        observation[:9] = self.numbers  # Numbers state
        observation[9] = self.player_scores[
            self.current_player
        ]  # Current player's score
        observation[10] = self.player_scores[-self.current_player]  # Opponent's score
        return observation
