import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the sequence range
        self.max_num = 21  # Numbers from 1 to 21
        self.initial_sequence = list(range(1, self.max_num + 1))
        self.max_score = sum(self.initial_sequence)

        # Define action and observation space
        # Actions: 0 (pick first number), 1 (pick last number)
        self.action_space = spaces.Discrete(2)

        # Observation space:
        # [first_number, last_number, player1_score, player2_score]
        self.observation_space = spaces.Box(
            low=0, high=self.max_score, shape=(4,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = self.initial_sequence.copy()
        self.player_scores = [0, 0]  # [Player 0 score, Player 1 score]
        self.current_player = 0  # Player 0 starts
        self.done = False

        return self._get_observation(), {}

    def step(self, action):
        # Check if the game has already ended
        if self.done:
            return self._get_observation(), -10, True, False, {}

        # Check for invalid action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Perform the action
        if action == 0:
            chosen_number = self.sequence.pop(0)  # Pick first number
        elif action == 1:
            chosen_number = self.sequence.pop(-1)  # Pick last number

        # Update player's score
        self.player_scores[self.current_player] += chosen_number

        # Check if the game has ended
        if not self.sequence:
            self.done = True
            player_score = self.player_scores[self.current_player]
            opponent_score = self.player_scores[1 - self.current_player]

            if player_score > opponent_score:
                reward = 1
            else:
                reward = 0

            return self._get_observation(), reward, True, False, {}
        else:
            # Switch to the next player
            self.current_player = 1 - self.current_player
            return self._get_observation(), 0, False, False, {}

    def render(self):
        sequence_str = " ".join(map(str, self.sequence))
        return (
            f"Current Sequence: {sequence_str}\n"
            f"Player 1's Score: {self.player_scores[0]}\n"
            f"Player 2's Score: {self.player_scores[1]}\n"
            f"Current Player: Player {self.current_player + 1}\n"
        )

    def valid_moves(self):
        if self.done or not self.sequence:
            return []
        else:
            return [0, 1]

    def _get_observation(self):
        if self.sequence:
            first_number = self.sequence[0]
            last_number = self.sequence[-1]
        else:
            first_number = 0
            last_number = 0

        player1_score = self.player_scores[0]
        player2_score = self.player_scores[1]

        observation = np.array(
            [first_number, last_number, player1_score, player2_score], dtype=np.int32
        )

        return observation
