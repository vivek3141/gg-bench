import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = Roll, 1 = Pass
        self.action_space = spaces.Discrete(2)
        # Observation: [current_player_score, opponent_player_score]
        self.observation_space = spaces.Box(low=0, high=20, shape=(2,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.scores = [0, 0]  # Scores for player 0 and player 1
        self.current_player = 0
        self.turn_over = False
        self.done = False
        return np.array(self._get_obs()), {}

    def step(self, action):
        if self.done:
            return np.array(self._get_obs()), 0, True, False, {}

        if action not in [0, 1]:
            # Invalid action
            return np.array(self._get_obs()), -10, True, False, {}

        if self.turn_over:
            # Turn has ended; invalid action
            return np.array(self._get_obs()), -10, True, False, {}

        reward = -10  # Default reward for a valid move

        if action == 1:
            # Player chooses to Pass
            self.turn_over = True
        elif action == 0:
            # Player chooses to Roll
            die_roll = np.random.randint(1, 7)  # Roll a die (1-6)
            self.scores[self.current_player] += die_roll

            if self.scores[self.current_player] > 20:
                # Exceeded 20; reset score to 0 and end turn
                self.scores[self.current_player] = 0
                self.turn_over = True
            else:
                if die_roll % 2 == 0:
                    # Rolled even number; turn ends
                    self.turn_over = True

            if self.scores[self.current_player] == 20:
                # Current player wins
                reward = 1
                self.done = True
                return np.array(self._get_obs()), reward, True, False, {}

        # Switch to next player if turn is over and game is not done
        if self.turn_over and not self.done:
            self.current_player = 1 - self.current_player
            self.turn_over = False

        return np.array(self._get_obs()), reward, self.done, False, {}

    def render(self):
        state = f"Player {self.current_player + 1}'s turn.\n"
        state += f"Scores: Player 1: {self.scores[0]}, Player 2: {self.scores[1]}\n"
        return state

    def valid_moves(self):
        if self.done or self.turn_over:
            return []
        else:
            return [0, 1]

    def _get_obs(self):
        # Observation is from the current player's perspective
        return [self.scores[self.current_player], self.scores[1 - self.current_player]]
