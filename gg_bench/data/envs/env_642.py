import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 (add die roll to own score), 1 (add die roll to opponent's score)
        self.action_space = spaces.Discrete(2)

        # Observation space: [player1_score, player2_score, die_roll, current_player]
        # Scores can range from 25 to 56 (since they can temporarily exceed 50 before reset)
        # Die roll ranges from 1 to 6
        # Current player is 0 or 1
        self.observation_space = spaces.Box(
            low=np.array([25, 25, 1, 0]), high=np.array([56, 56, 6, 1]), dtype=np.int32
        )

        # Initialize environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize scores to 25 for both players
        self.scores = [25, 25]
        # Start with player 0
        self.current_player = 0
        self.done = False
        # Generate initial die roll
        self.die_roll = np.random.randint(1, 7)
        # Construct the initial observation
        observation = np.array(
            [self.scores[0], self.scores[1], self.die_roll, self.current_player],
            dtype=np.int32,
        )
        return observation, {}  # Return initial observation and empty info dict

    def step(self, action):
        # Check if the action is valid
        if not self.action_space.contains(action):
            # Invalid action
            observation = np.array(
                [self.scores[0], self.scores[1], self.die_roll, self.current_player],
                dtype=np.int32,
            )
            self.done = True
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        if self.done:
            # Game already over
            observation = np.array(
                [self.scores[0], self.scores[1], self.die_roll, self.current_player],
                dtype=np.int32,
            )
            return observation, -10, True, False, {}

        # Process the action
        if action == 0:
            # Add die roll to current player's own score
            self.scores[self.current_player] += self.die_roll
        else:
            # Add die roll to opponent's score
            opponent = 1 - self.current_player
            self.scores[opponent] += self.die_roll

        # Initialize reward and termination flags
        reward = 0
        terminated = False
        truncated = False

        # Check for win condition
        if self.scores[self.current_player] == 50:
            # Current player wins
            reward = 1
            self.done = True
            terminated = True
        elif self.scores[1 - self.current_player] == 50:
            # Opponent wins
            reward = -1
            self.done = True
            terminated = True
        else:
            # Check for score exceeding 50 and reset if necessary
            for i in [0, 1]:
                if self.scores[i] > 50:
                    self.scores[i] = 25

            # Switch current player for the next turn
            self.current_player = 1 - self.current_player
            # Generate new die roll for the next player
            self.die_roll = np.random.randint(1, 7)

        # Construct the new observation
        observation = np.array(
            [self.scores[0], self.scores[1], self.die_roll, self.current_player],
            dtype=np.int32,
        )

        return observation, reward, terminated, truncated, {}

    def render(self):
        # Create a string representation of the current game state
        render_str = (
            f"Player 0 Score: {self.scores[0]}, Player 1 Score: {self.scores[1]}\n"
        )
        render_str += (
            f"Current Player: {self.current_player}, Die Roll: {self.die_roll}\n"
        )
        return render_str

    def valid_moves(self):
        # Return valid moves: [0, 1] if game is ongoing, else empty list
        if self.done:
            return []
        else:
            return [0, 1]
