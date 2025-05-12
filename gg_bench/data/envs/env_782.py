import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to combinations of secret move and opponent's guess
        # 0: secret_move=1, opponent_guess=1
        # 1: secret_move=1, opponent_guess=2
        # 2: secret_move=2, opponent_guess=1
        # 3: secret_move=2, opponent_guess=2
        self.action_space = spaces.Discrete(4)

        # Observation space includes positions of both players and current player
        # observation[0]: current player's position (0-10)
        # observation[1]: opponent's position (0-10)
        # observation[2]: current player (1 or 2)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1]),
            high=np.array([10, 10, 2]),
            shape=(3,),
            dtype=np.int32,
        )

        # Internal state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = {1: 0, 2: 0}  # Positions of players
        self.current_player = 1
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info (dict)

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action to secret_move and opponent_guess
        action_to_move_guess = {
            0: (1, 1),
            1: (1, 2),
            2: (2, 1),
            3: (2, 2),
        }
        if action not in action_to_move_guess:
            # Invalid action
            return self._get_observation(), -10, True, False, {}

        secret_move, opponent_guess = action_to_move_guess[action]

        # Current positions
        current_position = self.positions[self.current_player]

        # Check if secret_move is valid
        if current_position + secret_move > 10:
            # Invalid move
            return self._get_observation(), -10, True, False, {}

        # Apply game rules
        if opponent_guess == secret_move:
            # Opponent guessed correctly, player does not move
            pass
        else:
            # Opponent guessed incorrectly, player moves forward
            self.positions[self.current_player] += secret_move

        # Check for win
        if self.positions[self.current_player] == 10:
            self.done = True
            reward = 1  # Current player wins
            return self._get_observation(), reward, True, False, {}
        else:
            # Switch current player
            self.current_player = 2 if self.current_player == 1 else 1
            reward = 0
            return self._get_observation(), reward, False, False, {}

    def render(self):
        current_pos = self.positions[self.current_player]
        opponent = 2 if self.current_player == 1 else 1
        opponent_pos = self.positions[opponent]
        return f"Player 1 Position: {self.positions[1]}, Player 2 Position: {self.positions[2]}, Current Player: {self.current_player}"

    def valid_moves(self):
        # Determine valid secret moves
        current_position = self.positions[self.current_player]
        valid_secret_moves = []
        if current_position + 1 <= 10:
            valid_secret_moves.append(1)
        if current_position + 2 <= 10:
            valid_secret_moves.append(2)

        # Generate valid actions based on valid secret moves
        valid_actions = []
        for secret_move in valid_secret_moves:
            for opponent_guess in [1, 2]:
                action = self._move_guess_to_action(secret_move, opponent_guess)
                valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Return observation: [current player's position, opponent's position, current_player]
        opponent = 2 if self.current_player == 1 else 1
        current_pos = self.positions[self.current_player]
        opponent_pos = self.positions[opponent]
        observation = np.array(
            [current_pos, opponent_pos, self.current_player], dtype=np.int32
        )
        return observation

    def _move_guess_to_action(self, secret_move, opponent_guess):
        # Helper function to map (secret_move, opponent_guess) to action index
        move_guess_to_action = {
            (1, 1): 0,
            (1, 2): 1,
            (2, 1): 2,
            (2, 2): 3,
        }
        return move_guess_to_action[(secret_move, opponent_guess)]
