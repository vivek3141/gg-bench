import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Numbers from 2 to 10 inclusive (indices 0 to 8)
        self.action_space = spaces.Discrete(9)

        # Observation space:
        # - Available numbers: 9 elements (1 if available, 0 if not)
        # - Player totals: 2 elements (totals of player 0 and player 1)
        # - Last opponent's chosen number: 1 element
        self.observation_space = spaces.Box(
            low=0, high=30, shape=(12,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the game state
        self.available_numbers = np.ones(
            9, dtype=np.float32
        )  # Numbers 2-10 all available
        self.player_totals = np.zeros(2, dtype=np.float32)
        self.current_player = 0  # Starting player is 0
        self.last_moves = [None, None]  # Last numbers chosen by players
        self.done = False

        observation = self._get_observation()
        info = {}
        return observation, info  # Return observation and info

    def _get_observation(self):
        # Return the observation as an array of shape (12,)
        # Includes available_numbers, player_totals, last opponent's number
        last_opponent_number = 0
        opponent = 1 - self.current_player
        if self.last_moves[opponent] is not None:
            last_opponent_number = self.last_moves[opponent]

        observation = np.concatenate(
            (self.available_numbers, self.player_totals, [last_opponent_number])
        )
        return observation

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done, False, {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return self._get_observation(), reward, self.done, False, {}

        # Map action index to the actual number
        number = self._action_to_number(action)

        # Update the state
        self.available_numbers[action] = 0  # Number is now unavailable
        self.last_moves[self.current_player] = number
        self.player_totals[self.current_player] += number

        # Check for win condition
        if self.player_totals[self.current_player] >= 30:
            self.done = True
            reward = 1
            return self._get_observation(), reward, self.done, False, {}

        # Check if opponent can move
        opponent = 1 - self.current_player
        # Temporarily set current_player to opponent to use valid_moves()
        self.current_player = opponent
        opponent_valid_moves = self.valid_moves()
        self.current_player = 1 - opponent  # Switch back
        if not opponent_valid_moves:
            # Opponent cannot move, current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, self.done, False, {}

        # Switch to next player
        self.current_player = opponent
        reward = 0
        return self._get_observation(), reward, self.done, False, {}

    def render(self):
        output = "Numbers available: "
        numbers_available = [
            str(self._action_to_number(i))
            for i in range(9)
            if self.available_numbers[i] == 1
        ]
        output += ", ".join(numbers_available) + "\n"
        output += f"Player totals: Player 0: {self.player_totals[0]}, Player 1: {self.player_totals[1]}\n"
        output += f"Current player: Player {self.current_player}\n"
        output += f"Last moves: Player 0: {self.last_moves[0]}, Player 1: {self.last_moves[1]}"
        return output

    def valid_moves(self):
        # Return a list of valid actions for the current player
        if self.done:
            return []

        available_actions = [i for i in range(9) if self.available_numbers[i] == 1]
        opponent = 1 - self.current_player
        if self.last_moves[opponent] is None:
            # First move can be any available number
            return available_actions

        last_opponent_number = self.last_moves[opponent]

        valid_actions = []
        for idx in available_actions:
            number = self._action_to_number(idx)
            if number % last_opponent_number == 0 or last_opponent_number % number == 0:
                valid_actions.append(idx)
        return valid_actions

    def _action_to_number(self, action):
        # Map action index to number (indices 0-8 correspond to numbers 2-10)
        return action + 2
