import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 20 possible actions, numbers 1 to 20
        self.action_space = spaces.Discrete(20)

        # Observation space:
        # - First 20 elements: availability of numbers (1 or 0)
        # - Next 20 elements: one-hot encoding of the last opponent's number
        # - Last element: current player indicator (1 or -1)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(41,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_pool = np.ones(
            20, dtype=np.float32
        )  # Numbers 1 to 20 are available
        self.current_player = 1  # Player 1 starts
        self.last_numbers = {1: None, -1: None}  # Last numbers selected by each player
        self.done = False
        return self._get_observation(), {}  # Return observation and info

    def _get_observation(self):
        # Create the observation vector
        obs = np.zeros(41, dtype=np.float32)
        # Availability of numbers
        obs[0:20] = self.number_pool
        # One-hot encoding of the last opponent's number
        opponent = -self.current_player
        last_opponent_number = self.last_numbers.get(opponent)
        if last_opponent_number is not None:
            index = int(last_opponent_number) - 1
            obs[20 + index] = 1.0
        # Current player indicator
        obs[40] = self.current_player
        return obs

    def _valid_moves(self, player):
        # Return a list of valid actions for the specified player
        available_numbers = np.where(self.number_pool == 1)[
            0
        ]  # Indices of available numbers

        opponent = -player
        last_opponent_number = self.last_numbers.get(opponent)

        valid_actions = []
        if last_opponent_number is None:
            # First move: any available number can be chosen
            valid_actions = available_numbers.tolist()
        else:
            opponent_number = last_opponent_number
            for idx in available_numbers:
                number = idx + 1  # Numbers range from 1 to 20
                if opponent_number % number == 0 or number % opponent_number == 0:
                    valid_actions.append(idx)
        return valid_actions

    def valid_moves(self):
        # Return valid moves for the current player
        return self._valid_moves(self.current_player)

    def step(self, action):
        if self.done:
            # Game is over
            return self._get_observation(), 0, True, False, {}

        if action < 0 or action >= 20 or self.number_pool[action] == 0:
            # Invalid action: number not available
            self.done = True
            reward = -10  # Penalty for invalid move
            return self._get_observation(), reward, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move according to game rules
            self.done = True
            reward = -10  # Penalty for invalid move
            return self._get_observation(), reward, True, False, {}

        # Valid move
        number = action + 1  # Numbers range from 1 to 20
        self.number_pool[action] = 0  # Remove the number from the pool
        self.last_numbers[self.current_player] = (
            number  # Update last number for current player
        )

        # Check if the opponent has any valid moves
        opponent = -self.current_player
        opponent_valid_moves = self._valid_moves(opponent)
        if not opponent_valid_moves:
            # Opponent cannot make a move; current player wins
            self.done = True
            reward = 1  # Reward for winning
            return self._get_observation(), reward, True, False, {}

        # Switch to opponent
        self.current_player = opponent
        # Continue the game
        reward = 0  # No reward for a regular valid move
        return self._get_observation(), reward, False, False, {}

    def render(self):
        # Generate a string representation of the current game state
        available_numbers = [str(i + 1) for i in range(20) if self.number_pool[i] == 1]
        state_str = f"Available Numbers: {', '.join(available_numbers)}\n"
        opponent = -self.current_player
        last_opponent_number = self.last_numbers.get(opponent)
        if last_opponent_number is not None:
            state_str += f"Opponent's Last Number: {int(last_opponent_number)}\n"
        else:
            state_str += "No Opponent's Last Number.\n"
        state_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return state_str


import gymnasium as gym

env = CustomEnv()
observation, info = env.reset()
done = False

while not done:
    valid_actions = env.valid_moves()
    action = valid_actions[0]  # Replace with your action selection logic
    observation, reward, done, truncated, info = env.step(action)
    print(env.render())
    if reward == 1:
        print("Current player wins!")
    elif reward == -10:
        print("Invalid move. Current player loses.")
