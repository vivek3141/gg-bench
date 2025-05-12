import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 49 possible actions (selecting numbers 2 to 50)
        self.action_space = spaces.Discrete(49)

        # Observation space:
        # Indices 0-48: Number pool (1 if available, 0 if taken)
        # Index 49: Current player's current number (1 to 50)
        # Index 50: Opponent's current number (1 to 50)
        self.observation_space = spaces.Box(low=0, high=50, shape=(51,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the number pool: numbers 2 to 50 are available (represented by 1)
        self.number_pool = np.ones(
            49, dtype=np.int32
        )  # Indices 0 to 48 represent numbers 2 to 50

        # Both players start with current number 1
        self.current_numbers = {
            1: 1,  # Player 1's current number
            -1: 1,  # Player 2's current number
        }

        # Player 1 starts
        self.current_player = 1

        # Game is not done
        self.done = False

        # Prepare the initial observation
        observation = self._get_observation()

        return observation, {}  # Return observation and info

    def _get_observation(self):
        # Construct the observation array
        # Indices 0-48: Number pool
        # Index 49: Current player's current number
        # Index 50: Opponent's current number
        current_player_number = self.current_numbers[self.current_player]
        opponent_number = self.current_numbers[-self.current_player]
        observation = np.concatenate(
            [
                self.number_pool,
                np.array([current_player_number, opponent_number], dtype=np.int32),
            ]
        )
        return observation

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}

        selected_number = action + 2  # Map action index to number (2 to 50)

        # Check if selected number is available in the number pool
        if self.number_pool[action] == 0:
            self.done = True
            return self._get_observation(), -10, True, False, {}

        current_number = self.current_numbers[self.current_player]

        # Check if the move is valid
        valid = False

        if current_number == 1:
            valid = True  # Any number is valid when current number is 1
        else:
            if (
                selected_number > current_number
                and selected_number % current_number == 0
            ):
                valid = True

        if not valid:
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move, update the game state
        self.number_pool[action] = 0  # Remove the selected number from the pool
        self.current_numbers[self.current_player] = (
            selected_number  # Update current player's number
        )

        # Check if the opponent has any valid moves
        opponent_valid_moves = self._get_valid_moves_for_player(-self.current_player)
        if len(opponent_valid_moves) == 0:
            # Current player wins
            self.done = True
            reward = 1  # Reward for winning
            observation = self._get_observation()
            return observation, reward, True, False, {}
        else:
            # Switch to the next player
            self.current_player *= -1
            reward = -10  # Reward for a valid move
            observation = self._get_observation()
            return observation, reward, False, False, {}

    def _get_valid_moves_for_player(self, player):
        # Get the current number for the player
        current_number = self.current_numbers[player]

        # Find valid moves
        valid_moves = []
        for idx in range(49):  # Indices correspond to numbers 2 to 50
            if self.number_pool[idx] == 1:
                number = idx + 2
                if current_number == 1:
                    valid = True
                else:
                    if number > current_number and number % current_number == 0:
                        valid = True
                    else:
                        valid = False
                if valid:
                    valid_moves.append(idx)
        return valid_moves

    def valid_moves(self):
        return self._get_valid_moves_for_player(self.current_player)

    def render(self):
        # Generate a string representation of the game state
        number_pool_list = [
            str(idx + 2) for idx in range(49) if self.number_pool[idx] == 1
        ]
        number_pool_str = ", ".join(number_pool_list)
        current_player_number = self.current_numbers[self.current_player]
        opponent_number = self.current_numbers[-self.current_player]
        render_str = f"Current Number Pool: [{number_pool_str}]\n"
        render_str += f"Player {1 if self.current_player == 1 else 2}'s Current Number: {current_player_number}\n"
        render_str += f"Player {1 if self.current_player == -1 else 2}'s Current Number: {opponent_number}\n"
        render_str += (
            f"Player {1 if self.current_player == 1 else 2}, it's your turn.\n"
        )
        return render_str
