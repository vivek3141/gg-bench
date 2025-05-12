import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            20
        )  # Actions from 0 to 19 corresponding to numbers 1 to 20
        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observation = np.zeros(
            20, dtype=np.int8
        )  # 0: unpicked, 1: picked by player 1, -1: picked by player 2
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False

        return self.observation.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.observation.copy(), 0, True, False, {}  # Game is already over

        # Check if action is within valid range
        if action < 0 or action >= 20:
            reward = -10
            self.done = True
            return self.observation.copy(), reward, True, False, {}

        number = action + 1  # Convert action to number (1-20)

        # Check if the number is available
        if self.observation[action] != 0:
            reward = -10
            self.done = True
            return self.observation.copy(), reward, True, False, {}

        # Check adjacency restriction
        opponent = -self.current_player
        opponent_picks = (
            np.where(self.observation == opponent)[0] + 1
        )  # Get numbers picked by opponent
        opponent_adjacent_numbers = set()
        for n in opponent_picks:
            adjacent = [n - 1, n + 1]
            opponent_adjacent_numbers.update(adjacent)
        opponent_adjacent_numbers = set(
            filter(lambda x: 1 <= x <= 20, opponent_adjacent_numbers)
        )

        if number in opponent_adjacent_numbers:
            reward = -10
            self.done = True
            return self.observation.copy(), reward, True, False, {}

        # Valid move; update the game state
        self.observation[action] = (
            self.current_player
        )  # Mark the number as picked by the current player

        # Check if all numbers have been picked
        if np.all(self.observation != 0):
            reward = 1  # Current player wins by picking the last number
            self.done = True
            return self.observation.copy(), reward, True, False, {}

        # Check if the next player has valid moves
        next_player = -self.current_player
        if not self.get_valid_moves(next_player):
            reward = 1  # Current player wins; opponent has no valid moves
            self.done = True
            return self.observation.copy(), reward, True, False, {}

        # Switch to the next player
        self.current_player = next_player
        reward = 0  # No reward for a regular valid move
        return self.observation.copy(), reward, False, False, {}

    def get_valid_moves(self, player):
        available_positions = np.where(self.observation == 0)[0]
        opponent = -player
        opponent_picks = (
            np.where(self.observation == opponent)[0] + 1
        )  # Numbers picked by opponent
        opponent_adjacent_numbers = set()
        for n in opponent_picks:
            adjacent = [n - 1, n + 1]
            opponent_adjacent_numbers.update(adjacent)
        opponent_adjacent_numbers = set(
            filter(lambda x: 1 <= x <= 20, opponent_adjacent_numbers)
        )
        valid_moves = []
        for pos in available_positions:
            number = pos + 1
            if number not in opponent_adjacent_numbers:
                valid_moves.append(pos)
        return valid_moves

    def render(self):
        # Visual representation of the game state
        output = "Available Numbers: "
        available_numbers = [str(i + 1) for i in range(20) if self.observation[i] == 0]
        output += ", ".join(available_numbers) + "\n"

        output += "Player 1 Picks: "
        player1_numbers = [str(i + 1) for i in range(20) if self.observation[i] == 1]
        output += ", ".join(sorted(player1_numbers, key=int)) + "\n"

        output += "Player 2 Picks: "
        player2_numbers = [str(i + 1) for i in range(20) if self.observation[i] == -1]
        output += ", ".join(sorted(player2_numbers, key=int)) + "\n"

        if not self.done:
            current_player_str = "Player 1" if self.current_player == 1 else "Player 2"
            output += f"Current Player: {current_player_str}\n"
        else:
            output += "Game Over\n"

        return output

    def valid_moves(self):
        if self.done:
            return []

        return self.get_valid_moves(self.current_player)
