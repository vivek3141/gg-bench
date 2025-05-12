import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Numbers from 1 to 9
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)
        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observation = np.zeros(
            9, dtype=np.int8
        )  # Numbers 1 to 9; 0=available, 1=player1, -1=player2
        self.current_player = 1  # Start with player 1
        self.done = False
        return self.observation, {}

    def step(self, action):
        if self.done:
            return self.observation, 0, True, False, {}

        if self.observation[action] != 0:
            # Invalid move
            self.done = True
            return self.observation, -10, True, False, {}

        # Valid move
        self.observation[action] = self.current_player

        # Check for victory
        victory = False
        player_numbers = [
            i + 1 for i in range(9) if self.observation[i] == self.current_player
        ]
        if len(player_numbers) >= 3:
            # Check all combinations of 3 numbers
            for combo in combinations(player_numbers, 3):
                if self.is_arithmetic_sequence(combo):
                    victory = True
                    break

        if victory:
            self.done = True
            return self.observation, 1, True, False, {}  # Current player wins

        # Check if all numbers have been picked
        if np.all(self.observation != 0):
            self.done = True  # Game over
            # Compute sums
            player1_numbers = [i + 1 for i in range(9) if self.observation[i] == 1]
            player2_numbers = [i + 1 for i in range(9) if self.observation[i] == -1]
            player1_sum = sum(player1_numbers)
            player2_sum = sum(player2_numbers)
            if player1_sum > player2_sum:
                winner = 1
            elif player2_sum > player1_sum:
                winner = -1
            else:
                # Sums are equal; player who picked second (the player who took the last turn) wins
                winner = self.current_player  # Player who just moved
            reward = 1 if winner == self.current_player else 0
            return self.observation, reward, True, False, {}

        # Switch players
        self.current_player *= -1

        return self.observation, 0, False, False, {}

    def render(self):
        shared_pool = [i + 1 for i in range(9) if self.observation[i] == 0]
        player1_hand = [i + 1 for i in range(9) if self.observation[i] == 1]
        player2_hand = [i + 1 for i in range(9) if self.observation[i] == -1]

        output = "Shared Pool: " + " ".join(map(str, shared_pool)) + "\n"
        output += "Player 1's Hand: " + " ".join(map(str, player1_hand)) + "\n"
        output += "Player 2's Hand: " + " ".join(map(str, player2_hand)) + "\n"
        output += f"Player {1 if self.current_player == 1 else 2}'s turn.\n"
        print(output)

    def valid_moves(self):
        return [i for i in range(9) if self.observation[i] == 0]

    def is_arithmetic_sequence(self, numbers):
        numbers = sorted(numbers)
        return (numbers[1] - numbers[0]) == (numbers[2] - numbers[1])
