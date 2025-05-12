import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Game parameters
        self.master_element = 15  # Target value to achieve

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pool = list(range(1, 10))  # Elements from 1 to 9
        self.hands = {1: [], -1: []}  # Player 1 and Player -1 hands
        self.current_player = 1
        self.done = False
        self.truncated = False

        # Observation: 0 if in pool, 1 if in current player's hand, -1 if in opponent's hand
        self.state = np.zeros(9, dtype=np.int8)

        return self.state, {}  # Observation and info

    def step(self, action):
        if self.done:
            return self.state, 0, True, False, {}

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self.state, -10, True, False, {}  # Invalid move

        # Select element from pool
        available_elements = sorted(self.pool)
        selected_element = available_elements[action]
        self.pool.remove(selected_element)
        self.hands[self.current_player].append(selected_element)

        # Update state
        idx = selected_element - 1  # Elements are 1-indexed
        self.state[idx] = self.current_player

        # Check for victory
        player_hand = self.hands[self.current_player]
        if self.check_victory(player_hand):
            self.done = True
            return self.state, 1, True, False, {}  # Current player wins

        # Check for game over (pool exhausted)
        if not self.pool:
            self.done = True
            # Determine winner based on total sums
            current_total = sum(self.hands[self.current_player])
            opponent_total = sum(self.hands[-self.current_player])
            if current_total < opponent_total:
                return self.state, 1, True, False, {}  # Current player wins
            elif current_total > opponent_total:
                return self.state, -1, True, False, {}  # Current player loses
            else:
                return self.state, 0, True, False, {}  # Draw

        # Switch to next player
        self.current_player *= -1

        return self.state, 0, False, False, {}  # Continue game

    def render(self):
        representation = "Current Player: {}\n".format(
            "Player 1" if self.current_player == 1 else "Player 2"
        )
        representation += "Master Element to achieve: {}\n".format(self.master_element)
        representation += "Available Elements: {}\n".format(
            " ".join(map(str, sorted(self.pool)))
        )
        representation += "Player 1's Hand: {}\n".format(
            " ".join(map(str, sorted(self.hands[1])))
        )
        representation += "Player 2's Hand: {}\n".format(
            " ".join(map(str, sorted(self.hands[-1])))
        )
        return representation

    def valid_moves(self):
        return list(range(len(self.pool)))

    def check_victory(self, hand):
        # Check all combinations of the player's hand for sums equal to the master element
        for r in range(1, len(hand) + 1):
            for combo in combinations(hand, r):
                if sum(combo) == self.master_element:
                    return True
        return False
