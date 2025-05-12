import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 0 - Attack, 1 - Heal
        self.action_space = spaces.Discrete(2)

        # Observation space:
        # Index 0: Player 1 LP (0-20)
        # Index 1: Player 2 LP (0-20)
        # Indices 2-11: Deck state for numbers 1-10 (1 if in deck, 0 if used)
        self.observation_space = spaces.Box(low=0, high=20, shape=(12,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_LP = 20
        self.player2_LP = 20
        self.current_player = 1

        # Initialize the deck and discard pile
        self.deck_numbers = list(range(1, 11))  # Numbers 1 through 10
        np.random.shuffle(self.deck_numbers)
        self.discard_pile = []

        # Deck state: 1 if number is in deck, 0 if not
        self.deck_state = np.ones(
            10, dtype=np.int32
        )  # Indices 0-9 correspond to numbers 1-10

        self.done = False

        return self.get_observation(), {}  # Return observation and info

    def get_observation(self):
        obs = np.concatenate(
            (
                np.array([self.player1_LP, self.player2_LP], dtype=np.int32),
                self.deck_state,
            )
        )
        return obs

    def step(self, action):
        if self.done:
            return (
                self.get_observation(),
                0,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        if action not in [0, 1]:
            self.done = True
            return (
                self.get_observation(),
                -10,
                True,
                False,
                {},
            )  # Invalid action, reward -10

        # Check if deck is empty, reshuffle if needed
        if len(self.deck_numbers) == 0:
            self.deck_numbers = self.discard_pile.copy()
            np.random.shuffle(self.deck_numbers)
            self.discard_pile = []
            self.deck_state = np.ones(10, dtype=np.int32)

        # Draw Phase
        drawn_number = self.deck_numbers.pop(0)
        self.deck_state[drawn_number - 1] = 0  # Update deck state

        # Identify current player and opponent
        if self.current_player == 1:
            current_player_LP = self.player1_LP
            opponent_LP = self.player2_LP
        else:
            current_player_LP = self.player2_LP
            opponent_LP = self.player1_LP

        # Action Phase
        if action == 0:  # Attack
            opponent_LP -= drawn_number
        elif action == 1:  # Heal
            heal_amount = drawn_number // 2
            current_player_LP += heal_amount
            if current_player_LP > 20:
                current_player_LP = 20  # LP cannot exceed 20

        # Discard Phase
        self.discard_pile.append(drawn_number)

        # Update LPs
        if self.current_player == 1:
            self.player1_LP = current_player_LP
            self.player2_LP = opponent_LP
        else:
            self.player2_LP = current_player_LP
            self.player1_LP = opponent_LP

        # Check for win condition
        if opponent_LP <= 0:
            self.done = True
            reward = 1
            return self.get_observation(), reward, True, False, {}
        else:
            reward = 0

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1

        return self.get_observation(), reward, self.done, False, {}

    def render(self):
        output = "Player 1 LP: {}\n".format(self.player1_LP)
        output += "Player 2 LP: {}\n".format(self.player2_LP)
        output += "Current Player: Player {}\n".format(self.current_player)
        output += "Deck State:\n"
        for i in range(10):
            status = "In Deck" if self.deck_state[i] == 1 else "Used"
            output += "Number {}: {}\n".format(i + 1, status)
        return output

    def valid_moves(self):
        # Valid moves are always [0, 1] (Attack or Heal)
        return [0, 1]
