import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - Attack, 1 - Heal
        self.action_space = spaces.Discrete(2)

        # Define observation space:
        # [current_player_LP, opponent_LP, card_value]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1]), high=np.array([20, 20, 10]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize Life Points for both players
        self.player_life_points = [20, 20]  # [Player 1 LP, Player 2 LP]

        # Create and shuffle the deck
        self.deck = [i for i in range(1, 11) for _ in range(4)]
        random.shuffle(self.deck)

        # Initialize the discard pile
        self.discard_pile = []

        # Set the current player (0 - Player 1, 1 - Player 2)
        self.current_player = 0

        # Draw a card for the current player
        self.card_value = self.draw_card()

        # Game not over
        self.done = False

        # Prepare the initial observation
        observation = np.array(
            [
                self.player_life_points[self.current_player],
                self.player_life_points[1 - self.current_player],
                self.card_value,
            ],
            dtype=np.int32,
        )

        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                self.get_observation(),
                0,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Check if the action is valid
        if action not in [0, 1]:
            self.done = True
            return (
                self.get_observation(),
                -10,
                True,
                False,
                {},
            )  # Invalid action

        # Current player and opponent indices
        curr_player = self.current_player
        opp_player = 1 - self.current_player

        # Apply the action
        if action == 0:  # Attack
            self.player_life_points[opp_player] -= self.card_value
            # Ensure Life Points do not drop below zero
            if self.player_life_points[opp_player] < 0:
                self.player_life_points[opp_player] = self.player_life_points[
                    opp_player
                ]
        elif action == 1:  # Heal
            self.player_life_points[curr_player] += self.card_value
            # Life Points cannot exceed 20
            if self.player_life_points[curr_player] > 20:
                self.player_life_points[curr_player] = 20

        # Discard the used card
        self.discard_pile.append(self.card_value)

        # Check for win condition
        if self.player_life_points[opp_player] <= 0:
            # Current player wins
            self.done = True
            reward = 1  # Win reward
            return (
                self.get_observation(),
                reward,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Penalize for valid move
        reward = -10

        # Switch current player
        self.current_player = opp_player

        # Draw a card for the new current player
        self.card_value = self.draw_card()

        # Prepare the new observation
        observation = self.get_observation()

        return (
            observation,
            reward,
            False,
            False,
            {},
        )  # obs, reward, terminated, truncated, info

    def render(self):
        current_state = f"Player {self.current_player + 1}'s Turn\n"
        current_state += (
            f"Your Life Points: {self.player_life_points[self.current_player]}\n"
        )
        current_state += f"Opponent's Life Points: {self.player_life_points[1 - self.current_player]}\n"
        current_state += f"You drew a {self.card_value}.\n"
        return current_state

    def valid_moves(self):
        # Both actions are always valid
        return [0, 1]

    def draw_card(self):
        # If the deck is empty, shuffle the discard pile into the deck
        if not self.deck:
            self.deck = self.discard_pile.copy()
            self.discard_pile = []
            random.shuffle(self.deck)

        # Draw the top card
        return self.deck.pop()

    def get_observation(self):
        observation = np.array(
            [
                self.player_life_points[self.current_player],
                self.player_life_points[1 - self.current_player],
                self.card_value,
            ],
            dtype=np.int32,
        )
        return observation
