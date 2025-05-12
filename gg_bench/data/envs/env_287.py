import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 (Keep the card), 1 (Give the card)
        self.action_space = spaces.Discrete(2)

        # Define observation space
        # Observation consists of:
        # Index 0: Own stack total (0 to 50)
        # Index 1: Opponent's stack total (0 to 50)
        # Index 2: Cards remaining in deck (0 to 40)
        # Index 3: Drawn card value (0 to 10, where 0 means no card drawn)
        # Indexes 4-13: Counts of cards remaining for numbers 1 to 10 (each 0 to 4)
        self.observation_space = spaces.Box(
            low=np.array([0] * 14),
            high=np.array([50, 50, 40, 10] + [4] * 10),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the deck: four copies of each card numbered 1 to 10
        self.deck = [i for i in range(1, 11) for _ in range(4)]
        np.random.shuffle(self.deck)

        # Initialize player stacks
        self.player_stacks = {1: 0, -1: 0}  # Player 1 and Player 2 (-1)

        # Initialize counts of cards remaining
        self.cards_counts = {i: 4 for i in range(1, 11)}

        self.current_player = 1  # Player 1 starts
        self.drawn_card = 0  # No card drawn yet
        self.done = False

        # Return initial observation and info
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_obs(), -10, True, False, {}

        # If no card has been drawn yet, draw a card
        if self.drawn_card == 0:
            if len(self.deck) > 0:
                # Draw a card from the top of the deck
                self.drawn_card = self.deck.pop()
                self.cards_counts[self.drawn_card] -= 1
            else:
                # No cards left to draw, pass the turn
                self.current_player *= -1
                return self._get_obs(), -10, False, False, {}

            # After drawing a card, return observation to decide action
            return self._get_obs(), -10, False, False, {}

        # Check if the action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action, player loses
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Process the action
        if action == 0:
            # Keep the card
            self.player_stacks[self.current_player] += self.drawn_card
        elif action == 1:
            # Give the card to opponent
            self.player_stacks[-self.current_player] += self.drawn_card

        # Check win or lose conditions for both players
        reward = -10  # Default reward for a valid move
        if self.player_stacks[self.current_player] > 50:
            # Current player loses
            self.done = True
            return self._get_obs(), -10, True, False, {}
        elif self.player_stacks[self.current_player] == 50:
            # Current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}
        elif self.player_stacks[-self.current_player] > 50:
            # Opponent loses, current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}
        elif self.player_stacks[-self.current_player] == 50:
            # Opponent wins, current player loses
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # End of turn, reset the drawn card and switch player
        self.drawn_card = 0
        self.current_player *= -1

        return self._get_obs(), reward, False, False, {}

    def render(self):
        output = ""
        output += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        output += f"Your Stack Total: {self.player_stacks[self.current_player]}\n"
        output += (
            f"Opponent's Stack Total: {self.player_stacks[-self.current_player]}\n"
        )
        output += f"Cards Remaining in Deck: {len(self.deck)}\n"
        if self.drawn_card != 0:
            output += f"You have drawn a {self.drawn_card}\n"
        else:
            output += "No card drawn yet.\n"
        output += "Cards Counts Remaining:\n"
        for i in range(1, 11):
            output += f"Card {i}: {self.cards_counts[i]}\n"
        return output

    def valid_moves(self):
        # Returns a list of valid actions (0 or 1)
        if self.done or self.drawn_card == 0:
            return []

        valid_actions = []

        # Action 0: Keep the card
        if self.player_stacks[self.current_player] + self.drawn_card <= 50:
            valid_actions.append(0)

        # Action 1: Give the card to opponent
        if self.player_stacks[-self.current_player] + self.drawn_card <= 50:
            valid_actions.append(1)

        # If both actions would exceed 50, player must keep the card
        if not valid_actions:
            valid_actions = [0]

        return valid_actions

    def _get_obs(self):
        obs = np.zeros(14, dtype=np.int32)
        obs[0] = self.player_stacks[self.current_player]  # Own stack total
        obs[1] = self.player_stacks[-self.current_player]  # Opponent's stack total
        obs[2] = len(self.deck)  # Cards remaining in deck
        obs[3] = self.drawn_card  # Drawn card value
        # Counts of cards remaining for numbers 1 to 10
        for i in range(1, 11):
            obs[3 + i] = self.cards_counts[i]
        return obs
