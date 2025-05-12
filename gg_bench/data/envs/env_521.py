import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - 'No Action', 1-4 - Discard card at index 0-3 in hand
        self.action_space = spaces.Discrete(5)

        # Observation space: an array of shape (10,)
        # Each index corresponds to a card from -5 to +5 (excluding 0), mapped as:
        # - Index 0: -5, Index 1: -4, ..., Index 9: +5
        # Values:
        # 0 - Card is in deck or opponent's hand (unknown)
        # 1 - Card is in player's hand
        # 2 - Card is in discard pile
        self.observation_space = spaces.Box(low=0, high=2, shape=(10,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts first
        self.done = False

        # Initialize deck (-5 to +5 excluding 0)
        self.card_values = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        self.card_value_to_index = {
            val: idx for idx, val in enumerate(self.card_values)
        }
        self.deck = self.card_values.copy()
        np.random.shuffle(self.deck)  # Shuffle the deck

        # Hands and discard pile
        self.player_hands = {1: [], -1: []}
        self.discard_pile = []

        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        reward = 0  # Default reward for valid move

        # Draw phase
        if len(self.deck) == 0:
            # Shuffle the discard pile to form new deck
            self.deck = self.discard_pile.copy()
            self.discard_pile = []
            np.random.shuffle(self.deck)

        # Draw the top card
        drawn_card = self.deck.pop()
        self.player_hands[self.current_player].append(drawn_card)

        # Hand limit check
        need_discard = len(self.player_hands[self.current_player]) > 3

        if need_discard:
            # Discard phase
            valid_actions = [1, 2, 3, 4][: len(self.player_hands[self.current_player])]
            if action not in valid_actions:
                # Invalid action
                reward = -10
                self.done = True
                return self._get_observation(), reward, True, False, {}
            else:
                discard_index = action - 1  # Adjust for action indexing
                # Discard the chosen card
                discarded_card = self.player_hands[self.current_player].pop(
                    discard_index
                )
                self.discard_pile.append(discarded_card)
        else:
            # No discard needed
            if action != 0:
                # Invalid action
                reward = -10
                self.done = True
                return self._get_observation(), reward, True, False, {}
            # Else action == 0, no action needed

        # Hand evaluation
        hand_sum = sum(self.player_hands[self.current_player])
        if hand_sum == 0:
            # Current player wins
            reward = 1
            self.done = True
            self._update_observation()
            return self._get_observation(), reward, True, False, {}
        else:
            # Game continues
            self._update_observation()
            # Switch turns
            self.current_player *= -1
            return self._get_observation(), reward, False, False, {}

    def render(self):
        # Build a string representation of the state
        s = f"Current player: {'Player 1' if self.current_player ==1 else 'Player 2'}\n"
        s += f"Your hand: {self.player_hands[self.current_player]}\n"
        s += f"Sum of your hand: {sum(self.player_hands[self.current_player])}\n"
        s += f"Discard pile: {self.discard_pile}\n"
        s += f"Cards remaining in deck: {len(self.deck)}\n"
        return s

    def valid_moves(self):
        if self.done:
            return []
        need_discard = len(self.player_hands[self.current_player]) > 3
        if need_discard:
            # Valid actions are discarding one of the cards in hand
            return [1, 2, 3, 4][: len(self.player_hands[self.current_player])]
        else:
            # Valid action is 0 (No action needed)
            return [0]

    def _get_observation(self):
        # Reset observation
        observation = np.zeros(10, dtype=np.int8)
        # Update for current player's hand
        for card in self.player_hands[self.current_player]:
            idx = self.card_value_to_index[card]
            observation[idx] = 1
        # Update for discard pile
        for card in self.discard_pile:
            idx = self.card_value_to_index[card]
            observation[idx] = 2
        return observation

    def _update_observation(self):
        # This method doesn't need to do anything because observation is generated fresh each time
        pass
