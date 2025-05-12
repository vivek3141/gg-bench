import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 (Add card), 1 (Discard card)
        self.action_space = spaces.Discrete(2)

        # Observation space consists of:
        # - Player's sequence (5 numbers)
        # - Opponent's sequence (5 numbers)
        # - Discard histogram (counts of numbers 1-9)
        # - Current drawn card
        # Total size: 5 + 5 + 9 + 1 = 20
        self.observation_space = spaces.Box(low=0, high=9, shape=(20,), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create and shuffle the deck
        self.deck = [i for i in range(1, 10)] * 2  # Numbers 1-9, two copies each
        self.np_random.shuffle(self.deck)  # Use environment's random generator

        # Initialize player sequences and discard pile
        self.player_sequences = [[], []]
        self.discard_pile = []
        self.current_player = 0  # Player 1 starts (index 0)
        self.current_card = None  # No card drawn yet
        self.done = False  # Game is not over

        # Return the initial observation
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        terminated = False
        truncated = False

        # Draw a card if not already drawn
        if self.current_card is None:
            if len(self.deck) == 0:
                # Deck is empty, check for winner
                self.done = True
                terminated = True
                reward = self._check_winner()
                return self._get_observation(), reward, terminated, truncated, {}
            else:
                self.current_card = self.deck.pop(0)  # Draw the top card

        player_seq = self.player_sequences[self.current_player]
        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid action
            self.done = True
            terminated = True
            return self._get_observation(), -10, terminated, truncated, {}

        if action == 0:  # Add the card
            if len(player_seq) == 0 or self.current_card > player_seq[-1]:
                player_seq.append(self.current_card)
            else:
                # Cannot add the card
                self.done = True
                terminated = True
                return self._get_observation(), -10, terminated, truncated, {}
        elif action == 1:  # Discard the card
            self.discard_pile.append(self.current_card)
        else:
            # Invalid action
            self.done = True
            terminated = True
            return self._get_observation(), -10, terminated, truncated, {}

        self.current_card = None  # Reset current card

        # Check for winning condition
        if len(player_seq) >= 5:
            # Current player wins
            self.done = True
            terminated = True
            return self._get_observation(), 1, terminated, truncated, {}

        # Check if deck is empty after the move
        if len(self.deck) == 0:
            # Game ends, determine winner
            self.done = True
            terminated = True
            reward = self._check_winner()
            return self._get_observation(), reward, terminated, truncated, {}

        # Switch to the next player
        self.current_player = 1 - self.current_player

        # Return observation and zero reward
        return self._get_observation(), 0, False, False, {}

    def render(self):
        output = []
        output.append(f"Player {self.current_player + 1}'s turn.")
        output.append(f"Player 1's Sequence: {self.player_sequences[0]}")
        output.append(f"Player 2's Sequence: {self.player_sequences[1]}")
        output.append(f"Discard Pile: {self.discard_pile}")
        output.append(f"Deck has {len(self.deck)} cards remaining.")
        if self.current_card is not None:
            output.append(f"Current card drawn: {self.current_card}")
        return "\n".join(output)

    def valid_moves(self):
        if self.current_card is None:
            return []  # No valid moves if no card is drawn

        player_seq = self.player_sequences[self.current_player]
        if len(player_seq) == 0 or self.current_card > player_seq[-1]:
            # Can choose to add or discard
            return [0, 1]
        else:
            # Must discard
            return [1]

    def _get_observation(self):
        # Pad sequences to length 5 with zeros
        own_seq = self.player_sequences[self.current_player] + [0] * (
            5 - len(self.player_sequences[self.current_player])
        )
        opp_seq = self.player_sequences[1 - self.current_player] + [0] * (
            5 - len(self.player_sequences[1 - self.current_player])
        )

        # Create discard histogram
        discard_hist = [0] * 9
        for card in self.discard_pile:
            discard_hist[card - 1] += 1

        # Current drawn card
        current_card = self.current_card if self.current_card is not None else 0

        # Combine into a single observation array
        observation = np.array(
            own_seq + opp_seq + discard_hist + [current_card], dtype=np.int32
        )
        return observation

    def _check_winner(self):
        # Determine the winner when the deck is empty
        len0 = len(self.player_sequences[0])
        len1 = len(self.player_sequences[1])
        if len0 > len1:
            if self.current_player == 0:
                return 1  # Current player wins
            else:
                return -1  # Current player loses
        elif len1 > len0:
            if self.current_player == 1:
                return 1  # Current player wins
            else:
                return -1  # Current player loses
        else:
            # Sequences are equal in length
            last0 = self.player_sequences[0][-1] if len0 > 0 else 0
            last1 = self.player_sequences[1][-1] if len1 > 0 else 0
            if last0 > last1:
                if self.current_player == 0:
                    return 1  # Current player wins
                else:
                    return -1
            elif last1 > last0:
                if self.current_player == 1:
                    return 1  # Current player wins
                else:
                    return -1
            else:
                # Tie-breaker: player who went second loses
                if self.current_player == 1:
                    return -1  # Current player loses
                else:
                    return 1  # Current player wins
        return 0  # Should not reach here as per game rules
