import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Add drawn card to the beginning of sequence
        #          1 - Add drawn card to the end of sequence
        #          2 - Discard the drawn card
        self.action_space = spaces.Discrete(3)

        # Observation space:
        # [0-4]: Current player's sequence (padded with -1)
        # [5-9]: Opponent's sequence (padded with -1)
        # [10]: Current drawn card (1-10)
        self.observation_space = spaces.Box(
            low=-1, high=10, shape=(11,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the deck with two copies of numbers 1 to 10
        self.deck = [i for i in range(1, 11)] * 2
        random.shuffle(self.deck)

        # Initialize player sequences: lists of length 5, filled with -1
        self.player_sequences = [
            np.full(5, -1, dtype=np.int32),
            np.full(5, -1, dtype=np.int32),
        ]

        # Randomly decide who goes first (0 or 1)
        self.current_player = random.choice([0, 1])

        # Draw the first card for the starting player
        self.current_card = self.deck.pop()

        # Game not done
        self.done = False

        # Prepare the initial observation
        observation = self._get_observation()

        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If game is over, return current observation
            return self._get_observation(), 0, True, False, {}

        reward = 0
        info = {}

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, info

        current_seq = self.player_sequences[self.current_player]

        if action == 0 or action == 1:
            # Attempt to add the drawn card to the sequence
            sequence_valid = self._can_add_card(current_seq, self.current_card, action)
            if not sequence_valid:
                # Invalid move according to the Sequence Rule
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, info
            else:
                # Add the card to the sequence
                self._add_card_to_sequence(current_seq, self.current_card, action)

                # Check for win condition
                if self._sequence_length(current_seq) == 5:
                    # Current player wins
                    self.done = True
                    reward = 1
                    return self._get_observation(), reward, True, False, info
        elif action == 2:
            # Discard the drawn card
            pass  # Discarded cards are removed from play

        # Check if the deck is exhausted
        if not self.deck:
            # Determine winner per exhausted deck condition
            self.done = True
            winner = self._determine_winner()
            if winner == self.current_player:
                reward = 1  # Current player wins
            else:
                reward = -1  # Current player loses
            return self._get_observation(), reward, True, False, info

        # Switch to the next player
        self.current_player = 1 - self.current_player

        # Draw a new card for the next player
        self.current_card = self.deck.pop()

        # Prepare the next observation
        observation = self._get_observation()

        return (
            observation,
            reward,
            False,
            False,
            info,
        )  # Observation, reward, terminated, truncated, info

    def render(self):
        # Return a visual representation of the environment state as a string
        seq_strs = []
        for i, seq in enumerate(self.player_sequences):
            seq_display = [str(num) if num != -1 else "_" for num in seq if num != -1]
            seq_str = " ".join(seq_display)
            seq_strs.append(f"Player {i + 1}'s Sequence: [{seq_str}]")

        deck_str = f"Cards remaining in deck: {len(self.deck)}"

        current_card_str = f"Current Card: {self.current_card}"

        current_player_str = f"Current Player: Player {self.current_player + 1}"

        render_str = "\n".join(
            [current_player_str, current_card_str] + seq_strs + [deck_str]
        )

        return render_str

    def valid_moves(self):
        # Return a list of valid actions (indices)
        valid_actions = []

        current_seq = self.player_sequences[self.current_player]
        current_card = self.current_card

        # Action 0: Add to beginning
        if self._can_add_card(current_seq, current_card, 0):
            valid_actions.append(0)

        # Action 1: Add to end
        if self._can_add_card(current_seq, current_card, 1):
            valid_actions.append(1)

        # Action 2: Discard is always a valid action
        valid_actions.append(2)

        return valid_actions

    def _get_observation(self):
        # Prepare observation array
        observation = np.full(11, -1, dtype=np.int32)

        # Current player's sequence (positions 0-4)
        curr_seq = self.player_sequences[self.current_player]
        observation[0:5] = curr_seq

        # Opponent's sequence (positions 5-9)
        opp_seq = self.player_sequences[1 - self.current_player]
        observation[5:10] = opp_seq

        # Current drawn card (position 10)
        observation[10] = self.current_card

        return observation

    def _sequence_length(self, sequence):
        # Counts the number of valid numbers in the sequence
        return np.count_nonzero(sequence != -1)

    def _can_add_card(self, sequence, card, action):
        # Checks if the card can be added to the sequence according to the Sequence Rule
        seq_len = self._sequence_length(sequence)
        if seq_len == 0:
            # Sequence is empty; can add any card
            return True

        if action == 0:
            # Add to beginning
            first_card = sequence[0]
            if abs(first_card - card) == 1:
                return True
        elif action == 1:
            # Add to end
            last_card_index = seq_len - 1
            last_card = sequence[last_card_index]
            if abs(last_card - card) == 1:
                return True

        return False

    def _add_card_to_sequence(self, sequence, card, action):
        # Adds the card to the sequence at the specified end
        seq_len = self._sequence_length(sequence)
        if action == 0:
            # Add to beginning
            for i in range(seq_len, 0, -1):
                sequence[i] = sequence[i - 1]
            sequence[0] = card
        elif action == 1:
            # Add to end
            sequence[seq_len] = card

    def _determine_winner(self):
        # Determine the winner when the deck runs out
        seq_lengths = [self._sequence_length(seq) for seq in self.player_sequences]
        if seq_lengths[0] > seq_lengths[1]:
            return 0
        elif seq_lengths[1] > seq_lengths[0]:
            return 1
        else:
            # Sequences are equal in length; compare sums
            seq_sums = [np.sum(seq[seq != -1]) for seq in self.player_sequences]
            if seq_sums[0] > seq_sums[1]:
                return 0
            else:
                return 1  # Ties are not possible as per game rules
