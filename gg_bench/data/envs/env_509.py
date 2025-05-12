import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 5 reveal actions + 10 swap actions = 15 possible actions
        self.action_space = spaces.Discrete(15)

        # Observation space: 5 cards + 4 hand indicators = 9 elements
        # Cards: -1 (face-down), 1 (Lock), 2 (Key), 3 (Treasure)
        # Hands: 0 (do not have), 1 (have)
        self.observation_space = spaces.Box(low=-1, high=3, shape=(9,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize cards: 2 Locks, 2 Keys, 1 Treasure
        cards = ["Lock", "Lock", "Key", "Key", "Treasure"]
        np.random.shuffle(cards)
        # Each card has 'type' and 'face_up' status
        self.cards = [{"type": card, "face_up": False} for card in cards]
        # Players' hands: Each player can hold 'Lock' and 'Key'
        self.player_hands = {
            1: {"Lock": False, "Key": False},
            -1: {"Lock": False, "Key": False},
        }
        # Set current player (1 or -1)
        self.current_player = 1
        # Game over status
        self.done = False

        # Return observation and info
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), -10, True, False, {}
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}
        reward = -10  # Reward for a valid move
        terminated = False
        # Perform action
        if action >= 0 and action <= 4:
            # Reveal action at position action (0-4)
            card = self.cards[action]
            # Reveal the card
            card["face_up"] = True
            card_type = card["type"]
            if card_type == "Lock" or card_type == "Key":
                # Attempt to take the card
                if not self.player_hands[self.current_player][card_type]:
                    # Player does not have this card type, take it
                    self.player_hands[self.current_player][card_type] = True
                else:
                    # Player already has this type, leave it face-up on the table
                    pass
            elif card_type == "Treasure":
                # Check if player holds both a Lock and a Key
                has_lock = self.player_hands[self.current_player]["Lock"]
                has_key = self.player_hands[self.current_player]["Key"]
                if has_lock and has_key:
                    # Player wins
                    reward = 1  # Winning reward
                    self.done = True
                    terminated = True
                    return self._get_obs(), reward, True, False, {}
                else:
                    # Treasure remains face-up
                    pass
        elif action >= 5 and action <= 14:
            # Swap action
            # Map action to positions to swap
            swap_indices = [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 3),
                (2, 4),
                (3, 4),
            ]
            swap_action_idx = action - 5
            pos1, pos2 = swap_indices[swap_action_idx]
            card1 = self.cards[pos1]
            card2 = self.cards[pos2]
            # Swap only if both cards are face-down
            if not card1["face_up"] and not card2["face_up"]:
                # Swap the cards
                self.cards[pos1], self.cards[pos2] = self.cards[pos2], self.cards[pos1]
            else:
                # Invalid move: swapping face-up cards
                self.done = True
                terminated = True
                return self._get_obs(), -10, True, False, {}
        else:
            # Invalid action index
            self.done = True
            terminated = True
            return self._get_obs(), -10, True, False, {}

        # Switch to the other player
        self.current_player *= -1

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        card_states = []
        for idx, card in enumerate(self.cards):
            if card["face_up"]:
                card_states.append(f"{idx + 1}:{card['type']}")
            else:
                card_states.append(f"{idx + 1}:[ ]")
        player1_hand = []
        if self.player_hands[1]["Lock"]:
            player1_hand.append("Lock")
        if self.player_hands[1]["Key"]:
            player1_hand.append("Key")
        player2_hand = []
        if self.player_hands[-1]["Lock"]:
            player2_hand.append("Lock")
        if self.player_hands[-1]["Key"]:
            player2_hand.append("Key")
        s = f"Cards on table: {'  '.join(card_states)}\n"
        s += f"Player 1 Hand: {', '.join(player1_hand) if player1_hand else 'Empty'}\n"
        s += f"Player 2 Hand: {', '.join(player2_hand) if player2_hand else 'Empty'}\n"
        s += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return s

    def valid_moves(self):
        valid_actions = []
        # Reveal actions (indices 0-4)
        for idx in range(5):
            card = self.cards[idx]
            if not card["face_up"]:
                valid_actions.append(idx)
        # Swap actions (indices 5-14), swap pairs of face-down cards
        swap_indices = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
        ]
        for i, (pos1, pos2) in enumerate(swap_indices):
            card1 = self.cards[pos1]
            card2 = self.cards[pos2]
            if not card1["face_up"] and not card2["face_up"]:
                valid_actions.append(5 + i)
        return valid_actions

    def _get_obs(self):
        obs = np.zeros(9, dtype=np.int8)
        for idx, card in enumerate(self.cards):
            if not card["face_up"]:
                obs[idx] = -1
            else:
                if card["type"] == "Lock":
                    obs[idx] = 1
                elif card["type"] == "Key":
                    obs[idx] = 2
                elif card["type"] == "Treasure":
                    obs[idx] = 3
        # Player 1 hand
        obs[5] = int(self.player_hands[1]["Lock"])
        obs[6] = int(self.player_hands[1]["Key"])
        # Player 2 hand
        obs[7] = int(self.player_hands[-1]["Lock"])
        obs[8] = int(self.player_hands[-1]["Key"])
        return obs
