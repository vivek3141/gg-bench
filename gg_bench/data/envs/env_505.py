import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Maximum action space size is 41 (31 attack actions + 5 defend + 5 swap)
        self.action_space = spaces.Discrete(41)

        # Observation: [Player_CI, Opponent_CI, Hand[5]]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),
            high=np.array([15, 15, 5, 5, 5, 5, 5]),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize Game State
        self.player_CI = 15
        self.opponent_CI = 15

        # Create Deck: 5 copies of cards 1-5 (total 25 cards)
        self.deck = [i for i in range(1, 6)] * 5
        random.shuffle(self.deck)

        # Initialize Hands
        self.player_hand = [self.draw_card() for _ in range(3)]
        self.opponent_hand = [self.draw_card() for _ in range(3)]

        self.discard_pile = []

        # Current player (1 for player, -1 for opponent)
        self.current_player = 1  # Start with player

        self.done = False

        observation = self.get_observation()
        return observation, {}  # Observation and info

    def step(self, action):
        if self.done:
            return self.get_observation(), -10, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self.get_observation(), -10, True, False, {}

        # Draw Phase
        self.draw_phase()

        # Action Phase
        reward = 0

        if action < 31:
            # Attack action
            attack_cards = self.get_attack_cards(action)
            attack_value, sequence_bonus = self.calculate_attack_value(attack_cards)
            total_attack = attack_value + sequence_bonus

            # Apply damage
            if self.current_player == 1:
                self.opponent_CI -= total_attack
                # Discard used cards
                self.discard_pile.extend(attack_cards)
                for card in attack_cards:
                    self.player_hand.remove(card)
            else:
                self.player_CI -= total_attack
                # Discard used cards
                self.discard_pile.extend(attack_cards)
                for card in attack_cards:
                    self.opponent_hand.remove(card)

        elif action < 36:
            # Defend action
            card_index = action - 31
            if self.current_player == 1:
                card = self.player_hand.pop(card_index)
                self.player_CI += card
                if self.player_CI > 15:
                    self.player_CI = 15
                self.discard_pile.append(card)
            else:
                card = self.opponent_hand.pop(card_index)
                self.opponent_CI += card
                if self.opponent_CI > 15:
                    self.opponent_CI = 15
                self.discard_pile.append(card)

        else:
            # Swap action
            card_index = action - 36
            if self.current_player == 1:
                player_card = self.player_hand.pop(card_index)
                opponent_card = random.choice(self.opponent_hand)
                self.opponent_hand.remove(opponent_card)
                self.player_hand.append(opponent_card)
                self.opponent_hand.append(player_card)
            else:
                opponent_card = self.opponent_hand.pop(card_index)
                player_card = random.choice(self.player_hand)
                self.player_hand.remove(player_card)
                self.opponent_hand.append(player_card)
                self.player_hand.append(opponent_card)

        # End Phase
        if len(self.player_hand) > 5:
            self.player_hand = self.player_hand[:5]
        if len(self.opponent_hand) > 5:
            self.opponent_hand = self.opponent_hand[:5]

        # Check for win condition
        if self.opponent_CI <= 0 and self.current_player == 1:
            reward = 1
            self.done = True
        elif self.player_CI <= 0 and self.current_player == -1:
            reward = 1
            self.done = True

        # Switch players
        self.current_player *= -1

        observation = self.get_observation()
        return (
            observation,
            reward,
            self.done,
            False,
            {},
        )  # Observation, reward, done, truncated, info

    def render(self):
        if self.current_player == 1:
            hand = self.player_hand
            CI = self.player_CI
            opponent_CI = self.opponent_CI
        else:
            hand = self.opponent_hand
            CI = self.opponent_CI
            opponent_CI = self.player_CI

        hand_str = ", ".join(str(card) for card in hand)
        deck_size = len(self.deck)
        discard_size = len(self.discard_pile)

        state = f"=== Player {1 if self.current_player == 1 else 2} Turn ===\n"
        state += f"Your CI: {CI}\n"
        state += f"Opponent's CI: {opponent_CI}\n"
        state += f"Your Hand: [{hand_str}]\n"
        state += f"Deck Size: {deck_size}\n"
        state += f"Discard Pile Size: {discard_size}\n"
        return state

    def valid_moves(self):
        valid_actions = []

        if self.current_player == 1:
            hand = self.player_hand
        else:
            hand = self.opponent_hand

        hand_size = len(hand)
        card_indices = list(range(hand_size))

        # Attack actions (non-empty subsets of hand)
        subsets = self.get_all_non_empty_subsets(card_indices)
        for subset in subsets:
            action_id = self.get_action_id_from_subset(subset)
            valid_actions.append(action_id)

        # Defend actions
        for idx in card_indices:
            action_id = 31 + idx
            valid_actions.append(action_id)

        # Swap actions
        for idx in card_indices:
            action_id = 36 + idx
            valid_actions.append(action_id)

        return valid_actions

    # Helper methods
    def draw_card(self):
        if not self.deck:
            if not self.discard_pile:
                return None
            self.deck = self.discard_pile
            self.discard_pile = []
            random.shuffle(self.deck)
        return self.deck.pop(0)

    def draw_phase(self):
        card = self.draw_card()
        if card is not None:
            if self.current_player == 1 and len(self.player_hand) < 5:
                self.player_hand.append(card)
            elif self.current_player == -1 and len(self.opponent_hand) < 5:
                self.opponent_hand.append(card)
            else:
                self.discard_pile.append(card)

    def get_observation(self):
        if self.current_player == 1:
            hand = self.player_hand
            CI = self.player_CI
            opponent_CI = self.opponent_CI
        else:
            hand = self.opponent_hand
            CI = self.opponent_CI
            opponent_CI = self.player_CI

        hand_padded = hand + [0] * (5 - len(hand))
        observation = np.array([CI, opponent_CI] + hand_padded, dtype=np.int32)
        return observation

    def get_all_non_empty_subsets(self, elements):
        subsets = []
        for i in range(1, 1 << len(elements)):
            subset = [elements[j] for j in range(len(elements)) if (i & (1 << j))]
            subsets.append(subset)
        return subsets

    def get_action_id_from_subset(self, subset):
        # Subsets are converted to action IDs from 0 to 30
        action_id = 0
        for idx in subset:
            action_id |= 1 << idx
        action_id -= 1  # Zero-based indexing
        return action_id

    def get_attack_cards(self, action_id):
        # Convert action ID to subset of card indices
        mask = action_id + 1
        indices = [i for i in range(5) if (mask & (1 << i))]
        if self.current_player == 1:
            hand = self.player_hand
        else:
            hand = self.opponent_hand
        attack_cards = [hand[i] for i in indices]
        return attack_cards

    def calculate_attack_value(self, cards):
        attack_value = sum(cards)
        # Check for sequence
        sorted_cards = sorted(cards)
        is_sequence = all(
            sorted_cards[i] + 1 == sorted_cards[i + 1]
            for i in range(len(sorted_cards) - 1)
        )
        sequence_bonus = len(cards) if is_sequence and len(cards) >= 2 else 0
        return attack_value, sequence_bonus
