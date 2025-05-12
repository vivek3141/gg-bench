import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space

        # Action space: Discrete(9), representing card values from 1 to 9
        self.action_space = spaces.Discrete(9)

        # Observation space: Box with shape (7,)
        # [player_hand (3 cards), opponent_last_card, player_HP, opponent_HP, current_player]
        self.observation_space = spaces.Box(low=0, high=10, shape=(7,), dtype=np.int32)

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create and shuffle deck: four copies of each number from 1 to 9
        self.deck = [i for i in range(1, 10)] * 4
        np.random.shuffle(self.deck)

        # Initialize discard pile
        self.discard_pile = []

        # Initialize players
        self.players = {
            1: {
                "hand": [self.deck.pop() for _ in range(3)],
                "HP": 10,
                "last_card": 0,  # 0 indicates no card played yet
            },
            -1: {
                "hand": [self.deck.pop() for _ in range(3)],
                "HP": 10,
                "last_card": 0,
            },
        }

        # Set current player
        self.current_player = 1  # Players are 1 and -1

        # Game not done
        self.done = False

        # Return initial observation
        return self._get_obs(), {}

    def step(self, action):
        cp = self.current_player
        opp = -self.current_player

        # Map action to card value (1-9)
        card_played = action + 1  # Actions 0-8 correspond to cards 1-9

        # Check for valid action
        if card_played not in self.players[cp]["hand"] or self.done:
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Play the card
        self.players[cp]["hand"].remove(card_played)
        self.players[cp]["last_card"] = card_played
        self.discard_pile.append(card_played)

        # Resolve the clash
        opp_last_card = self.players[opp]["last_card"]

        if opp_last_card == 0:
            # First turn, no damage is dealt
            pass
        else:
            if card_played > opp_last_card:
                damage = card_played - opp_last_card
                self.players[opp]["HP"] -= damage
            elif card_played < opp_last_card:
                damage = opp_last_card - card_played
                self.players[cp]["HP"] -= damage
            else:
                # Cards are equal, both players take 1 damage
                self.players[cp]["HP"] -= 1
                self.players[opp]["HP"] -= 1

        # Check for game end conditions
        if self.players[opp]["HP"] <= 0:
            # Current player wins
            self.done = True
            reward = 1
            terminated = True
        elif self.players[cp]["HP"] <= 0:
            # Current player loses
            self.done = True
            reward = -1
            terminated = True
        else:
            # Game continues
            reward = 0
            terminated = False

        # Draw a card for current player
        self._draw_card(cp)

        # Switch current player
        self.current_player *= -1

        # Return observation from the new current player's perspective
        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        cp = self.current_player
        opp = -self.current_player

        hand = self.players[cp]["hand"]
        opp_last_card = self.players[opp]["last_card"]
        cp_HP = self.players[cp]["HP"]
        opp_HP = self.players[opp]["HP"]

        result = f"Player {cp}'s turn.\n"
        result += f"Your HP: {cp_HP}\n"
        result += f"Opponent's HP: {opp_HP}\n"
        result += f"Your hand: {hand}\n"
        if opp_last_card == 0:
            result += "Opponent's last card: None\n"
        else:
            result += f"Opponent's last card: {opp_last_card}\n"

        return result

    def valid_moves(self):
        # Return a list of valid actions based on current player's hand
        cp = self.current_player
        valid_actions = [card - 1 for card in self.players[cp]["hand"]]
        return valid_actions

    def _get_obs(self):
        # Get observation for current player
        cp = self.current_player
        opp = -self.current_player

        hand = self.players[cp]["hand"]
        opp_last_card = self.players[opp]["last_card"]
        cp_HP = self.players[cp]["HP"]
        opp_HP = self.players[opp]["HP"]

        # Observation array: [hand (3 cards), opponent's last card, player's HP, opponent's HP, current player]
        obs = np.array(hand + [opp_last_card, cp_HP, opp_HP, cp], dtype=np.int32)
        return obs

    def _draw_card(self, player):
        # Replenish hand to three cards
        if len(self.deck) == 0:
            # Shuffle discard pile to form a new deck
            self.deck = self.discard_pile
            self.discard_pile = []
            np.random.shuffle(self.deck)

        if len(self.deck) > 0:
            new_card = self.deck.pop()
            self.players[player]["hand"].append(new_card)
