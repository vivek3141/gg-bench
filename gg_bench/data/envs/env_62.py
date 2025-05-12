import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - Pass / Do not defend, 1-3 - Play card at position 0-2 in hand
        self.action_space = spaces.Discrete(4)

        # Define observation space
        # Observation consists of:
        #   - Player's hand: 3 values (0 if empty, else card number 1-10)
        #   - Current player's Life Points (0-10)
        #   - Opponent's Life Points (0-10)
        #   - Current phase indicator: 0 - Action Phase, 1 - Defense Phase
        #   - Attack card value: 0 if not in defense phase
        #   - Deck size (0-40)
        self.observation_space = spaces.Box(
            low=np.array([0] * 3 + [0, 0, 0, 0, 0]),
            high=np.array([10] * 3 + [10, 10, 1, 10, 40]),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the deck
        self.deck = [i for i in range(1, 11)] * 4  # 4 copies of cards 1-10
        np.random.shuffle(self.deck)

        # Initialize Life Points
        self.player_life_points = [10, 10]

        # Initialize hands
        self.player_hands = [[], []]
        for _ in range(3):
            self.draw_card(0)
            self.draw_card(1)

        # Initialize phase and attack card
        self.current_player = 0  # Player 0 starts
        self.current_phase = "action"
        self.attack_card = 0

        # Prepare initial observation
        observation = self.get_observation()
        info = {}
        return observation, info

    def draw_card(self, player):
        if len(self.deck) > 0 and len(self.player_hands[player]) < 3:
            card = self.deck.pop()
            self.player_hands[player].append(card)

    def get_observation(self):
        # Get current player's hand
        current_hand = self.player_hands[self.current_player]
        hand_obs = [0, 0, 0]
        for i, card in enumerate(current_hand):
            hand_obs[i] = card
        # Create observation array
        observation = np.array(
            hand_obs
            + [self.player_life_points[self.current_player]]
            + [self.player_life_points[1 - self.current_player]]
            + [1 if self.current_phase == "defense" else 0]
            + [self.attack_card]
            + [len(self.deck)]
        )
        return observation

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            reward = -10
            terminated = True
            observation = self.get_observation()
            return observation, reward, terminated, truncated, info

        if self.current_phase == "action":
            # Action Phase
            if action == 0:
                # Pass
                self.switch_player()
                self.do_draw_phase()
            else:
                # Attack with a card
                hand = self.player_hands[self.current_player]
                card_index = action - 1
                if card_index >= len(hand):
                    # Invalid action
                    reward = -10
                    terminated = True
                else:
                    self.attack_card = hand.pop(card_index)
                    self.current_phase = "defense"
        elif self.current_phase == "defense":
            # Defense Phase
            defender = self.current_player
            attacker = 1 - defender
            if action == 0:
                # Do not defend
                damage = self.attack_card
                self.player_life_points[defender] -= damage
                self.attack_card = 0
                self.current_phase = "action"
                self.current_player = attacker
                if self.player_life_points[defender] <= 0:
                    reward = 1  # Current player wins
                    terminated = True
            else:
                # Defend with a card
                hand = self.player_hands[defender]
                card_index = action - 1
                if card_index >= len(hand):
                    # Invalid action
                    reward = -10
                    terminated = True
                else:
                    defense_card = hand[card_index]
                    if defense_card >= self.attack_card:
                        # Successful block
                        hand.pop(card_index)
                        self.attack_card = 0
                        self.current_phase = "action"
                        self.current_player = attacker
                    else:
                        # Invalid defense
                        reward = -10
                        terminated = True
        else:
            raise Exception("Invalid phase")

        # Prepare observation
        observation = self.get_observation()

        # Check for game over
        if not terminated and (
            self.player_life_points[0] <= 0 or self.player_life_points[1] <= 0
        ):
            terminated = True
            if self.player_life_points[self.current_player] > 0:
                reward = 1  # Current player wins
            else:
                reward = -1  # Current player loses

        return observation, reward, terminated, truncated, info

    def do_draw_phase(self):
        # Draw cards for the current player
        player = self.current_player
        while len(self.player_hands[player]) < 3 and len(self.deck) > 0:
            self.draw_card(player)

    def switch_player(self):
        self.current_player = 1 - self.current_player

    def valid_moves(self):
        valid_actions = []
        if self.current_phase == "action":
            # Can always pass
            valid_actions.append(0)
            hand_size = len(self.player_hands[self.current_player])
            for i in range(hand_size):
                valid_actions.append(i + 1)
        elif self.current_phase == "defense":
            # Can choose not to defend
            valid_actions.append(0)
            defender = self.current_player
            hand = self.player_hands[defender]
            for i, card in enumerate(hand):
                if card >= self.attack_card:
                    valid_actions.append(i + 1)
        return valid_actions

    def render(self):
        s = ""
        s += f"Current Player: Player {self.current_player + 1}\n"
        s += f"Phase: {self.current_phase}\n"
        s += f"Player 1 Life Points: {self.player_life_points[0]}\n"
        s += f"Player 1 Hand: {self.player_hands[0]}\n"
        s += f"Player 2 Life Points: {self.player_life_points[1]}\n"
        s += f"Player 2 Hand: {self.player_hands[1]}\n"
        s += f"Deck Size: {len(self.deck)}\n"
        if self.current_phase == "defense":
            s += f"Attack Card: {self.attack_card}\n"
        return s
