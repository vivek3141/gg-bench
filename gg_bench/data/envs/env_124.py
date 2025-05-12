import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0-4 (play card at index), 5 (choose not to defend)
        self.action_space = spaces.Discrete(6)

        # Observation space:
        # [0-4]: Player's hand (cards), padded with zeros if less than 5 cards
        # [5]: Player's HP
        # [6]: Opponent's HP
        # [7]: Opponent's last card played (0 if none)
        # [8]: Phase indicator (0: attack, 1: defense)
        self.observation_space = spaces.Box(low=0, high=20, shape=(9,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.current_player = 1  # Player 1 starts
        self.phase = 0  # 0: attack phase, 1: defense phase

        # Initialize players' HP
        self.player_hp = {1: 20, 2: 20}

        # Initialize decks
        self.decks = {}
        for player in [1, 2]:
            deck = [i for i in range(1, 11)] * 2  # Two copies of 1-10
            self.np_random.shuffle(deck)
            self.decks[player] = deck

        # Initialize hands and discard piles
        self.hands = {1: [], 2: []}
        self.discard_piles = {1: [], 2: []}

        # Draw initial hands
        for player in [1, 2]:
            self.draw_cards(player, 5)

        # Initialize last opponent card
        self.last_opponent_card = 0

        # Initialize attack and defense cards
        self.attack_card = 0
        self.defense_card = 0

        observation = self.get_observation()
        return observation, {}

    def draw_cards(self, player, num_cards):
        for _ in range(num_cards):
            if len(self.decks[player]) == 0:
                # Shuffle discard pile into deck
                self.decks[player] = self.discard_piles[player][:]
                self.discard_piles[player] = []
                self.np_random.shuffle(self.decks[player])
            if len(self.decks[player]) > 0:
                # Draw a card
                card = self.decks[player].pop()
                self.hands[player].append(card)

    def get_observation(self):
        player = self.current_player
        opponent = 2 if player == 1 else 1

        # Player's hand
        hand = self.hands[player]
        hand_obs = hand + [0] * (5 - len(hand))

        # Player's HP
        player_hp = self.player_hp[player]

        # Opponent's HP
        opponent_hp = self.player_hp[opponent]

        # Opponent's last card
        opp_last_card = self.last_opponent_card

        # Phase indicator
        phase = self.phase

        observation = np.array(
            hand_obs + [player_hp, opponent_hp, opp_last_card, phase],
            dtype=np.int32,
        )
        return observation

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        reward = 0
        player = self.current_player
        opponent = 2 if player == 1 else 1

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            reward = -10
            self.done = True
            return self.get_observation(), reward, True, False, {}

        if self.phase == 0:
            # Attack phase
            hand = self.hands[player]
            chosen_card = hand[action]
            self.attack_card = chosen_card
            # Remove card from hand
            self.hands[player].pop(action)
            self.cards_to_discard = {player: [chosen_card]}
            # Set phase to defense
            self.phase = 1
            # Switch current player to defender
            self.current_player = opponent
            # Set last opponent card (attacker's card)
            self.last_opponent_card = self.attack_card
            observation = self.get_observation()
            return observation, reward, False, False, {}
        elif self.phase == 1:
            # Defense phase
            if action == 5:
                # Choose not to defend
                self.defense_card = 0
                self.cards_to_discard[player] = []
            else:
                hand = self.hands[player]
                chosen_card = hand[action]
                self.defense_card = chosen_card
                # Remove card from hand
                self.hands[player].pop(action)
                self.cards_to_discard[player] = [chosen_card]

            # Resolve attack
            attack_value = self.attack_card
            defense_value = self.defense_card
            damage = max(0, attack_value - defense_value)
            # Adjust HP of defender (current player)
            self.player_hp[player] -= damage

            # Check for game over
            if self.player_hp[player] <= 0:
                self.done = True
                reward = 0
                # Discard used cards
                for p in self.cards_to_discard:
                    self.discard_piles[p] += self.cards_to_discard[p]
                # Prepare observation
                observation = self.get_observation()
                return observation, reward, True, False, {}

            # Discard used cards
            for p in self.cards_to_discard:
                self.discard_piles[p] += self.cards_to_discard[p]

            # Both players draw cards to refill hand to 5 cards
            self.draw_cards(1, 5 - len(self.hands[1]))
            self.draw_cards(2, 5 - len(self.hands[2]))

            # Reset phase to attack
            self.phase = 0
            # Switch current player to next attacker
            self.current_player = opponent
            # Reset last opponent card
            self.last_opponent_card = 0
            # Reset attack and defense cards
            self.attack_card = 0
            self.defense_card = 0

            # Prepare observation
            observation = self.get_observation()
            return observation, reward, False, False, {}
        else:
            # Should not reach here
            reward = -10
            self.done = True
            return self.get_observation(), reward, True, False, {}

    def render(self):
        player = self.current_player
        opponent = 2 if player == 1 else 1
        phase_str = "Attack Phase" if self.phase == 0 else "Defense Phase"
        hand = self.hands[player]
        hand_str = ", ".join(map(str, hand))
        obs = self.get_observation()
        render_str = f"Current Player: Player {player}\n"
        render_str += f"Phase: {phase_str}\n"
        render_str += f"Player {player} HP: {self.player_hp[player]}\n"
        render_str += f"Player {opponent} HP: {self.player_hp[opponent]}\n"
        render_str += f"Player {player} Hand: {hand_str}\n"
        render_str += f"Opponent's Last Card: {self.last_opponent_card}\n"
        return render_str

    def valid_moves(self):
        player = self.current_player
        if self.phase == 0:
            # Attack phase: must play a card from hand
            return list(range(len(self.hands[player])))
        else:
            # Defense phase: can play a card or choose not to defend (action 5)
            valid_actions = list(range(len(self.hands[player])))
            valid_actions.append(5)  # Action 5: choose not to defend
            return valid_actions
