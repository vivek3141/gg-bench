import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to card indices 0-4 (cards 1-5)
        self.action_space = spaces.Discrete(5)

        # Observation space consists of:
        # - Player's hand (5 cards): 1 if available, 0 if used
        # - Opponent's hand (5 cards): same as above
        # - Player's score: 0 to 3
        # - Opponent's score: 0 to 3
        # - Current player: 0 or 1
        self.observation_space = spaces.Box(
            low=np.array([0] * 5 + [0] * 5 + [0, 0] + [0]),
            high=np.array([1] * 5 + [1] * 5 + [3, 3] + [1]),
            dtype=np.int8,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize hands and scores
        self.player_hands = [
            np.array([1, 1, 1, 1, 1], dtype=np.int8),  # Player 1's hand
            np.array([1, 1, 1, 1, 1], dtype=np.int8),  # Player 2's hand
        ]
        self.scores = [0, 0]
        self.done = False
        self.current_player = 0  # Start with Player 1 (index 0)

        # Prepare for the first round
        self.round_active = False  # Indicates if a round is in progress
        self.selected_cards = [None, None]  # To store selected cards of both players

        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        own_hand = self.player_hands[self.current_player]
        opponent_hand = self.player_hands[1 - self.current_player]
        own_score = self.scores[self.current_player]
        opponent_score = self.scores[1 - self.current_player]
        current_player = self.current_player
        observation = np.concatenate(
            [
                own_hand,
                opponent_hand,
                np.array([own_score, opponent_score]),
                np.array([current_player]),
            ]
        )
        return observation

    def valid_moves(self):
        # Return indices of available cards for the current player
        valid = [i for i in range(5) if self.player_hands[self.current_player][i] == 1]
        return valid

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {"error": "Game is over"}

        # Check if the action is valid
        if action not in self.valid_moves():
            self.done = True
            reward = -10
            terminated = True
            truncated = False
            info = {"error": "Invalid action"}
            return self._get_observation(), reward, terminated, truncated, info

        # Store the action
        self.selected_cards[self.current_player] = action
        self.player_hands[self.current_player][action] = 0  # Mark the card as used

        # If both players have selected their cards, resolve the round
        if not self.round_active:
            # Opponent selects a card
            opponent_valid_moves = [
                i
                for i in range(5)
                if self.player_hands[1 - self.current_player][i] == 1
            ]
            opponent_action = np.random.choice(opponent_valid_moves)
            self.selected_cards[1 - self.current_player] = opponent_action
            self.player_hands[1 - self.current_player][
                opponent_action
            ] = 0  # Mark card as used

            # Resolve the round
            player_card_value = self.selected_cards[self.current_player] + 1
            opponent_card_value = self.selected_cards[1 - self.current_player] + 1

            info = {
                "player_action": player_card_value,
                "opponent_action": opponent_card_value,
            }

            if player_card_value > opponent_card_value:
                # Current player wins the challenge
                self.scores[self.current_player] += 1
                reward = 1
            elif player_card_value < opponent_card_value:
                # Opponent wins the challenge
                self.scores[1 - self.current_player] += 1
                reward = 0
            else:
                # Tie
                reward = 0

            # Reset for next round
            self.selected_cards = [None, None]
            self.round_active = False

            # Check for game end conditions
            if self.scores[self.current_player] >= 3:
                self.done = True
                terminated = True
            elif self.scores[1 - self.current_player] >= 3:
                self.done = True
                terminated = True
                reward = -1  # Opponent wins
            else:
                # Check if all cards have been used
                if (
                    np.sum(self.player_hands[0]) == 0
                    and np.sum(self.player_hands[1]) == 0
                ):
                    # Begin Sudden Death Rounds: reset hands
                    self.player_hands = [
                        np.array([1, 1, 1, 1, 1], dtype=np.int8),
                        np.array([1, 1, 1, 1, 1], dtype=np.int8),
                    ]
            # Switch current player
            self.current_player = 1 - self.current_player
            observation = self._get_observation()
            terminated = self.done
            truncated = False
            return observation, reward, terminated, truncated, info
        else:
            # Wait for the next player's action
            self.round_active = False
            # Switch current player
            self.current_player = 1 - self.current_player
            observation = self._get_observation()
            reward = 0
            terminated = False
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info

    def render(self):
        own_hand = self.player_hands[self.current_player]
        opponent_hand = self.player_hands[1 - self.current_player]
        own_score = self.scores[self.current_player]
        opponent_score = self.scores[1 - self.current_player]
        current_player = self.current_player + 1
        hand_str = ", ".join(str(i + 1) for i, x in enumerate(own_hand) if x == 1)
        opponent_used_cards = ", ".join(
            str(i + 1) for i, x in enumerate(1 - opponent_hand) if x == 1
        )
        render_str = (
            f"--- Player {current_player}'s Turn ---\n"
            f"Your Hand: {hand_str}\n"
            f"Opponent's Used Cards: {opponent_used_cards}\n"
            f"Scores - You: {own_score}, Opponent: {opponent_score}\n"
        )
        print(render_str)
