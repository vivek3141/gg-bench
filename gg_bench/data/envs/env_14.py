import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Player selects a card numbered from 1 to 9
        self.action_space = spaces.Discrete(
            9
        )  # Actions are integers 0-8, representing cards 1-9

        # Observation space:
        # [Player's hand (9 elements), Player's previous plays (5), Opponent's previous plays (5), Scores (2)]
        self.observation_space = spaces.Box(low=0, high=9, shape=(21,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Each player's hand is a list of cards numbered 1 through 9
        self.player_hand = list(range(1, 10))
        self.opponent_hand = list(range(1, 10))

        # Keep track of previous plays
        self.player_previous_plays = [0] * 5  # Initialize with zeros
        self.opponent_previous_plays = [0] * 5

        # Initialize scores
        self.player_score = 0
        self.opponent_score = 0

        # Current round (0 to 4)
        self.current_round = 0

        # Game over flag
        self.done = False

        # Create the initial observation
        observation = self._get_observation()

        return observation, {}  # Return observation and info

    def step(self, action):
        reward = 0
        info = {}
        truncated = False

        if self.done:
            # If the game is already over, ignore further actions
            return self._get_observation(), 0, True, False, info

        # Validate action
        if action < 0 or action >= 9:
            # Invalid action index
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, info

        selected_card = action + 1  # Map action index to card number (1-9)

        if selected_card not in self.player_hand:
            # Player played a card not in hand
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, info

        # Remove the selected card from player's hand
        self.player_hand.remove(selected_card)

        # Opponent selects a card (random policy)
        opponent_card = self._opponent_policy()

        # Remove opponent's selected card from their hand
        self.opponent_hand.remove(opponent_card)

        # Record the plays
        self.player_previous_plays[self.current_round] = selected_card
        self.opponent_previous_plays[self.current_round] = opponent_card

        # Resolve the round
        self._resolve_round(selected_card, opponent_card)

        # Move to next round
        self.current_round += 1

        if self.current_round >= 5:
            # Game over after five rounds
            self.done = True
            # Determine final reward
            if self.player_score > self.opponent_score:
                reward = 1  # Player wins the game
            else:
                reward = 0  # Player loses or ties (no draws per game rules)
        else:
            # Game continues
            reward = 0

        observation = self._get_observation()

        return observation, reward, self.done, truncated, info

    def render(self):
        # Build a string representation of the game state
        render_str = f"Round {self.current_round + 1}\n"
        render_str += "Player's Hand: " + ", ".join(map(str, self.player_hand)) + "\n"
        render_str += (
            f"Scores - Player: {self.player_score} | Opponent: {self.opponent_score}\n"
        )
        render_str += "Previous Plays:\n"
        for i in range(self.current_round):
            render_str += f"  Round {i+1}: Player played {self.player_previous_plays[i]}, Opponent played {self.opponent_previous_plays[i]}\n"
        return render_str

    def valid_moves(self):
        # Return the indices of valid moves (0-8 for cards 1-9)
        return [card - 1 for card in self.player_hand]

    def _get_observation(self):
        # Build the observation array
        # Player's hand: 9 elements, 1 if the card is in hand, 0 otherwise
        hand_array = np.zeros(9, dtype=np.int32)
        for card in self.player_hand:
            hand_array[card - 1] = 1

        # Player's previous plays: 5 elements
        player_plays_array = np.array(self.player_previous_plays, dtype=np.int32)

        # Opponent's previous plays: 5 elements
        opponent_plays_array = np.array(self.opponent_previous_plays, dtype=np.int32)

        # Scores: 2 elements
        scores_array = np.array(
            [self.player_score, self.opponent_score], dtype=np.int32
        )

        # Concatenate all parts to form the observation
        observation = np.concatenate(
            (hand_array, player_plays_array, opponent_plays_array, scores_array)
        )

        return observation

    def _opponent_policy(self):
        # Simple opponent policy: random selection from available cards
        return self.np_random.choice(self.opponent_hand)

    def _resolve_round(self, player_card, opponent_card):
        # Determine the outcome of the round
        if player_card > opponent_card:
            # Player wins the round
            self.player_score += 1
        elif player_card < opponent_card:
            # Opponent wins the round
            self.opponent_score += 1
        else:
            # Tie occurred, apply tie-breaker rules
            winner = self._apply_tie_breaker(self.current_round)
            if winner == "player":
                self.player_score += 2  # Tie-breaker win earns 2 points
            elif winner == "opponent":
                self.opponent_score += 2
            else:
                # No points awarded
                pass

    def _apply_tie_breaker(self, current_round):
        if current_round == 0:
            # First round tie, no points awarded
            return None
        else:
            # Compare previous rounds to determine tie-breaker
            previous_index = current_round - 1
            while previous_index >= 0:
                prev_player_card = self.player_previous_plays[previous_index]
                prev_opponent_card = self.opponent_previous_plays[previous_index]
                if prev_player_card != prev_opponent_card:
                    if prev_player_card < prev_opponent_card:
                        return "player"
                    else:
                        return "opponent"
                previous_index -= 1
            # All previous rounds were ties
            return None
