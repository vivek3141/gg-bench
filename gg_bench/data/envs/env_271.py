import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 8 possible actions
        # 0-4: Play Charge Card adding 1-5 points
        # 5: Play Block Card
        # 6: Play Steal Card
        # 7: Forfeit turn to refresh Charge Cards
        self.action_space = spaces.Discrete(8)

        # Define observation space:
        # [ own_charge_points, own_charge_cards, own_block_cards, own_steal_cards,
        #   opponent_charge_points, blocked_on_current_turn, amount_opponent_gained_last_turn ]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),
            high=np.array([15, 5, 2, 1, 15, 1, 5]),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the game state
        # Both players' charge points
        self.player_charge_points = {1: 0, -1: 0}

        # Both players' hands
        self.player_hands = {
            1: {"Charge": 5, "Block": 2, "Steal": 1},
            -1: {"Charge": 5, "Block": 2, "Steal": 1},
        }

        # Both players' blocked status
        self.player_blocked = {1: False, -1: False}

        # Both players' last gain
        self.player_last_gain = {1: 0, -1: 0}

        # Current player: 1 or -1
        self.current_player = 1

        self.done = False

        # Observation
        observation = self._get_observation()

        return observation, {}

    def step(self, action):
        if self.done:
            # Game is over
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Process the action
        reward = 0
        opponent = -self.current_player

        # Get current player's data
        current_hand = self.player_hands[self.current_player]
        current_charge = self.player_charge_points[self.current_player]
        opponent_charge = self.player_charge_points[opponent]

        if action in [0, 1, 2, 3, 4]:  # Charge Card adding 1-5 points
            amount = action + 1  # action 0 corresponds to adding 1 point
            current_hand["Charge"] -= 1  # Discard one Charge Card

            if self.player_blocked[self.current_player]:
                # Charge Card has no effect
                self.player_blocked[self.current_player] = False  # Block effect ends
                self.player_last_gain[self.current_player] = 0
            else:
                # Add charge points
                current_charge += amount
                # Check for exceeding 15
                if current_charge > 15:
                    current_charge = 0
                # Update charge points
                self.player_charge_points[self.current_player] = current_charge
                self.player_last_gain[self.current_player] = amount

            # After action, check for win
            if current_charge == 15:
                self.done = True
                reward = 1
                return self._get_observation(), reward, True, False, {}

        elif action == 5:  # Block Card
            current_hand["Block"] -= 1
            # Apply block effect on opponent
            self.player_blocked[opponent] = True
            self.player_last_gain[self.current_player] = 0  # No gain this turn

        elif action == 6:  # Steal Card
            current_hand["Steal"] -= 1

            amount_to_steal = self.player_last_gain[opponent]
            if amount_to_steal > 0:
                # Adjust charge points
                current_charge += amount_to_steal
                opponent_charge -= amount_to_steal

                # Check for exceeding 15
                if current_charge > 15:
                    current_charge = 0

                if opponent_charge < 0:
                    opponent_charge = 0  # Safety check

                self.player_charge_points[self.current_player] = current_charge
                self.player_charge_points[opponent] = opponent_charge
                self.player_last_gain[self.current_player] = amount_to_steal
            else:
                # Steal Card has no effect
                self.player_last_gain[self.current_player] = 0

            # After action, check for win
            if current_charge == 15:
                self.done = True
                reward = 1
                return self._get_observation(), reward, True, False, {}

        elif action == 7:  # Forfeit turn to refresh Charge Cards
            current_hand["Charge"] = 5
            # No gain this turn
            self.player_last_gain[self.current_player] = 0

        # Switch player
        self.current_player = opponent

        return self._get_observation(), reward, self.done, False, {}

    def render(self):
        # Return a string representing the game state
        s = f"Player {self.current_player}'s turn\n"
        s += f"Your Charge Points: {self.player_charge_points[self.current_player]}\n"
        s += f"Opponent's Charge Points: {self.player_charge_points[-self.current_player]}\n"
        s += f"Your Hand: Charge Cards - {self.player_hands[self.current_player]['Charge']}, "
        s += f"Block Cards - {self.player_hands[self.current_player]['Block']}, "
        s += f"Steal Cards - {self.player_hands[self.current_player]['Steal']}\n"
        s += f"Opponent is {'blocked' if self.player_blocked[-self.current_player] else 'not blocked'} on their next turn\n"
        return s

    def valid_moves(self):
        valid_actions = []
        current_hand = self.player_hands[self.current_player]
        if current_hand["Charge"] > 0:
            # Actions 0-4 are valid (add 1-5 points)
            valid_actions.extend([0, 1, 2, 3, 4])

        if current_hand["Block"] > 0:
            valid_actions.append(5)

        if current_hand["Steal"] > 0:
            # Can only use Steal Card if opponent gained charge points on their last turn
            amount_opponent_gained = self.player_last_gain[-self.current_player]
            if amount_opponent_gained > 0:
                valid_actions.append(6)

        if current_hand["Charge"] == 0:
            # Can forfeit turn to refresh Charge Cards
            valid_actions.append(7)
        return valid_actions

    def _get_observation(self):
        own_charge = self.player_charge_points[self.current_player]
        own_hand = self.player_hands[self.current_player]
        opponent_charge = self.player_charge_points[-self.current_player]
        blocked_status = int(self.player_blocked[self.current_player])  # 0 or 1
        amount_opponent_gained = self.player_last_gain[-self.current_player]

        observation = np.array(
            [
                own_charge,
                own_hand["Charge"],
                own_hand["Block"],
                own_hand["Steal"],
                opponent_charge,
                blocked_status,
                amount_opponent_gained,
            ],
            dtype=np.int32,
        )

        return observation
