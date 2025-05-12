import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 -> move 1 position, 1 -> move 2 positions
        self.action_space = spaces.Discrete(2)

        # Observations: positions of both players
        # Positions range from 0 to 10 inclusive
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([10, 10]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.p1_pos = 0
        self.p2_pos = 10
        self.current_player = 1  # 1 for Player 1, 2 for Player 2
        self.done = False
        self.current_turn = 1
        self.moves_in_current_turn = 0
        self.p1_reached_center_on_turn = None
        self.p1_move_order_in_turn = None
        self.p2_reached_center_on_turn = None
        self.p2_move_order_in_turn = None
        observation = np.array([self.p1_pos, self.p2_pos], dtype=np.int32)
        return observation, {}

    def step(self, action):
        if self.done:
            return self.observation(), 0, True, False, {}

        self.moves_in_current_turn += 1
        current_move_order = self.moves_in_current_turn

        # Map action to move distance
        move_distance = action + 1

        # Check if move is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self.observation(), -10, True, False, {}

        # Make the move
        if self.current_player == 1:
            self.p1_pos += move_distance
            if self.p1_pos > 5:
                self.p1_pos = 5  # Cannot move past center
            if self.p1_pos == self.p2_pos and self.p1_pos != 5:
                # Cannot land on opponent's position unless at center
                self.done = True
                return self.observation(), -10, True, False, {}
            if self.p1_pos == 5:
                self.p1_reached_center_on_turn = self.current_turn
                self.p1_move_order_in_turn = current_move_order
        else:
            self.p2_pos -= move_distance
            if self.p2_pos < 5:
                self.p2_pos = 5  # Cannot move past center
            if self.p2_pos == self.p1_pos and self.p2_pos != 5:
                # Cannot land on opponent's position unless at center
                self.done = True
                return self.observation(), -10, True, False, {}
            if self.p2_pos == 5:
                self.p2_reached_center_on_turn = self.current_turn
                self.p2_move_order_in_turn = current_move_order

        # Check for victory
        if (
            self.p1_reached_center_on_turn == self.current_turn
            and self.p2_reached_center_on_turn == self.current_turn
        ):
            # Both players reached center on the same turn
            if self.p1_move_order_in_turn == 2:
                winner = 1
            else:
                winner = 2
            self.done = True
            reward = 1 if self.current_player == winner else 0
            return self.observation(), reward, True, False, {}
        elif (
            self.current_player == 1
            and self.p1_reached_center_on_turn == self.current_turn
        ):
            if self.moves_in_current_turn == 2:
                # Both players have moved; Player 1 wins
                self.done = True
                reward = 1
                return self.observation(), reward, True, False, {}
            else:
                # Wait for Player 2's move
                pass
        elif (
            self.current_player == 2
            and self.p2_reached_center_on_turn == self.current_turn
        ):
            if self.moves_in_current_turn == 2:
                # Both players have moved; Player 2 wins
                self.done = True
                reward = 1
                return self.observation(), reward, True, False, {}
            else:
                # Wait for Player 1's move
                pass

        # End of turn
        if self.moves_in_current_turn == 2:
            self.moves_in_current_turn = 0
            self.current_turn += 1

        # Switch current player
        self.current_player = 1 if self.current_player == 2 else 2

        return self.observation(), 0, False, False, {}

    def observation(self):
        return np.array([self.p1_pos, self.p2_pos], dtype=np.int32)

    def valid_moves(self):
        # Determine valid actions for the current player
        valid_actions = []
        if self.current_player == 1:
            distance_to_center = 5 - self.p1_pos
            if distance_to_center >= 1:
                target_pos = self.p1_pos + 1
                if self.is_valid_position(target_pos):
                    valid_actions.append(0)  # Move 1 position
            if distance_to_center >= 2:
                target_pos = self.p1_pos + 2
                if self.is_valid_position(target_pos):
                    valid_actions.append(1)  # Move 2 positions
        else:
            distance_to_center = self.p2_pos - 5
            if distance_to_center >= 1:
                target_pos = self.p2_pos - 1
                if self.is_valid_position(target_pos):
                    valid_actions.append(0)  # Move 1 position
            if distance_to_center >= 2:
                target_pos = self.p2_pos - 2
                if self.is_valid_position(target_pos):
                    valid_actions.append(1)  # Move 2 positions
        return valid_actions

    def is_valid_position(self, target_pos):
        opponent_pos = self.p2_pos if self.current_player == 1 else self.p1_pos
        # Cannot land on opponent's position unless at center
        if target_pos == opponent_pos and target_pos != 5:
            return False
        if target_pos < 0 or target_pos > 10:
            return False
        return True

    def render(self):
        # Visual representation of the game state
        track = ["   "] * 11
        if self.p1_pos == self.p2_pos and self.p1_pos == 5:
            track[5] = "P1P2"
        else:
            if self.p1_pos != 5:
                track[self.p1_pos] = " P1"
            if self.p2_pos != 5:
                track[self.p2_pos] = " P2"
            if track[5].strip() == "":
                track[5] = " C "
        track_str = "|".join(track)
        return f"Track: {track_str}\nCurrent player: {'Player 1' if self.current_player == 1 else 'Player 2'}"

    def valid_moves(self):
        # Return list of valid actions for the current player
        valid_actions = []
        if self.current_player == 1:
            distance_to_center = 5 - self.p1_pos
            if distance_to_center >= 1:
                target_pos = self.p1_pos + 1
                if self.is_valid_position(target_pos):
                    valid_actions.append(0)
            if distance_to_center >= 2:
                target_pos = self.p1_pos + 2
                if self.is_valid_position(target_pos):
                    valid_actions.append(1)
        else:
            distance_to_center = self.p2_pos - 5
            if distance_to_center >= 1:
                target_pos = self.p2_pos - 1
                if self.is_valid_position(target_pos):
                    valid_actions.append(0)
            if distance_to_center >= 2:
                target_pos = self.p2_pos - 2
                if self.is_valid_position(target_pos):
                    valid_actions.append(1)
        return valid_actions
