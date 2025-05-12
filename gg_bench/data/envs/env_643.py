import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        # Actions: 6 possible actions (move 1-3 squares, with or without setting a trap)
        self.action_space = spaces.Discrete(6)

        # Observation space: An array of 33 elements
        # - Index 0-10: Current player's position (1 if present, 0 otherwise)
        # - Index 11-21: Opponent's position (1 if present, 0 otherwise)
        # - Index 22-32: Current player's own traps (1 if trap is set on the square, 0 otherwise)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(33,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Positions of players: {player_id: position}
        self.player_positions = {1: 0, -1: 0}
        # Traps set by each player: {player_id: set of positions}
        self.player_traps = {1: set(), -1: set()}
        # Flags indicating if a player will be sent back to square 0
        self.player_trapped_next_turn = {1: False, -1: False}
        # Current player (1 or -1)
        self.current_player = 1
        # Game over flag
        self.done = False
        # Return the initial observation and info
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if the action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Decode the action into move amount and trap setting
        move_amount = (action // 2) + 1  # Actions are mapped to move amounts 1-3
        set_trap = (action % 2) == 1  # Actions with odd indices involve setting a trap

        # Handle being sent back to square 0 due to a trap
        if self.player_trapped_next_turn[self.current_player]:
            self.player_positions[self.current_player] = 0
            self.player_trapped_next_turn[self.current_player] = False

        current_position = self.player_positions[self.current_player]
        new_position = current_position + move_amount

        # Check if moving past square 10 (should not happen with valid actions)
        if new_position > 10:
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Move the player
        self.player_positions[self.current_player] = new_position

        # Handle trap setting
        if set_trap:
            if current_position == 0:
                # Cannot set a trap on square 0
                self.done = True
                return self._get_observation(), -10, True, False, {}
            elif (
                current_position in self.player_traps[1]
                or current_position in self.player_traps[-1]
            ):
                # Cannot set a trap on a square that already has a trap
                self.done = True
                return self._get_observation(), -10, True, False, {}
            else:
                # Set the trap
                self.player_traps[self.current_player].add(current_position)

        # Check for victory
        if new_position == 10:
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Check if the player lands on an opponent's trap
        opponent = -self.current_player
        if new_position in self.player_traps[opponent]:
            # Player will be sent back to square 0 at the start of their next turn
            self.player_trapped_next_turn[self.current_player] = True

        # Prepare for the next turn
        reward = -10  # Penalize each move to encourage faster victory
        self.current_player = opponent  # Switch turns
        return self._get_observation(), reward, False, False, {}

    def render(self):
        # Create a visual representation of the track
        track = []
        for i in range(11):
            cell = " "
            if self.player_positions[1] == i and self.player_positions[-1] == i:
                # Both players on the same square
                cell = "X"
            elif self.player_positions[1] == i:
                cell = "A"  # Player 1
            elif self.player_positions[-1] == i:
                cell = "B"  # Player -1
            else:
                cell = str(i)
            track.append(cell)
        # Include traps in the visual representation
        trap_info = ""
        for i in range(1, 10):
            trap = ""
            if i in self.player_traps[1]:
                trap += "A"
            if i in self.player_traps[-1]:
                trap += "B"
            if trap == "":
                trap = " "
            trap_info += f" {trap} "
        # Combine track and trap info
        track_str = "Track: " + " | ".join(track) + "\n"
        trap_str = "Traps:" + trap_info + "\n"
        return track_str + trap_str

    def valid_moves(self):
        valid_actions = []
        current_position = self.player_positions[self.current_player]

        # Adjust position if the player will be sent back to square 0
        if self.player_trapped_next_turn[self.current_player]:
            adjusted_position = 0
        else:
            adjusted_position = current_position

        for action in range(6):
            move_amount = (action // 2) + 1
            set_trap = (action % 2) == 1

            new_position = adjusted_position + move_amount
            # Cannot overshoot square 10
            if new_position > 10:
                continue

            # Validate trap setting
            trap_valid = True
            if set_trap:
                if adjusted_position == 0:
                    trap_valid = False  # Can't set a trap on square 0
                elif (
                    adjusted_position in self.player_traps[1]
                    or adjusted_position in self.player_traps[-1]
                ):
                    trap_valid = (
                        False  # Can't set a trap on a square that already has a trap
                    )

            if trap_valid:
                valid_actions.append(action)

        return valid_actions

    def _get_observation(self):
        # Initialize the observation array
        obs = np.zeros(33, dtype=np.float32)

        # Current player's position
        current_player_pos = self.player_positions[self.current_player]
        obs[current_player_pos] = 1.0  # Index 0-10

        # Opponent's position
        opponent_pos = self.player_positions[-self.current_player]
        obs[11 + opponent_pos] = 1.0  # Index 11-21

        # Current player's own traps
        for trap_pos in self.player_traps[self.current_player]:
            obs[22 + trap_pos] = 1.0  # Index 22-32

        return obs
