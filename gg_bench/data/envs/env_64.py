import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = move 1 position, 1 = move 2 positions, 2 = move 3 positions
        self.action_space = spaces.Discrete(3)

        # Observation: positions of Player 1 and Player 2 on the number line (0 to 10)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([10, 10]), shape=(2,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Positions: [Player 1 position, Player 2 position]
        self.positions = [0, 10]  # Player 1 at position 0, Player 2 at position 10
        self.current_player = 1  # 1 for Player 1's turn, 2 for Player 2's turn
        self.done = False
        return (
            np.array(self.positions, dtype=np.int32),
            {},
        )  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                np.array(self.positions, dtype=np.int32),
                -10,
                True,
                False,
                {},
            )  # Game is over

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            self.done = True
            return (
                np.array(self.positions, dtype=np.int32),
                -10,
                True,
                False,
                {},
            )  # Invalid move

        # Perform the action
        move_distance = (
            action + 1
        )  # action 0 -> move 1, action 1 -> move 2, action 2 -> move 3

        # Determine movement direction
        if self.current_player == 1:
            direction = 1  # Player 1 moves towards position 10
            opponent_player = 2
        else:
            direction = -1  # Player 2 moves towards position 0
            opponent_player = 1

        # Update position
        new_position = (
            self.positions[self.current_player - 1] + direction * move_distance
        )
        self.positions[self.current_player - 1] = new_position

        # Check for win conditions
        if (
            self.positions[self.current_player - 1]
            == self.positions[opponent_player - 1]
        ):
            # Captured the opponent
            self.done = True
            return np.array(self.positions, dtype=np.int32), 1, True, False, {}
        elif self.positions[self.current_player - 1] == (
            10 if self.current_player == 1 else 0
        ):
            # Reached opponent's base
            self.done = True
            return np.array(self.positions, dtype=np.int32), 1, True, False, {}

        # Switch to the other player
        self.current_player = opponent_player

        return (
            np.array(self.positions, dtype=np.int32),
            0,
            False,
            False,
            {},
        )  # Continue game

    def render(self):
        # Visual representation of the battlefield
        battlefield = [" "] * 11  # Positions from 0 to 10
        if self.positions[0] == self.positions[1]:
            # Both players on the same position
            battlefield[self.positions[0]] = "B"  # B for both
        else:
            battlefield[self.positions[0]] = "1"  # Player 1's position
            battlefield[self.positions[1]] = "2"  # Player 2's position
        battlefield_str = "|".join(battlefield)
        return (
            f"Battlefield:\n{battlefield_str}\n"
            f"Player 1 at position {self.positions[0]}, "
            f"Player 2 at position {self.positions[1]}"
        )

    def valid_moves(self):
        valid_actions = []
        player_pos = self.positions[self.current_player - 1]
        opponent_base = 10 if self.current_player == 1 else 0
        direction = 1 if self.current_player == 1 else -1

        for action in range(3):  # Actions: 0, 1, 2
            move_distance = action + 1
            new_position = player_pos + direction * move_distance
            # Check if the new position is within the battlefield boundaries
            if 0 <= new_position <= 10:
                valid_actions.append(action)
        return valid_actions
