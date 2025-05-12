import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2, shape=(22,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Positions of soldiers: [Soldier1_pos, Soldier2_pos]
        self.player_soldiers = [0, 0]  # Player 1's soldiers start at position 0
        self.opponent_soldiers = [10, 10]  # Player 2's soldiers start at position 10

        return self._get_observation(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self._get_observation(), -10, True, False, {}  # Invalid move

        # Map the action to soldier index and steps
        action_map = {
            0: (0, 1),  # Move Soldier1 forward 1 position
            1: (0, 2),  # Move Soldier1 forward 2 positions
            2: (1, 1),  # Move Soldier2 forward 1 position
            3: (1, 2),  # Move Soldier2 forward 2 positions
        }

        soldier_idx, steps = action_map[action]
        # Get current position of the soldier
        soldier_pos = self.player_soldiers[soldier_idx]

        # Determine the movement direction
        direction = 1 if self.current_player == 1 else -1
        new_pos = soldier_pos + direction * steps

        # Check if the move is within bounds
        if new_pos < 0 or new_pos > 10:
            self.done = True
            return self._get_observation(), -10, True, False, {}  # Invalid move

        # Update soldier position
        self.player_soldiers[soldier_idx] = new_pos

        # Check for combat
        opponent_indices = [
            idx for idx, pos in enumerate(self.opponent_soldiers) if pos == new_pos
        ]
        if opponent_indices:
            # Send opponent's soldiers back to their flag
            opponent_flag = 0 if self.current_player == -1 else 10
            for idx in opponent_indices:
                self.opponent_soldiers[idx] = opponent_flag

        # Check for victory
        opponent_flag = 10 if self.current_player == 1 else 0
        if new_pos == opponent_flag:
            self.done = True
            return self._get_observation(), 1, True, False, {}  # Current player wins

        # Switch turns
        self.current_player *= -1
        self.player_soldiers, self.opponent_soldiers = (
            self.opponent_soldiers,
            self.player_soldiers,
        )

        return (
            self._get_observation(),
            0,
            False,
            False,
            {},
        )  # Observation, reward, done, info

    def render(self):
        board_str = ""
        for pos in range(11):
            p1_soldiers = [
                f"S{idx +1}"
                for idx, s_pos in enumerate(self.player_soldiers)
                if s_pos == pos
            ]
            p2_soldiers = [
                f"S{idx +3}"
                for idx, s_pos in enumerate(self.opponent_soldiers)
                if s_pos == pos
            ]
            cell = f"Position {pos}: "
            if pos == 0:
                cell += "[P1_Flag] "
            elif pos == 10:
                cell += "[P2_Flag] "
            cell += " ".join(p1_soldiers + p2_soldiers)
            board_str += cell + "\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        action_map = {
            0: (0, 1),  # Move Soldier1 forward 1 position
            1: (0, 2),  # Move Soldier1 forward 2 positions
            2: (1, 1),  # Move Soldier2 forward 1 position
            3: (1, 2),  # Move Soldier2 forward 2 positions
        }
        direction = 1 if self.current_player == 1 else -1
        for action, (soldier_idx, steps) in action_map.items():
            soldier_pos = self.player_soldiers[soldier_idx]
            new_pos = soldier_pos + direction * steps
            if 0 <= new_pos <= 10:
                valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        obs = np.zeros(22, dtype=np.int8)
        # Current player's soldiers
        for pos in self.player_soldiers:
            idx = pos * 2  # Positions are from 0 to 10
            obs[idx] += 1  # Increment count at position for current player
        # Opponent's soldiers
        for pos in self.opponent_soldiers:
            idx = pos * 2 + 1
            obs[idx] += 1  # Increment count at position for opponent
        return obs
