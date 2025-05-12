import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 6 possible actions:
        # Action 0: Option 1, remove 1 stone from own pile.
        # Action 1: Option 1, remove 2 stones from own pile.
        # Action 2: Option 1, remove 3 stones from own pile.
        # Action 3: Option 2, remove 1 stone from opponent's pile.
        # Action 4: Option 2, remove 2 stones from opponent's pile.
        # Action 5: Option 2, remove 3 stones from opponent's pile.
        self.action_space = spaces.Discrete(6)

        # Observation space represents [current_player_stones, opponent_stones]
        # Stones can be any non-negative integer, upper bound set to 100 for practical purposes
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_stones = 15
        self.player2_stones = 15
        self.current_player = 1  # Current player: 1 or 2
        self.done = False
        return self._get_observation(), {}

    def _get_observation(self):
        if self.current_player == 1:
            return np.array([self.player1_stones, self.player2_stones], dtype=np.int32)
        else:
            return np.array([self.player2_stones, self.player1_stones], dtype=np.int32)

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        reward = 0

        # Map action to move
        if action == 0:
            option = 1  # Self-reduce and transfer to opponent
            stones_to_remove = 1
        elif action == 1:
            option = 1
            stones_to_remove = 2
        elif action == 2:
            option = 1
            stones_to_remove = 3
        elif action == 3:
            option = 2  # Attack opponent's pile and steal stones
            stones_to_remove = 1
        elif action == 4:
            option = 2
            stones_to_remove = 2
        elif action == 5:
            option = 2
            stones_to_remove = 3

        # Execute action
        if self.current_player == 1:
            if option == 1:
                # Remove stones from own pile
                self.player1_stones -= stones_to_remove
                # Add half (rounded down) to opponent's pile
                stones_to_add = stones_to_remove // 2
                self.player2_stones += stones_to_add
            elif option == 2:
                # Remove stones from opponent's pile
                self.player2_stones -= stones_to_remove
                # Add half (rounded down) to own pile
                stones_to_add = stones_to_remove // 2
                self.player1_stones += stones_to_add
        else:
            if option == 1:
                # Remove stones from own pile
                self.player2_stones -= stones_to_remove
                # Add half (rounded down) to opponent's pile
                stones_to_add = stones_to_remove // 2
                self.player1_stones += stones_to_add
            elif option == 2:
                # Remove stones from opponent's pile
                self.player1_stones -= stones_to_remove
                # Add half (rounded down) to own pile
                stones_to_add = stones_to_remove // 2
                self.player2_stones += stones_to_add

        # Check game end condition
        if self.player1_stones <= 0 or self.player2_stones <= 0:
            self.done = True
            if (self.current_player == 1 and self.player2_stones <= 0) or (
                self.current_player == 2 and self.player1_stones <= 0
            ):
                # Current player wins
                reward = 1
            else:
                # Current player loses
                reward = -1
            return self._get_observation(), reward, True, False, {}

        # Switch current player
        self.current_player = 1 if self.current_player == 2 else 2

        return self._get_observation(), reward, False, False, {}

    def render(self):
        if self.current_player == 1:
            state = f"Player 1's turn.\nPlayer 1 stones: {self.player1_stones}\nPlayer 2 stones: {self.player2_stones}"
        else:
            state = f"Player 2's turn.\nPlayer 2 stones: {self.player2_stones}\nPlayer 1 stones: {self.player1_stones}"
        return state

    def valid_moves(self):
        if self.done:
            return []

        valid_actions = []

        if self.current_player == 1:
            own_stones = self.player1_stones
            opponent_stones = self.player2_stones
        else:
            own_stones = self.player2_stones
            opponent_stones = self.player1_stones

        # Option 1 actions (actions 0-2)
        for i in range(3):
            stones_to_remove = i + 1
            if own_stones >= stones_to_remove:
                valid_actions.append(i)

        # Option 2 actions (actions 3-5)
        for i in range(3, 6):
            stones_to_remove = i - 2
            if opponent_stones >= stones_to_remove:
                valid_actions.append(i)

        return valid_actions
