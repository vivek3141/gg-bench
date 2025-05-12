import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=1, target_number=20):
        super(CustomEnv, self).__init__()

        # Game parameters
        self.starting_number = starting_number
        self.target_number = target_number

        # Define action space: 0 -> Add 1, 1 -> Multiply by 2
        self.action_space = spaces.Discrete(2)

        # Define observation space
        # Observation consists of:
        #   - Current Number (float)
        #   - Last Operation Used by Self (-1 for None, 0 for Add 1, 1 for Multiply by 2)
        #   - Last Operation Used by Opponent (-1 for None, 0 for Add 1, 1 for Multiply by 2)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0]),
            high=np.array([float(self.target_number), 1.0, 1.0]),
            dtype=np.float32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = float(self.starting_number)
        self.last_op_self = -1  # -1 indicates no operation yet
        self.last_op_opponent = -1
        self.current_player = 1  # 1 or 2
        self.done = False
        observation = np.array(
            [self.current_number, self.last_op_self, self.last_op_opponent],
            dtype=np.float32,
        )
        info = {}
        return observation, info  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, ignore further actions
            observation = np.array(
                [self.current_number, self.last_op_self, self.last_op_opponent],
                dtype=np.float32,
            )
            info = {}
            return observation, 0.0, True, False, info  # No reward, game already over

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action penalized
            self.done = True
            reward = -10.0
            terminated = True
            truncated = False
            observation = np.array(
                [self.current_number, self.last_op_self, self.last_op_opponent],
                dtype=np.float32,
            )
            info = {}
            return observation, reward, terminated, truncated, info

        # Apply the action
        if action == 0:
            # Add 1
            new_number = self.current_number + 1
            op_used = 0
        elif action == 1:
            # Multiply by 2
            new_number = self.current_number * 2
            op_used = 1

        # Check for exceeding target number
        if new_number > self.target_number:
            # Current player loses
            self.current_number = new_number
            self.done = True
            reward = -10.0
            terminated = True
            truncated = False
            # Update last operation before ending game
            self.last_op_self, self.last_op_opponent = (
                op_used,
                self.last_op_self,
            )
            observation = np.array(
                [self.current_number, self.last_op_self, self.last_op_opponent],
                dtype=np.float32,
            )
            info = {}
            return observation, reward, terminated, truncated, info

        # Update game state
        self.current_number = new_number
        self.last_op_self, self.last_op_opponent = op_used, self.last_op_self

        # Check for win condition
        if self.current_number == self.target_number:
            # Current player wins
            self.done = True
            reward = 1.0
            terminated = True
            truncated = False
            observation = np.array(
                [self.current_number, self.last_op_self, self.last_op_opponent],
                dtype=np.float32,
            )
            info = {}
            return observation, reward, terminated, truncated, info

        # Game continues
        reward = 0.0
        terminated = False
        truncated = False

        # Swap players
        self.current_player = 2 if self.current_player == 1 else 1

        observation = np.array(
            [self.current_number, self.last_op_self, self.last_op_opponent],
            dtype=np.float32,
        )
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        player_str = f"Player {self.current_player}'s Turn"
        last_opponent = (
            "None"
            if self.last_op_opponent == -1
            else "Add 1" if self.last_op_opponent == 0 else "Multiply by 2"
        )
        last_self = (
            "None"
            if self.last_op_self == -1
            else "Add 1" if self.last_op_self == 0 else "Multiply by 2"
        )
        render_str = (
            f"{player_str}\n"
            f"Current Number: {self.current_number}\n"
            f"Your Last Operation: {last_self}\n"
            f"Opponent's Last Operation: {last_opponent}\n"
        )
        return render_str

    def valid_moves(self):
        # Determine valid moves for the current player
        valid_actions = []

        # Cannot use the same operation as last time
        possible_actions = [0, 1]  # 0 -> Add 1, 1 -> Multiply by 2
        actions_to_consider = [
            action for action in possible_actions if action != self.last_op_self
        ]

        for action in actions_to_consider:
            if action == 0:
                # Add 1
                result = self.current_number + 1
            elif action == 1:
                # Multiply by 2
                result = self.current_number * 2

            if result <= self.target_number:
                valid_actions.append(action)

        return valid_actions
