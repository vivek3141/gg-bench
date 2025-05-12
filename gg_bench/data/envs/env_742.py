import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 9 numbers * 2 actions (Attack, Defend) = 18 discrete actions
        self.action_space = spaces.Discrete(18)  # Actions 0 to 17

        # Observation space:
        # [current_LP, current_SP, opponent_LP, opponent_SP, n1_available, ..., n9_available]
        # LP: 0 to 20, SP: 0 to 50 (arbitrary upper limit), Number availability: 0 or 1
        low_obs = np.array([0, 0, 0, 0] + [0] * 9, dtype=np.int32)
        high_obs = np.array([20, 50, 20, 50] + [1] * 9, dtype=np.int32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.int32)

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_LP = 20
        self.current_SP = 0
        self.opponent_LP = 20
        self.opponent_SP = 0
        self.number_pool = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Available numbers
        self.current_player = 0  # Player 0 starts
        self.done = False

        # Create initial observation
        self._create_observation()

        return self.observation, {}

    def _create_observation(self):
        # Observation: [current_LP, current_SP, opponent_LP, opponent_SP, n1_available, ..., n9_available]
        num_availability = [1 if num in self.number_pool else 0 for num in range(1, 10)]
        if self.current_player == 0:
            self.observation = np.array(
                [self.current_LP, self.current_SP, self.opponent_LP, self.opponent_SP]
                + num_availability,
                dtype=np.int32,
            )
        else:
            self.observation = np.array(
                [self.opponent_LP, self.opponent_SP, self.current_LP, self.current_SP]
                + num_availability,
                dtype=np.int32,
            )

    def step(self, action):
        if self.done:
            return self.observation, 0, True, False, {}

        if not self.action_space.contains(action):
            return self.observation, -10, True, False, {}

        # Decode action
        action_number = action // 2  # 0 to 8
        action_type = action % 2  # 0: Attack, 1: Defend
        number = action_number + 1  # Actual number (1 to 9)

        if number not in self.number_pool:
            # Invalid move: number already used
            return self.observation, -10, True, False, {}

        # Valid move, process action
        self.number_pool.remove(number)
        damage_dealt = 0

        if action_type == 0:
            # Attack
            if self.current_player == 0:
                # Player 1 attacks Player 2
                actual_damage = number - self.opponent_SP
                actual_damage = max(actual_damage, 0)
                damage_dealt = actual_damage
                self.opponent_SP = max(self.opponent_SP - number, 0)
                self.opponent_LP -= damage_dealt
            else:
                # Player 2 attacks Player 1
                actual_damage = number - self.current_SP
                actual_damage = max(actual_damage, 0)
                damage_dealt = actual_damage
                self.current_SP = max(self.current_SP - number, 0)
                self.current_LP -= damage_dealt
        elif action_type == 1:
            # Defend
            if self.current_player == 0:
                self.current_SP += number
            else:
                self.opponent_SP += number
        else:
            # Invalid action type
            return self.observation, -10, True, False, {}

        # Check for victory conditions
        terminated = False
        reward = 0

        if self.current_player == 0 and self.opponent_LP <= 0:
            # Player 1 wins
            terminated = True
            reward = 1
        elif self.current_player == 1 and self.current_LP <= 0:
            # Player 2 wins
            terminated = True
            reward = 1
        elif len(self.number_pool) == 0:
            # Number pool is empty, determine winner by LP
            if self.current_LP > self.opponent_LP:
                winner = self.current_player
            elif self.opponent_LP > self.current_LP:
                winner = 1 - self.current_player
            else:
                winner = -1  # Draw

            terminated = True
            if winner == self.current_player:
                reward = 1  # Current player wins
            elif winner == -1:
                reward = 0  # Draw
            else:
                reward = -1  # Current player loses
        else:
            # Game continues, switch player
            self.current_player = 1 - self.current_player
            reward = 0

        # Update observation
        self._create_observation()
        self.done = terminated

        return self.observation, reward, terminated, False, {}

    def render(self):
        divider = "---\n"
        lines = []
        lines.append(divider)
        lines.append(f"Player {self.current_player + 1}'s turn\n")
        lines.append(
            f"Player 1 LP: {(self.current_LP if self.current_player == 0 else self.opponent_LP)}, "
            f"SP: {(self.current_SP if self.current_player == 0 else self.opponent_SP)}\n"
        )
        lines.append(
            f"Player 2 LP: {(self.opponent_LP if self.current_player == 0 else self.current_LP)}, "
            f"SP: {(self.opponent_SP if self.current_player == 0 else self.current_SP)}\n"
        )
        lines.append("Available Numbers: ")
        available_numbers = [num for num in range(1, 10) if num in self.number_pool]
        lines.append(str(available_numbers) + "\n")
        lines.append(divider)
        return "".join(lines)

    def valid_moves(self):
        valid_actions = []
        for num in self.number_pool:
            action_number = num - 1  # 0 to 8
            for action_type in [0, 1]:  # Attack and Defend
                action = action_number * 2 + action_type
                valid_actions.append(action)
        return valid_actions
