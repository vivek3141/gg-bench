import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers 1-10 (indices 0-9)
        self.action_space = spaces.Discrete(10)

        # Observation space:
        # - [0]: Current player's life points (0-20)
        # - [1]: Opponent's life points (0-20)
        # - [2:12]: Current player's remaining uses of numbers 1-10 (each 0-2)
        # - [12:22]: Opponent's remaining uses of numbers 1-10 (each 0-2)
        # - [22]: Pending attack number (0 if none)
        # - [23]: Phase indicator (0: Attack Phase, 1: Defense Phase)
        self.observation_space = spaces.Box(low=0, high=20, shape=(24,), dtype=np.int32)

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Life points for players [player0, player1]
        self.life_points = [20, 20]
        # Number uses for each player [ [uses of numbers 1-10], [uses of numbers 1-10] ]
        self.num_uses = [np.zeros(10, dtype=np.int32) for _ in range(2)]
        # Current player: 0 or 1
        self.current_player = 0
        # Phase: 0 for Attack Phase, 1 for Defense Phase
        self.phase = 0
        # Pending attack number
        self.pending_attack_number = 0
        # Observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        action = int(action) + 1  # Convert action space index to number (1-10)
        reward = 0
        done = False
        info = {}

        # Validate action: must be between 1 and 10
        if action < 1 or action > 10:
            return self._get_observation(), -10, True, False, info  # Invalid move

        # Check if number has remaining uses
        if self.num_uses[self.current_player][action - 1] >= 2:
            return self._get_observation(), -10, True, False, info  # Invalid move

        # Update number uses
        self.num_uses[self.current_player][action - 1] += 1

        if self.phase == 0:  # Attack Phase
            # Store pending attack number
            self.pending_attack_number = action
            # Switch to Defense Phase
            self.phase = 1
            # No reward yet
            return self._get_observation(), 0, False, False, info

        elif self.phase == 1:  # Defense Phase
            defense_number = action
            attack_number = self.pending_attack_number

            # Resolve attack
            if defense_number >= attack_number:
                # Attack blocked
                damage = 0
            else:
                # Successful attack
                damage = attack_number - defense_number
                defender = (self.current_player + 1) % 2
                self.life_points[defender] -= damage

            # Check for win condition
            defender = (self.current_player + 1) % 2
            if self.life_points[defender] <= 0:
                # Current player wins
                reward = 1
                done = True
            else:
                # Continue game
                reward = 0
                done = False

            # Reset pending attack number
            self.pending_attack_number = 0

            # Swap roles
            self.current_player = (self.current_player + 1) % 2
            # Switch to Attack Phase
            self.phase = 0

            return self._get_observation(), reward, done, False, info

    def render(self):
        attacker = (
            self.current_player if self.phase == 0 else (self.current_player + 1) % 2
        )
        defender = (
            (self.current_player + 1) % 2 if self.phase == 0 else self.current_player
        )
        phase_str = "Attack Phase" if self.phase == 0 else "Defense Phase"
        pending_attack = (
            self.pending_attack_number if self.pending_attack_number > 0 else "None"
        )

        render_str = "--- Number Clash Game State ---\n"
        render_str += f"Phase: {phase_str}\n"
        render_str += f"Current Player (Attacker): Player {attacker + 1}\n"
        render_str += f"Opponent (Defender): Player {defender + 1}\n"
        render_str += f"Pending Attack Number: {pending_attack}\n"
        render_str += "Life Points:\n"
        render_str += f"  Player 1: {self.life_points[0]} LP\n"
        render_str += f"  Player 2: {self.life_points[1]} LP\n"
        render_str += f"Number Uses (Player {self.current_player + 1}):\n"
        for i in range(10):
            uses = self.num_uses[self.current_player][i]
            render_str += f"  Number {i + 1}: Used {uses}/2 times\n"
        render_str += f"Number Uses (Player {((self.current_player + 1) % 2) + 1}):\n"
        for i in range(10):
            uses = self.num_uses[(self.current_player + 1) % 2][i]
            render_str += f"  Number {i + 1}: Used {uses}/2 times\n"
        return render_str

    def valid_moves(self):
        # Return a list of valid numbers (indices 0-9 for numbers 1-10)
        valid_numbers = []
        for i in range(10):
            if self.num_uses[self.current_player][i] < 2:
                valid_numbers.append(i)
        return valid_numbers

    def _get_observation(self):
        # Construct the observation array
        observation = np.zeros(24, dtype=np.int32)

        # Current player's life points
        observation[0] = self.life_points[self.current_player]
        # Opponent's life points
        opponent = (self.current_player + 1) % 2
        observation[1] = self.life_points[opponent]
        # Current player's number uses (positions 2-11)
        observation[2:12] = 2 - self.num_uses[self.current_player]
        # Opponent's number uses (positions 12-21)
        observation[12:22] = 2 - self.num_uses[opponent]
        # Pending attack number (position 22)
        observation[22] = self.pending_attack_number
        # Phase indicator (position 23)
        observation[23] = self.phase  # 0: Attack Phase, 1: Defense Phase

        return observation
