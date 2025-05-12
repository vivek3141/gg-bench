import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        # Action space: Discrete actions from 0 to 65
        self.action_space = spaces.Discrete(66)

        # Observation space: 26 integers representing the game state
        # - For each of the 11 positions:
        #   - Owner (0: empty, 1: Player 1, 2: Player 2)
        #   - Power level (0 if empty, 1-10 if occupied)
        # - Current player's energy reserve (0-15)
        # - Current player's available nodes (0-5)
        # - Opponent's energy reserve (0-15)
        # - Opponent's available nodes (0-5)
        self.observation_space = spaces.Box(low=0, high=15, shape=(26,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the Power Line and player states
        self.power_line = np.zeros(
            (11, 2), dtype=np.int32
        )  # Each position has [owner, power_level]
        self.players = {
            1: {
                "energy": 15,
                "available_nodes": 5,
            },
            2: {
                "energy": 15,
                "available_nodes": 5,
            },
        }
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.info = {}
        return self._get_observation(), self.info

    def step(self, action):
        # If the game is already over, penalize the player
        if self.done:
            return self._get_observation(), -10, True, False, {}

        reward = 0

        # Map the action to the corresponding move
        if action < 55:
            # Place a new power node
            position = action // 5
            power_level = (action % 5) + 1  # Power level between 1 and 5

            # Check if the move is valid
            if self.power_line[position, 0] != 0:
                # Position is already occupied
                reward = -10
                self.done = True
                return self._get_observation(), reward, self.done, False, {}

            if self.players[self.current_player]["available_nodes"] <= 0:
                # No available nodes to place
                reward = -10
                self.done = True
                return self._get_observation(), reward, self.done, False, {}

            energy_cost = power_level
            if self.players[self.current_player]["energy"] < energy_cost:
                # Not enough energy to place the node
                reward = -10
                self.done = True
                return self._get_observation(), reward, self.done, False, {}

            # Place the node
            self.power_line[position, 0] = self.current_player
            self.power_line[position, 1] = power_level
            self.players[self.current_player]["energy"] -= energy_cost
            self.players[self.current_player]["available_nodes"] -= 1

            # Check for capturing mechanics
            self._check_capturing(position)
        elif 55 <= action <= 65:
            # Upgrade an existing power node
            position = action - 55

            # Check if the move is valid
            if self.power_line[position, 0] != self.current_player:
                # No player's node at the position
                reward = -10
                self.done = True
                return self._get_observation(), reward, self.done, False, {}

            current_power = self.power_line[position, 1]
            if current_power >= 10:
                # Node is already at maximum power level
                reward = -10
                self.done = True
                return self._get_observation(), reward, self.done, False, {}

            new_power = current_power + 1
            energy_cost = new_power
            if self.players[self.current_player]["energy"] < energy_cost:
                # Not enough energy to upgrade the node
                reward = -10
                self.done = True
                return self._get_observation(), reward, self.done, False, {}

            # Upgrade the node
            self.power_line[position, 1] = new_power
            self.players[self.current_player]["energy"] -= energy_cost

            # Check for capturing mechanics
            self._check_capturing(position)
        else:
            # Invalid action
            reward = -10
            self.done = True
            return self._get_observation(), reward, self.done, False, {}

        # Check if the current player has won
        if self._check_victory():
            reward = 1
            self.done = True
            return self._get_observation(), reward, self.done, False, {}

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        return self._get_observation(), reward, self.done, False, {}

    def render(self):
        # Generate a visual representation of the Power Line and player states
        power_line_str = "Power Line:\n"
        for pos in range(11):
            owner = self.power_line[pos, 0]
            power = self.power_line[pos, 1]
            if owner == 0:
                power_line_str += "[ ]"
            elif owner == 1:
                power_line_str += f"[A({power})]"
            else:
                power_line_str += f"[B({power})]"
        power_line_str += "\n"

        energy_str = (
            f"Player 1 - Energy: {self.players[1]['energy']}, "
            f"Available Nodes: {self.players[1]['available_nodes']}\n"
            f"Player 2 - Energy: {self.players[2]['energy']}, "
            f"Available Nodes: {self.players[2]['available_nodes']}\n"
        )
        return power_line_str + energy_str

    def valid_moves(self):
        # Generate a list of valid actions for the current player
        valid_actions = []

        # Actions for placing a new node
        for position in range(11):
            if self.power_line[position, 0] == 0:
                for power_level in range(1, 6):
                    energy_cost = power_level
                    if (
                        self.players[self.current_player]["available_nodes"] > 0
                        and self.players[self.current_player]["energy"] >= energy_cost
                    ):
                        action = position * 5 + (power_level - 1)
                        valid_actions.append(action)

        # Actions for upgrading an existing node
        for position in range(11):
            if self.power_line[position, 0] == self.current_player:
                current_power = self.power_line[position, 1]
                if current_power < 10:
                    new_power = current_power + 1
                    energy_cost = new_power
                    if self.players[self.current_player]["energy"] >= energy_cost:
                        action = 55 + position
                        valid_actions.append(action)

        return valid_actions

    def _get_observation(self):
        # Build the observation array
        obs = np.zeros(26, dtype=np.int32)

        # Power Line state
        for pos in range(11):
            obs[pos * 2] = self.power_line[pos, 0]
            obs[pos * 2 + 1] = self.power_line[pos, 1]

        # Current player's energy and available nodes
        obs[22] = self.players[self.current_player]["energy"]
        obs[23] = self.players[self.current_player]["available_nodes"]

        # Opponent's energy and available nodes
        opponent = 2 if self.current_player == 1 else 1
        obs[24] = self.players[opponent]["energy"]
        obs[25] = self.players[opponent]["available_nodes"]

        return obs

    def _check_capturing(self, position):
        # Check for capturing mechanics after placing or upgrading a node
        def capture(pos):
            # Capture opponent's node if applicable
            if pos < 0 or pos > 10:
                return  # Out of bounds
            owner = self.power_line[pos, 0]
            if owner == 0 or owner == self.current_player:
                return  # Empty or own node
            # Compare power levels
            opponent_power = self.power_line[pos, 1]
            my_power = self.power_line[position, 1]
            if my_power > opponent_power:
                # Capture the opponent's node
                opponent = owner
                self.power_line[pos, 0] = 0
                self.power_line[pos, 1] = 0
                self.players[opponent]["available_nodes"] += 1
                # Recursively check for chain reactions
                capture(pos - 1)
                capture(pos + 1)

        # Check adjacent positions for capturing
        capture(position - 1)
        capture(position + 1)

    def _check_victory(self):
        # Check if the current player has an unbroken chain from position 0 to 10
        visited = set()
        components = []

        for pos in range(11):
            if self.power_line[pos, 0] == self.current_player and pos not in visited:
                # Perform DFS to find connected components
                component = set()
                stack = [pos]
                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    component.add(current)
                    # Add adjacent positions
                    for neighbor in [current - 1, current + 1]:
                        if 0 <= neighbor <= 10:
                            if (
                                self.power_line[neighbor, 0] == self.current_player
                                and neighbor not in visited
                            ):
                                stack.append(neighbor)
                components.append(component)

        # Check if any component connects position 0 and position 10
        for comp in components:
            if 0 in comp and 10 in comp:
                return True
        return False
