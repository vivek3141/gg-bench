import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 70 possible actions (10 movement choices * 7 observation choices)
        self.action_space = spaces.Discrete(70)

        # Observation space: Position of both players, quantum tunneling used, previous observations, current player
        # [player1_position, player1_qt_used, player1_prev_obs,
        #  player2_position, player2_qt_used, player2_prev_obs,
        #  current_player]
        self.observation_space = spaces.Box(low=0, high=7, shape=(7,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Positions: 0 (off-grid), 1-7 (grid positions)
        self.player_positions = [0, 0]
        # Quantum tunneling used: False (0) or True (1)
        self.quantum_tunneling_used = [0, 0]
        # Previous observations (positions 1-7), 0 if none
        self.prev_observations = [0, 0]
        # Current player: 0 or 1
        self.current_player = 0
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), -10, True, False, {}

        movement_choice = action // 7
        observation_choice = (action % 7) + 1  # Positions 1-7

        opponent = 1 - self.current_player
        player_pos = self.player_positions[self.current_player]
        opponent_pos = self.player_positions[opponent]
        qt_used = self.quantum_tunneling_used[self.current_player]
        prev_obs = self.prev_observations[self.current_player]
        reward = 0

        # Validate observation (cannot be same as previous observation)
        if prev_obs == observation_choice and prev_obs != 0:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Validate movement
        new_position = player_pos
        if movement_choice == 0:  # Move Left
            if player_pos == 1 or player_pos == 0:
                self.done = True
                return self._get_obs(), -10, True, False, {}
            else:
                new_position -= 1
        elif movement_choice == 1:  # Move Right
            if player_pos == 7 or player_pos == 0:
                self.done = True
                return self._get_obs(), -10, True, False, {}
            else:
                new_position += 1
        elif movement_choice == 2:  # Stay
            if player_pos == 0:
                self.done = True
                return self._get_obs(), -10, True, False, {}
            else:
                new_position = player_pos
        elif 3 <= movement_choice <= 9:  # Quantum Tunneling
            if qt_used == 1:
                self.done = True
                return self._get_obs(), -10, True, False, {}
            else:
                qt_target = movement_choice - 2  # Positions 1-7
                new_position = qt_target
                self.quantum_tunneling_used[self.current_player] = 1
        else:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Update player's position
        self.player_positions[self.current_player] = new_position

        # Collision check
        if (
            self.player_positions[self.current_player]
            == self.player_positions[opponent]
            and self.player_positions[self.current_player] != 0
        ):
            # Determine who initiated the collision
            # For simplicity, we assume the player who moved into the position this turn initiated the collision
            # Since both players' positions are updated simultaneously, we need to track previous positions
            if self.current_player == self.current_player:
                # Current player initiated collision
                self.done = True
                reward = -10
                return self._get_obs(), reward, True, False, {}
            else:
                # Opponent initiated collision
                self.done = True
                reward = 1
                return self._get_obs(), reward, True, False, {}

        # Observation Phase
        if self.player_positions[opponent] == observation_choice:
            # Successful observation
            self.done = True
            reward = 1
            return self._get_obs(), reward, True, False, {}

        # Update previous observation
        self.prev_observations[self.current_player] = observation_choice

        # Switch current player
        self.current_player = opponent

        return self._get_obs(), reward, False, False, {}

    def render(self):
        # Visual representation of the environment state
        grid = ["[ ]"] * 7
        for i in range(7):
            pos = i + 1
            if self.player_positions[0] == pos and self.player_positions[1] == pos:
                grid[i] = "[Q1&Q2]"
            elif self.player_positions[0] == pos:
                grid[i] = "[Q1]"
            elif self.player_positions[1] == pos:
                grid[i] = "[Q2]"
        grid_str = " ".join(grid)
        return (
            f"Grid: {grid_str}\n"
            f"Player 1 position: {self.player_positions[0]}, Quantum Tunneling used: {self.quantum_tunneling_used[0]}, Previous observation: {self.prev_observations[0]}\n"
            f"Player 2 position: {self.player_positions[1]}, Quantum Tunneling used: {self.quantum_tunneling_used[1]}, Previous observation: {self.prev_observations[1]}\n"
            f"Current player: Player {self.current_player + 1}"
        )

    def valid_moves(self):
        # Returns a list of valid action indices for the current player
        valid_actions = []
        player_pos = self.player_positions[self.current_player]
        qt_used = self.quantum_tunneling_used[self.current_player]
        prev_obs = self.prev_observations[self.current_player]

        movement_choices = []
        # Regular movement
        if player_pos == 0:
            # Off-grid, invalid move unless quantum tunneling
            pass
        else:
            if player_pos > 1:
                movement_choices.append(0)  # Move Left
            if player_pos < 7:
                movement_choices.append(1)  # Move Right
            movement_choices.append(2)  # Stay

        # Quantum Tunneling
        if qt_used == 0:
            for qt_target in range(1, 8):
                movement_choices.append(2 + qt_target)  # movement_choice 3-9

        # Observation choices (positions 1-7), cannot be same as previous observation
        observation_choices = [pos for pos in range(1, 8) if pos != prev_obs]

        for m in movement_choices:
            for o in observation_choices:
                action_id = m * 7 + (o - 1)
                valid_actions.append(action_id)
        return valid_actions

    def _get_obs(self):
        return np.array(
            [
                self.player_positions[0],
                self.quantum_tunneling_used[0],
                self.prev_observations[0],
                self.player_positions[1],
                self.quantum_tunneling_used[1],
                self.prev_observations[1],
                self.current_player,
            ],
            dtype=np.int8,
        )
