import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions are 0, 1, 2 corresponding to moving 1, 2, or 3 steps
        self.action_space = spaces.Discrete(3)

        # Observation space: Shape (13,)
        # Index 0: Current player (1 or -1)
        # Index 1: Player 1's position (0 to 10)
        # Index 2: Player 2's position (0 to 10)
        # Indexes 3-12: Steps 1-10:
        #   0: No trap
        #   1: Trap set by Player 1, unrevealed
        #   2: Trap set by Player 1, revealed
        #   -1: Trap set by Player 2, unrevealed
        #   -2: Trap set by Player 2, revealed

        self.observation_space = spaces.Box(
            low=-2, high=2, shape=(13,), dtype=np.float32
        )

        # Initialize the game state
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        # Initialize player positions
        self.player_positions = {1: 0, -1: 0}

        # Randomly assign who starts
        self.current_player = self.np_random.choice([1, -1])

        # Randomly set traps for both players
        # Traps are sets of three distinct integers between 1 and 10
        all_steps = np.arange(1, 11)
        self.traps = {
            1: set(self.np_random.choice(all_steps, size=3, replace=False)),
            -1: set(self.np_random.choice(all_steps, size=3, replace=False)),
        }

        # Keep track of which traps have been revealed
        self.revealed_traps = {}
        for player in [1, -1]:
            self.revealed_traps[player] = set()

        self.done = False

        # Update observation
        self._update_observation()

        return self.observation, {}

    def _update_observation(self):
        # Observation is a numpy array of shape (13,)
        self.observation = np.zeros(13, dtype=np.float32)

        self.observation[0] = self.current_player
        self.observation[1] = self.player_positions[1]
        self.observation[2] = self.player_positions[-1]

        for idx, step in enumerate(range(1, 11), start=3):
            value = 0
            # Check traps for both players
            if step in self.traps[1]:
                if step in self.revealed_traps[1]:
                    value = 2  # Player 1's trap revealed
                else:
                    value = 1  # Player 1's trap unrevealed
            if step in self.traps[-1]:
                if step in self.revealed_traps[-1]:
                    value = -2  # Player 2's trap revealed
                else:
                    value = -1  # Player 2's trap unrevealed
            self.observation[idx] = value

    def valid_moves(self):
        # Return a list of valid actions (indices of action_space)
        current_pos = self.player_positions[self.current_player]
        valid_moves = []
        for i, steps in enumerate([1, 2, 3]):
            new_pos = current_pos + steps
            if new_pos <= 10:
                valid_moves.append(i)
        return valid_moves

    def step(self, action):
        if self.done:
            return self.observation, 0, True, False, {}

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            self._update_observation()
            return self.observation, -10, True, False, {}

        steps = action + 1  # action 0 corresponds to 1 step, etc.
        current_pos = self.player_positions[self.current_player]
        new_pos = current_pos + steps

        opponent = -self.current_player
        reward = 0

        # Move the player
        self.player_positions[self.current_player] = new_pos

        # Check for trap
        if new_pos in self.traps[opponent]:
            # Trap triggered
            self.player_positions[self.current_player] = 0
            # Mark the trap as revealed and deactivate it
            self.revealed_traps[opponent].add(new_pos)
            self.traps[opponent].remove(new_pos)
            # Update observation
            self._update_observation()
            # No reward, continue the game
        else:
            # Check for win
            if new_pos == 10:
                # Current player wins
                self.done = True
                reward = 1
                self._update_observation()
                return self.observation, reward, True, False, {}

        # Update observation
        self._update_observation()
        # Switch player
        self.current_player = opponent

        return self.observation, reward, False, False, {}

    def render(self):
        # Return a string representation of the current game state
        s = f"Current player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        s += f"Player 1 position: {int(self.player_positions[1])}\n"
        s += f"Player 2 position: {int(self.player_positions[-1])}\n"
        s += "Revealed traps:\n"
        for player in [1, -1]:
            traps = self.revealed_traps[player]
            if traps:
                s += f"  Player {'1' if player == 1 else '2'}'s traps revealed at steps: {sorted(traps)}\n"
            else:
                s += f"  Player {'1' if player == 1 else '2'} has no traps revealed.\n"
        s += "Board:\n"
        board = ""
        for i in range(1, 11):
            if i == self.player_positions[1] and i == self.player_positions[-1]:
                board += "[B]"  # Both players on the same step
            elif i == self.player_positions[1]:
                board += "[1]"
            elif i == self.player_positions[-1]:
                board += "[2]"
            else:
                board += "[ ]"
            if i % 5 == 0:
                board += "\n"
        s += board
        return s
