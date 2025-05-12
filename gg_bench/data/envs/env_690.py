import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0-14 for placing tiles, 15 for pass, 16-18 for sudden death remove actions
        self.action_space = spaces.Discrete(19)
        self.observation_space = spaces.Box(low=0, high=5, shape=(9,), dtype=np.int32)

        # Initialize game state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initial game state
        self.towers = [[], [], []]  # Towers A, B, C
        self.player_hands = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]  # Two players
        self.current_player = 0  # Player 0 starts
        self.done = False
        self.passes = [0, 0]  # Pass counts for each player
        self.last_player_to_place = None  # Last player who placed a tile
        self.sudden_death = False  # Flag for sudden death mode
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        if self.sudden_death:
            # Handle sudden death actions
            tower_index = action - 16
            # Remove top tile from tower
            if len(self.towers[tower_index]) > 0:
                removed_tile, removed_player = self.towers[tower_index].pop()
                # After removing tile, check if players have valid moves
                if self.has_valid_moves(self.current_player) or self.has_valid_moves(
                    1 - self.current_player
                ):
                    # Exit sudden death
                    self.sudden_death = False
                    # Switch to next player
                    self.current_player = 1 - self.current_player
                    observation = self._get_observation()
                    return observation, 0, False, False, {}
                else:
                    # Neither player has valid moves, continue sudden death
                    # Check if all towers are empty
                    if all(len(tower) == 0 for tower in self.towers):
                        # Game over, declare tie
                        self.done = True
                        return self._get_observation(), 0, True, False, {}
                    else:
                        # Switch player and continue sudden death
                        self.current_player = 1 - self.current_player
                        observation = self._get_observation()
                        return observation, 0, False, False, {}
            else:
                # Invalid move, cannot remove tile from empty tower
                self.done = True
                return self._get_observation(), -10, True, False, {}
        else:
            if action == 15:
                # Player passes
                self.passes[self.current_player] = 1
                # Check for consecutive passes
                if self.passes[0] == 1 and self.passes[1] == 1:
                    # Both players passed consecutively
                    tiles_remaining_p0 = len(self.player_hands[0])
                    tiles_remaining_p1 = len(self.player_hands[1])
                    if tiles_remaining_p0 < tiles_remaining_p1:
                        # Player 0 wins
                        reward = 1 if self.current_player == 0 else -1
                        self.done = True
                        return self._get_observation(), reward, True, False, {}
                    elif tiles_remaining_p1 < tiles_remaining_p0:
                        # Player 1 wins
                        reward = 1 if self.current_player == 1 else -1
                        self.done = True
                        return self._get_observation(), reward, True, False, {}
                    else:
                        # Tie-breaker: sum of tile numbers
                        sum_p0 = sum(self.player_hands[0])
                        sum_p1 = sum(self.player_hands[1])
                        if sum_p0 < sum_p1:
                            # Player 0 wins
                            reward = 1 if self.current_player == 0 else -1
                            self.done = True
                            return self._get_observation(), reward, True, False, {}
                        elif sum_p1 < sum_p0:
                            # Player 1 wins
                            reward = 1 if self.current_player == 1 else -1
                            self.done = True
                            return self._get_observation(), reward, True, False, {}
                        else:
                            # Sudden death proceeds
                            self.sudden_death = True
                            # Start sudden death with last player who placed a tile
                            if self.last_player_to_place is not None:
                                self.current_player = self.last_player_to_place
                            observation = self._get_observation()
                            return observation, 0, False, False, {}
                else:
                    # Switch to next player
                    self.passes[1 - self.current_player] = (
                        0  # Reset opponent's pass count
                    )
                    self.current_player = 1 - self.current_player
                    observation = self._get_observation()
                    return observation, 0, False, False, {}
            else:
                # Map action index to (tile, tower)
                tile_tower_pairs = [
                    (tile, tower) for tile in [1, 2, 3, 4, 5] for tower in [0, 1, 2]
                ]
                tile, tower_index = tile_tower_pairs[action]

                # Valid move
                self.player_hands[self.current_player].remove(tile)
                self.towers[tower_index].append((tile, self.current_player))
                self.passes[self.current_player] = 0  # Reset pass count
                self.passes[1 - self.current_player] = 0  # Reset opponent's pass count
                self.last_player_to_place = (
                    self.current_player
                )  # Update last player who placed a tile

                # Check for win condition
                if len(self.player_hands[self.current_player]) == 0:
                    # Current player wins
                    self.done = True
                    return self._get_observation(), 1, True, False, {}

                # Switch to next player
                self.current_player = 1 - self.current_player
                observation = self._get_observation()
                return observation, 0, False, False, {}

    def _get_observation(self):
        observation = np.zeros(9, dtype=np.int32)

        # Top tiles of towers A, B, C
        for i in range(3):
            if len(self.towers[i]) > 0:
                observation[i] = self.towers[i][-1][0]
            else:
                observation[i] = 0

        # Tiles in current player's hand
        hand = self.player_hands[self.current_player]
        for tile_number in range(1, 6):
            if tile_number in hand:
                observation[2 + tile_number] = 1
            else:
                observation[2 + tile_number] = 0

        # Opponent's number of tiles remaining
        observation[8] = len(self.player_hands[1 - self.current_player])

        return observation

    def render(self):
        towers_str = ""
        for idx, tower in enumerate(self.towers):
            tower_str = "Tower " + chr(ord("A") + idx) + ": "
            if len(tower) == 0:
                tower_str += "Empty"
            else:
                tiles_str = " ".join(str(tile) for tile, player in tower)
                tower_str += tiles_str
            towers_str += tower_str + "\n"

        hand_str = "Your Hand: " + " ".join(
            map(str, self.player_hands[self.current_player])
        )
        opponent_tiles = len(self.player_hands[1 - self.current_player])
        opponent_str = f"Opponent has {opponent_tiles} tile(s) remaining."

        game_state_str = f"{towers_str}\n{hand_str}\n\n{opponent_str}\n"
        return game_state_str

    def valid_moves(self):
        if self.sudden_death:
            # In sudden death, valid actions are removing top tile from any tower with tiles
            valid_actions = []
            for i in range(3):
                if len(self.towers[i]) > 0:
                    valid_actions.append(16 + i)
            return valid_actions
        else:
            valid_actions = []
            # Build list of tile-tower pairs
            tile_tower_pairs = [
                (tile, tower) for tile in [1, 2, 3, 4, 5] for tower in [0, 1, 2]
            ]

            for idx, (tile, tower_index) in enumerate(tile_tower_pairs):
                if tile in self.player_hands[self.current_player]:
                    tower = self.towers[tower_index]
                    if len(tower) == 0 or tile > tower[-1][0]:
                        valid_actions.append(idx)

            if len(valid_actions) == 0:
                # No valid moves, must pass
                valid_actions.append(15)

            return valid_actions

    def has_valid_moves(self, player_index):
        hand = self.player_hands[player_index]
        for tile in hand:
            for tower_index in range(3):
                tower = self.towers[tower_index]
                if len(tower) == 0 or tile > tower[-1][0]:
                    return True
        return False
