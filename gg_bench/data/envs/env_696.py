import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: scan_option (0-9), entangle_option (0-10), guess_position (0-24), guess_label (0-4)
        # Total actions = 10 * 11 * 25 * 5 = 13,750
        self.action_space = spaces.Discrete(13750)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(5, 5), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Internal full game state
        self.full_board = np.zeros(
            (5, 5), dtype=np.int8
        )  # Positive labels for player 1, negative labels for player -1
        self.particles_positions = {1: {}, -1: {}}  # Player's particles positions
        self.captured_particles = {1: [], -1: []}  # Captured opponent particles
        self.captured_positions = {
            1: {},
            -1: {},
        }  # Positions where opponent particles were captured
        self.entanglement_used = {1: False, -1: False}
        self.entangled_particles = {1: None, -1: None}  # Entangled particle labels
        self.entanglement_revealed = {
            1: None,
            -1: None,
        }  # Revealed entangled opponent particle

        # Randomly place particles for both players
        self._randomly_place_particles(1)
        self._randomly_place_particles(-1)

        return self._get_observation(), {}  # Return observation and info

    def _randomly_place_particles(self, player):
        positions = [(i, j) for i in range(5) for j in range(5)]
        self.np_random.shuffle(positions)
        for label in range(1, 6):
            row, col = positions.pop()
            self.full_board[row][col] = label * player
            self.particles_positions[player][label] = (row, col)

    def _get_observation(self):
        obs = np.zeros((5, 5), dtype=np.int8)
        player = self.current_player
        opponent = -player
        # Own particles
        for label, pos in self.particles_positions[player].items():
            row, col = pos
            obs[row][col] = label
        # Captured opponent particles
        for label in self.captured_particles[player]:
            row, col = self.captured_positions[player][label]
            obs[row][col] = -label
        # Revealed entangled opponent particle
        if self.entanglement_revealed[player]:
            label, pos = self.entanglement_revealed[player]
            row, col = pos
            obs[row][col] = -label
        return obs

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}
        # Decode action
        scan_option, entangle_option, guess_position, guess_label = self._decode_action(
            action
        )
        info = {}

        # Process entanglement
        if entangle_option != 0 and not self.entanglement_used[self.current_player]:
            entangle_pair = self._get_entangle_pair(entangle_option - 1)
            self.entanglement_used[self.current_player] = True
            self.entangled_particles[self.current_player] = entangle_pair
            info["entangle"] = entangle_pair
        else:
            entangle_pair = None

        # Process scan
        if scan_option != 0:
            scan_pos = self._get_scan_position(scan_option - 1)
            scan_result = self._perform_scan(scan_pos)
            info["scan_result"] = scan_result

        # Process guess
        row = guess_position // 5
        col = guess_position % 5
        guess_label += 1  # Labels are from 1 to 5

        opponent = -self.current_player
        correct_guess = False
        if self.full_board[row][col] == opponent * guess_label:
            # Correct guess, capture the particle
            correct_guess = True
            self.full_board[row][col] = 0
            del self.particles_positions[opponent][guess_label]
            self.captured_particles[self.current_player].append(guess_label)
            self.captured_positions[self.current_player][guess_label] = (row, col)
            # Check for entanglement
            if (
                self.entangled_particles[opponent]
                and guess_label in self.entangled_particles[opponent]
            ):
                other_label = (
                    self.entangled_particles[opponent][0]
                    if self.entangled_particles[opponent][1] == guess_label
                    else self.entangled_particles[opponent][1]
                )
                other_pos = self.particles_positions[opponent][other_label]
                self.entanglement_revealed[self.current_player] = (
                    other_label,
                    other_pos,
                )
                info["entanglement_reveal"] = (other_label, other_pos)
        # Check for win condition
        if len(self.particles_positions[opponent]) == 0:
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, info

        # Switch players
        self.current_player = opponent

        return (
            self._get_observation(),
            -10,
            False,
            False,
            info,
        )  # Negative reward for a valid move

    def render(self):
        obs = self._get_observation()
        board_str = "Current Player: {}\n".format(
            "1" if self.current_player == 1 else "2"
        )
        board_str += "  A B C D E\n"
        for i in range(5):
            row_str = "{} ".format(i + 1)
            for j in range(5):
                val = obs[i][j]
                if val > 0:
                    row_str += "{} ".format(val)
                elif val < 0:
                    row_str += "{} ".format(val)
                else:
                    row_str += ". "
            board_str += row_str + "\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        num_scan_options = 10  # 0 to 9
        num_entangle_options = 11  # 0 to 10
        num_guess_positions = 25  # 0 to 24
        num_labels = 5  # 0 to 4

        entanglement_already_used = self.entanglement_used[self.current_player]
        for scan_option in range(num_scan_options):
            for entangle_option in range(num_entangle_options):
                if entangle_option != 0 and entanglement_already_used:
                    continue  # Cannot entangle again
                for guess_position in range(num_guess_positions):
                    for guess_label in range(num_labels):
                        action = (
                            (scan_option * num_entangle_options + entangle_option)
                            * num_guess_positions
                            + guess_position
                        ) * num_labels + guess_label
                        valid_actions.append(action)
        return valid_actions

    def _decode_action(self, action):
        # Total actions: 10 * 11 * 25 * 5 = 13,750
        scan_option = action // (11 * 25 * 5)
        remainder = action % (11 * 25 * 5)
        entangle_option = remainder // (25 * 5)
        remainder = remainder % (25 * 5)
        guess_position = remainder // 5
        guess_label = remainder % 5
        return scan_option, entangle_option, guess_position, guess_label

    def _get_scan_position(self, scan_option_index):
        positions = [(i, j) for i in range(3) for j in range(3)]
        return positions[scan_option_index]

    def _perform_scan(self, scan_pos):
        start_row, start_col = scan_pos
        opponent = -self.current_player
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if 0 <= i < 5 and 0 <= j < 5:
                    if self.full_board[i][j] * opponent > 0:
                        return True  # Opponent's particle detected
        return False  # No opponent particles detected

    def _get_entangle_pair(self, entangle_option_index):
        labels = [1, 2, 3, 4, 5]
        pairs = list(combinations(labels, 2))
        return pairs[entangle_option_index]
