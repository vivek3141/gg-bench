import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Integers from 0 to 19 representing numbers 1 to 20
        self.action_space = spaces.Discrete(20)

        # Observation space: Array of 20 integers, each can be -1, 0, or 1
        # -1: Claimed by Player B, 0: Unclaimed, 1: Claimed by Player A
        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.int8)

        # Precompute all possible winning sequences (arithmetic sequences of length 4)
        self.winning_sequences = self.generate_winning_sequences()

        self.reset()

    def generate_winning_sequences(self):
        sequences = set()
        for start in range(1, 21):
            for diff in range(-19, 20):
                if diff == 0:
                    continue
                seq = []
                for i in range(4):
                    num = start + i * diff
                    if not (1 <= num <= 20):
                        break
                    seq.append(num)
                if len(seq) == 4:
                    sequences.add(tuple(sorted(seq)))
        return sequences

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the pool: 0 for unclaimed, 1 for Player A, -1 for Player B
        self.pool = np.zeros(20, dtype=np.int8)

        # Initialize player collections
        self.player_collections = {1: set(), -1: set()}

        # Current player: 1 for Player A, -1 for Player B
        self.current_player = 1

        # Game over flag
        self.done = False

        # Winner: 1 for Player A, -1 for Player B, None if no winner yet
        self.winner = None

        return self.pool.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.pool.copy(), 0, True, False, {}

        # Check if action is valid
        if not (0 <= action < 20) or self.pool[action] != 0:
            reward = -10
            self.done = True
            self.winner = None  # No winner due to invalid move
            return self.pool.copy(), reward, True, False, {}

        # Update the state
        self.pool[action] = self.current_player
        self.player_collections[self.current_player].add(
            action + 1
        )  # Numbers are from 1 to 20

        # Check for a winning sequence
        if self.check_win(self.player_collections[self.current_player]):
            reward = 1
            self.done = True
            self.winner = self.current_player
            return self.pool.copy(), reward, True, False, {}

        # Check if all numbers are claimed
        if np.all(self.pool != 0):
            # Game over, no winner
            reward = 0
            self.done = True
            self.winner = None
            return self.pool.copy(), reward, True, False, {}

        # Switch players
        self.current_player *= -1

        return self.pool.copy(), 0, False, False, {}

    def check_win(self, player_numbers):
        # player_numbers is a set of numbers (integers from 1 to 20)
        for sequence in self.winning_sequences:
            if set(sequence).issubset(player_numbers):
                return True
        return False

    def render(self):
        available_numbers = [i + 1 for i in range(20) if self.pool[i] == 0]
        player_a_collection = sorted([i + 1 for i in range(20) if self.pool[i] == 1])
        player_b_collection = sorted([i + 1 for i in range(20) if self.pool[i] == -1])

        render_str = "--- Number Sequence Duel ---\n"
        render_str += f"Available Numbers: {' '.join(map(str, available_numbers))}\n"
        render_str += (
            f"Player A's Collection: {' '.join(map(str, player_a_collection))}\n"
        )
        render_str += (
            f"Player B's Collection: {' '.join(map(str, player_b_collection))}\n"
        )

        if not self.done:
            current_player_name = "Player A" if self.current_player == 1 else "Player B"
            render_str += f"{current_player_name}, it's your turn.\n"
        else:
            if self.winner is not None:
                winner_name = "Player A" if self.winner == 1 else "Player B"
                render_str += f"Congratulations! {winner_name} wins the game!\n"
            else:
                render_str += "Game Over. No winner.\n"

        return render_str

    def valid_moves(self):
        return [i for i in range(20) if self.pool[i] == 0]
