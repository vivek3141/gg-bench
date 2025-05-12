import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions are indices [0-8], representing numbers 1-9
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_pool = set(range(1, 10))  # Numbers 1-9
        self.player_hands = {1: set(), -1: set()}
        self.current_player = 1  # 1 or -1
        self.state = np.zeros(9, dtype=np.int8)
        self.done = False
        return self.state.copy(), {}  # Return observation and info

    def step(self, action):
        number = action + 1  # Convert action to number (1-9)
        if self.done:
            return self.state.copy(), 0, True, False, {}
        if number not in self.shared_pool:
            # Invalid move
            self.done = True
            return self.state.copy(), -10, True, False, {}

        # Valid move
        self.shared_pool.remove(number)
        self.state[action] = self.current_player
        self.player_hands[self.current_player].add(number)

        # Check for win condition
        if self.has_consecutive_sequence(self.player_hands[self.current_player]):
            self.done = True
            return self.state.copy(), 1, True, False, {}

        # Check if shared pool is empty
        if not self.shared_pool:
            sum_current = sum(self.player_hands[self.current_player])
            sum_opponent = sum(self.player_hands[-self.current_player])
            if sum_current > sum_opponent:
                # Current player wins
                self.done = True
                return self.state.copy(), 1, True, False, {}
            elif sum_current < sum_opponent:
                # Current player loses
                self.done = True
                return self.state.copy(), -1, True, False, {}
            else:
                # Tie-breaker: last player to have taken a turn loses
                self.done = True
                # Current player is the last to have taken a turn
                return self.state.copy(), -1, True, False, {}

        # Switch current player
        self.current_player *= -1
        return (
            self.state.copy(),
            0,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def has_consecutive_sequence(self, numbers_set):
        if len(numbers_set) < 3:
            return False
        numbers = sorted(numbers_set)
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                for k in range(j + 1, len(numbers)):
                    nums = [numbers[i], numbers[j], numbers[k]]
                    nums.sort()
                    if nums[1] - nums[0] == 1 and nums[2] - nums[1] == 1:
                        return True
        return False

    def render(self):
        shared_pool_str = "Shared Pool: " + str(sorted(self.shared_pool))
        player1_hand = [num for num in range(1, 10) if self.state[num - 1] == 1]
        player2_hand = [num for num in range(1, 10) if self.state[num - 1] == -1]
        player1_hand_str = "Player 1's Hand: " + str(sorted(player1_hand))
        player2_hand_str = "Player 2's Hand: " + str(sorted(player2_hand))
        current_player_str = "Current Player: " + (
            "Player 1" if self.current_player == 1 else "Player 2"
        )
        state_str = "\n".join(
            [shared_pool_str, player1_hand_str, player2_hand_str, current_player_str]
        )
        return state_str

    def valid_moves(self):
        return [
            number - 1 for number in self.shared_pool
        ]  # Return indices corresponding to available numbers
