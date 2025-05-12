import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the maximum length of the number list
        self.max_number = 9
        self.max_length = self.max_number  # Maximum initial length of the number list

        # Define action and observation space
        # Two possible actions: 0 (left), 1 (right)
        self.action_space = spaces.Discrete(2)

        # Observation space:
        # - Current number list: array of length max_length
        # - Player 1 score: scalar
        # - Player 2 score: scalar
        # - Current player: 1 or -1

        # The observation is a 1D array with length max_length + 3
        self.observation_space = spaces.Box(
            low=0,
            high=45,  # Maximum possible total score
            shape=(self.max_length + 3,),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the number list with shuffled numbers from 1 to max_number
        self.number_list = np.arange(1, self.max_number + 1)
        self.np_random.shuffle(self.number_list)
        self.number_list = list(self.number_list)

        # Initialize players' collections
        self.player1_collection = []
        self.player2_collection = []

        # Set current player: 1 for Player 1, -1 for Player 2
        self.current_player = 1

        # Initialize done flag
        self.done = False

        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Perform action
        if action == 0:
            # Choose leftmost number
            chosen_number = self.number_list.pop(0)
        elif action == 1:
            # Choose rightmost number
            chosen_number = self.number_list.pop(-1)

        # Update current player's collection
        if self.current_player == 1:
            self.player1_collection.append(chosen_number)
        else:
            self.player2_collection.append(chosen_number)

        # Check if the game is over
        if not self.number_list:
            self.done = True
            # Calculate scores
            player1_score = sum(self.player1_collection)
            player2_score = sum(self.player2_collection)
            if player1_score > player2_score:
                winner = 1
            elif player2_score > player1_score:
                winner = -1
            else:
                # Tie-breaker: Last player to have taken a number wins
                winner = self.current_player

            if winner == self.current_player:
                # Current player wins
                reward = 1
            else:
                # Current player loses
                reward = -1

            return self._get_observation(), reward, True, False, {}
        else:
            # Switch current player
            self.current_player *= -1
            return self._get_observation(), 0, False, False, {}

    def render(self):
        # Return a string representation of the game state
        obs = self._get_observation()
        number_list = obs[: self.max_length]
        player1_score = obs[self.max_length]
        player2_score = obs[self.max_length + 1]
        current_player_indicator = obs[self.max_length + 2]
        current_player = "Player 1" if current_player_indicator == 1 else "Player 2"

        number_list_str = [num for num in number_list if num != 0]
        return (
            f"Current Number List: {number_list_str}\n"
            f"Player 1's Collection: {self.player1_collection} | Total Score: {sum(self.player1_collection)}\n"
            f"Player 2's Collection: {self.player2_collection} | Total Score: {sum(self.player2_collection)}\n"
            f"Current Turn: {current_player}\n"
        )

    def valid_moves(self):
        if not self.number_list:
            return []
        else:
            return [0, 1]  # 0 for left, 1 for right

    def _get_observation(self):
        # Prepare observation array
        obs = np.zeros(self.max_length + 3, dtype=np.int32)
        # Current number list (padded with zeros)
        for i in range(len(self.number_list)):
            obs[i] = self.number_list[i]
        # Player scores
        obs[self.max_length] = sum(self.player1_collection)
        obs[self.max_length + 1] = sum(self.player2_collection)
        # Current player indicator
        obs[self.max_length + 2] = self.current_player
        return obs
