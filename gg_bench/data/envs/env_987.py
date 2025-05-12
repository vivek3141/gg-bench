import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: actions from 0 to 99 correspond to bids from 1 to 100
        self.action_space = spaces.Discrete(100)

        # Define observation space
        # Observation vector includes:
        # - Current player's points (1)
        # - Opponent's points (1)
        # - Current item index (1)
        # - Items remaining to be auctioned (5)
        # - Items won by current player (5)
        # - Items won by opponent (5)
        # Total length: 1 + 1 + 1 + 5 + 5 + 5 = 18
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(18,),
            dtype=np.float32,
        )

        # Initialize variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Starting points for both players
        self.player_points = [100, 100]

        # Randomly assign hidden values to items (integers between 10 and 30)
        self.item_values = self.np_random.integers(10, 31, size=5)
        self.items = ["A", "B", "C", "D", "E"]
        self.current_item_index = 0

        # Track items won by players (lists of item indices)
        self.player_items = [[], []]

        # Set current player (0 or 1)
        self.current_player = 0

        # Game over flag
        self.done = False

        # Invalid move flag
        self.invalid_move = False

        # Prepare initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        bid_amount = action + 1  # Map action to bid amount (1 to 100)

        # Validate bid
        if bid_amount < 1 or bid_amount > self.player_points[self.current_player]:
            # Invalid bid: bid less than 1 or more than available points
            self.done = True
            reward = -10
            self.invalid_move = True
            return self._get_observation(), reward, True, False, {}

        # Generate opponent's bid (random policy)
        opponent = 1 - self.current_player
        opponent_bid = self.np_random.integers(1, self.player_points[opponent] + 1)

        # Compare bids
        if bid_amount > opponent_bid:
            # Current player wins the item
            winner = self.current_player
            winning_bid = bid_amount
        elif bid_amount < opponent_bid:
            # Opponent wins the item
            winner = opponent
            winning_bid = opponent_bid
        else:
            # Tie: randomly assign the item to one of the players
            winner = self.np_random.choice([self.current_player, opponent])
            winning_bid = bid_amount

        # Deduct bid amounts from players
        self.player_points[self.current_player] -= bid_amount
        self.player_points[opponent] -= opponent_bid

        # Assign item to winner
        self.player_items[winner].append(self.current_item_index)

        # Move to next item
        self.current_item_index += 1

        # Check if game is over
        if self.current_item_index >= len(self.items):
            self.done = True
            # Calculate total item values for both players
            player_total = sum(
                self.item_values[i] for i in self.player_items[self.current_player]
            )
            opponent_total = sum(
                self.item_values[i] for i in self.player_items[opponent]
            )
            if player_total > opponent_total:
                reward = 1  # Current player wins
            else:
                reward = 0  # Current player loses
            return self._get_observation(), reward, True, False, {}
        else:
            # Swap current player
            self.current_player = opponent
            reward = 0
            return self._get_observation(), reward, False, False, {}

    def render(self):
        # Print the game state
        print(f"Current Player: Player {self.current_player + 1}")
        print(f"Player 1 Points: {self.player_points[0]}")
        print(f"Player 2 Points: {self.player_points[1]}")
        print(
            f"Items Remaining: {[self.items[i] for i in range(self.current_item_index, len(self.items))]}"
        )
        print(f"Player 1 Items: {[self.items[i] for i in self.player_items[0]]}")
        print(f"Player 2 Items: {[self.items[i] for i in self.player_items[1]]}")
        print(
            f"Current Item: {self.items[self.current_item_index] if self.current_item_index < len(self.items) else 'None'}"
        )
        print("----------------------------------------------------")

    def valid_moves(self):
        # Return list of valid bid amounts (actions) for current player
        max_bid = self.player_points[self.current_player]
        return list(range(max_bid))

    def _get_observation(self):
        # Build the observation vector
        observation = np.zeros(18, dtype=np.float32)
        # Current player's points
        observation[0] = self.player_points[self.current_player]
        # Opponent's points
        observation[1] = self.player_points[1 - self.current_player]
        # Current item index (2)
        observation[2] = self.current_item_index
        # Items remaining to be auctioned (indices from current_item_index to end)
        for i in range(self.current_item_index, len(self.items)):
            observation[3 + i] = 1  # Indicate that item is remaining
        # Items won by current player
        for idx in self.player_items[self.current_player]:
            observation[8 + idx] = 1  # Indicate item won by current player
        # Items won by opponent
        for idx in self.player_items[1 - self.current_player]:
            observation[13 + idx] = 1  # Indicate item won by opponent
        return observation
