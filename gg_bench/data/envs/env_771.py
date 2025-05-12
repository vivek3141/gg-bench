import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(11)  # Bids from 0 to 10 inclusive
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(5,), dtype=np.int32
        )  # [Current Bidding Points, Current Resource Points, Opponent Bidding Points, Opponent Resource Points, Current Player]

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bidding_points = [10, 10]  # Bidding Points for Player 1 and Player 2
        self.resource_points = [0, 0]  # Resource Points for Player 1 and Player 2
        self.rounds_won = [0, 0]  # Rounds won by each player
        self.bidding_points_before_last_bid = [10, 10]  # Bidding Points before last bid
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.bid_history = [None, None]  # Bids for the current round
        self.last_winner = None  # Last player who won a round
        self.done = False  # Game over flag
        self.info = {}

        # Prepare the initial observation
        observation = self._get_observation()
        return observation, self.info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, self.info

        # Validate the action (bid)
        if (
            action < 0
            or action > 10
            or action > self.bidding_points[self.current_player]
        ):
            self.done = True
            reward = -10
            self.info = {"error": "Invalid bid"}
            return self._get_observation(), reward, True, False, self.info

        # Store the bidding points before the bid
        self.bidding_points_before_last_bid[self.current_player] = self.bidding_points[
            self.current_player
        ]

        # Deduct the bid from the player's bidding points
        self.bidding_points[self.current_player] -= action
        self.bid_history[self.current_player] = action

        # Initialize reward
        reward = -10

        # Check if both bids have been collected
        if self.bid_history[0] is not None and self.bid_history[1] is not None:
            # Resolve the round
            bid1 = self.bid_history[0]
            bid2 = self.bid_history[1]
            self.bid_history = [None, None]  # Reset bid history for next round

            if bid1 > bid2:
                winner = 0
                self.resource_points[0] += bid1 - bid2
                self.rounds_won[0] += 1
                self.last_winner = 0
            elif bid2 > bid1:
                winner = 1
                self.resource_points[1] += bid2 - bid1
                self.rounds_won[1] += 1
                self.last_winner = 1
            else:
                winner = None  # Tie, no resource points awarded

            # Check for immediate victory
            if (winner is not None) and (self.resource_points[winner] >= 7):
                self.done = True
                if winner == self.current_player:
                    reward = 1
                else:
                    reward = -10
                return self._get_observation(), reward, True, False, self.info

            # Check if both players have exhausted their bidding points
            if self.bidding_points[0] == 0 and self.bidding_points[1] == 0:
                # Determine the winner
                if self.resource_points[0] > self.resource_points[1]:
                    overall_winner = 0
                elif self.resource_points[1] > self.resource_points[0]:
                    overall_winner = 1
                else:
                    # Resource Points are tied, check rounds won
                    if self.rounds_won[0] > self.rounds_won[1]:
                        overall_winner = 0
                    elif self.rounds_won[1] > self.rounds_won[0]:
                        overall_winner = 1
                    else:
                        # Rounds won are tied, check bidding points before last bid
                        bp0 = self.bidding_points_before_last_bid[0]
                        bp1 = self.bidding_points_before_last_bid[1]
                        if bp0 > bp1:
                            overall_winner = 0
                        elif bp1 > bp0:
                            overall_winner = 1
                        else:
                            # Last resort, the player who last won a round wins
                            overall_winner = self.last_winner

                self.done = True
                if overall_winner == self.current_player:
                    reward = 1
                else:
                    reward = -10
                return self._get_observation(), reward, True, False, self.info

        # Switch to the other player
        self.current_player = 1 - self.current_player

        return self._get_observation(), reward, self.done, False, self.info

    def render(self):
        player_number = self.current_player + 1
        opponent_number = 2 if player_number == 1 else 1
        output = f"Player {player_number}'s turn\n"
        output += f"Player {player_number} - Bidding Points: {self.bidding_points[self.current_player]} | Resource Points: {self.resource_points[self.current_player]}\n"
        output += f"Player {opponent_number} - Bidding Points: {self.bidding_points[1 - self.current_player]} | Resource Points: {self.resource_points[1 - self.current_player]}\n"
        return output

    def valid_moves(self):
        # Return a list of valid bids for the current player
        max_bid = self.bidding_points[self.current_player]
        return list(range(0, max_bid + 1))

    def _get_observation(self):
        # Prepare the observation without revealing the opponent's bid
        observation = np.array(
            [
                self.bidding_points[self.current_player],
                self.resource_points[self.current_player],
                self.bidding_points[1 - self.current_player],
                self.resource_points[1 - self.current_player],
                self.current_player + 1,  # Player number: 1 or 2
            ],
            dtype=np.int32,
        )
        return observation
