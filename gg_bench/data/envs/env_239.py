import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 15 possible actions (5 single sticks + 10 pairs)
        self.action_space = spaces.Discrete(15)

        # Observation space: 5 sticks availability + 2 scores (current player and opponent)
        # Sticks availability: 1 if available, 0 if not
        # Scores range from 0 to 10
        self.observation_space = spaces.Box(low=0, high=10, shape=(7,), dtype=np.int32)

        # Mapping from action index to sticks picked
        self.actions = [
            (1,),  # 0
            (2,),  # 1
            (3,),  # 2
            (4,),  # 3
            (5,),  # 4
            (1, 2),  # 5
            (1, 3),  # 6
            (1, 4),  # 7
            (1, 5),  # 8
            (2, 3),  # 9
            (2, 4),  # 10
            (2, 5),  # 11
            (3, 4),  # 12
            (3, 5),  # 13
            (4, 5),  # 14
        ]

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize sticks availability: 1 if available, 0 if not
        self.sticks_available = np.ones(5, dtype=np.int32)
        # Initialize player scores
        self.player_scores = [0, 0]
        # Current player: 0 or 1
        self.current_player = 0
        self.done = False
        # Return initial observation and info
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            # If the game is already over, return current state
            observation = self._get_observation()
            return observation, 0, True, False, {}

        # Get the sticks to pick based on the action index
        sticks_to_pick = self.actions[action]
        # Check if the action is valid
        valid = self._is_valid_action(sticks_to_pick)

        if not valid:
            # Invalid action
            self.done = True
            observation = self._get_observation()
            return observation, -10, True, False, {}

        # Perform the action
        self._pick_sticks(sticks_to_pick)
        # Check for winning condition
        if self.player_scores[self.current_player] == 10:
            # Current player wins
            self.done = True
            observation = self._get_observation()
            return observation, 1, True, False, {}
        else:
            # Check if no valid moves remain
            if not self._has_valid_moves():
                # Game over, determine winner
                self.done = True
                winner = self._determine_winner()
                observation = self._get_observation()
                if winner == self.current_player:
                    return observation, 1, True, False, {}
                else:
                    return observation, -1, True, False, {}
            else:
                # Switch to the other player
                self.current_player = 1 - self.current_player
                observation = self._get_observation()
                return observation, 0, False, False, {}

    def render(self):
        sticks_str = "Available sticks: "
        for i in range(5):
            if self.sticks_available[i]:
                sticks_str += f"{i+1} "
        sticks_str = sticks_str.strip()
        player_scores_str = f"Player {self.current_player + 1} Score: {self.player_scores[self.current_player]}\n"
        opponent = 1 - self.current_player
        opponent_scores_str = (
            f"Player {opponent + 1} Score: {self.player_scores[opponent]}"
        )
        return f"{sticks_str}\n{player_scores_str}{opponent_scores_str}"

    def valid_moves(self):
        valid_actions = []
        for idx, sticks in enumerate(self.actions):
            if self._is_valid_action(sticks):
                valid_actions.append(idx)
        return valid_actions

    def _get_observation(self):
        # Observation includes sticks availability and both players' scores
        sticks_availability = self.sticks_available.copy()
        current_score = self.player_scores[self.current_player]
        opponent_score = self.player_scores[1 - self.current_player]
        observation = np.concatenate(
            (sticks_availability, [current_score, opponent_score])
        )
        return observation

    def _is_valid_action(self, sticks):
        # Check if the sticks are available
        for stick in sticks:
            if self.sticks_available[stick - 1] == 0:
                return False
        # Check if picking the sticks would exceed score of 10
        total_value = sum(sticks)
        if self.player_scores[self.current_player] + total_value > 10:
            return False
        return True

    def _pick_sticks(self, sticks):
        # Remove sticks from availability and add to player's score
        for stick in sticks:
            self.sticks_available[stick - 1] = 0
            self.player_scores[self.current_player] += stick

    def _has_valid_moves(self):
        # Check if there are valid moves for the next player
        next_player = 1 - self.current_player
        for sticks in self.actions:
            # Check if sticks are available
            sticks_available = all(
                self.sticks_available[stick - 1] == 1 for stick in sticks
            )
            # Check if picking the sticks would not exceed score of 10
            total_value = sum(sticks)
            if sticks_available and (
                self.player_scores[next_player] + total_value <= 10
            ):
                return True
        return False

    def _determine_winner(self):
        # Determine the winner based on who is closest to 10 without exceeding it
        player_score = self.player_scores[self.current_player]
        opponent_score = self.player_scores[1 - self.current_player]
        if player_score > 10:
            player_score = 0  # Exceeded 10, set to 0
        if opponent_score > 10:
            opponent_score = 0  # Exceeded 10, set to 0
        if player_score > opponent_score:
            return self.current_player
        else:
            return 1 - self.current_player
