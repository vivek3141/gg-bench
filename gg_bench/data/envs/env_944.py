import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space
        # Actions:
        # 0 - Accept generated number only
        # 1 - Accept rejected number only
        # 2 - Accept both numbers
        # 3 - Reject both numbers
        self.action_space = spaces.Discrete(4)

        # Define observation space
        # Observation consists of:
        # [player_score, opponent_score, generated_number, rejected_number, player_consecutive_rejections, opponent_consecutive_rejections]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1, 0, 0, 0], dtype=np.int32),
            high=np.array([52, 52, 10, 10, 2, 2], dtype=np.int32),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.scores = [0, 0]
        self.consecutive_rejections = [0, 0]
        self.rejected_number = 0  # No rejected number initially
        self.generated_number = np.random.randint(1, 11)
        self.done = False

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

        current_player = self.current_player
        opponent = 1 - self.current_player

        accepted_numbers = []
        rejected_number_for_next_player = 0

        # Process action
        if action == 0:
            # Accept generated number only
            accepted_numbers.append(self.generated_number)
            # Rejected number discarded
            rejected_number_for_next_player = 0
            self.consecutive_rejections[current_player] = 0
        elif action == 1:
            # Accept rejected number only
            accepted_numbers.append(self.rejected_number)
            # Generated number is rejected and passed to opponent
            rejected_number_for_next_player = self.generated_number
            self.consecutive_rejections[current_player] = 0
        elif action == 2:
            # Accept both numbers
            accepted_numbers.append(self.generated_number)
            accepted_numbers.append(self.rejected_number)
            rejected_number_for_next_player = 0
            self.consecutive_rejections[current_player] = 0
        elif action == 3:
            # Reject both numbers
            # Generated number is passed to opponent
            rejected_number_for_next_player = self.generated_number
            self.consecutive_rejections[current_player] += 1
            if self.consecutive_rejections[current_player] > 2:
                self.done = True
                return self._get_observation(), -10, True, False, {}
        else:
            # Should not reach here
            pass

        # Update player's score
        if accepted_numbers:
            score_increment = sum(accepted_numbers)
            self.scores[current_player] += score_increment
            if self.scores[current_player] > 50:
                self.scores[current_player] = 25
            self.consecutive_rejections[current_player] = 0

        # Check for winning condition
        if self.scores[current_player] == 50:
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Update rejected number for next player
        self.rejected_number = rejected_number_for_next_player

        # Generate new number for next player
        self.generated_number = np.random.randint(1, 11)

        # Switch current player
        self.current_player = opponent

        return self._get_observation(), -10, False, False, {}

    def render(self):
        s = f"Player {self.current_player + 1}'s Turn\n"
        s += "-" * 20 + "\n"
        s += f"Your current score: {self.scores[self.current_player]}\n"
        s += f"Opponent's score: {self.scores[1 - self.current_player]}\n"
        if self.rejected_number != 0:
            s += f"Rejected number from opponent: {self.rejected_number}\n"
        else:
            s += "No rejected number from opponent.\n"
        s += f"Generated number: {self.generated_number}\n"
        s += f"Your consecutive rejections: {self.consecutive_rejections[self.current_player]}\n"
        return s

    def valid_moves(self):
        actions = []
        current_player = self.current_player
        has_generated_number = True  # Generated number is always available
        has_rejected_number = self.rejected_number != 0

        if self.consecutive_rejections[current_player] >= 2:
            # Must accept at least one number
            if has_generated_number and has_rejected_number:
                actions.extend([0, 1, 2])
            elif has_generated_number:
                actions.append(0)
            elif has_rejected_number:
                actions.append(1)
        else:
            # Can choose any valid action
            if has_generated_number and has_rejected_number:
                actions.extend([0, 1, 2, 3])
            elif has_generated_number:
                actions.extend([0, 3])
            elif has_rejected_number:
                actions.extend([1, 3])
            else:
                # Should not happen
                pass
        return actions

    def _get_observation(self):
        current_player = self.current_player
        opponent = 1 - self.current_player
        observation = np.array(
            [
                self.scores[current_player],
                self.scores[opponent],
                self.generated_number,
                self.rejected_number,
                self.consecutive_rejections[current_player],
                self.consecutive_rejections[opponent],
            ],
            dtype=np.int32,
        )
        return observation
