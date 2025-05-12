import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Maximum tokens a player can have
        self.MAX_TOKENS = 20  # Maximum tokens in reserve

        # Define action space: actions from 0 to MAX_TOKENS
        # Action 0: Fortify (in "choose_action" phase) or commit 0 tokens (in "defend" phase)
        # Actions 1 to MAX_TOKENS: Attack or defend with N tokens
        self.action_space = spaces.Discrete(self.MAX_TOKENS + 1)

        # Observation space: [current_player_reserve, opponent_reserve, phase_indicator, current_player_id]
        # phase_indicator: 0 for "choose_action", 1 for "defend"
        # current_player_id: 1 or 2
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.MAX_TOKENS, self.MAX_TOKENS, 1, 2]),
            dtype=np.int32,
        )

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_reserves = {1: 10, 2: 10}  # Starting tokens
        self.current_player = 1  # Player 1 starts
        self.opponent_player = 2
        self.phase = "choose_action"  # "choose_action" or "defend"
        self.attacker_commit = 0
        self.done = False
        # Observation format: [current_player_reserve, opponent_reserve, phase_indicator, current_player_id]
        observation = np.array(
            [
                self.player_reserves[self.current_player],
                self.player_reserves[self.opponent_player],
                0,  # phase_indicator for "choose_action"
                self.current_player,
            ],
            dtype=np.int32,
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if game is already over
        if self.done:
            return self._get_observation(), 0, True, False, {}

        reward = 0
        info = {}

        current_reserve = self.player_reserves[self.current_player]
        opponent_reserve = self.player_reserves[self.opponent_player]

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid action
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, info

        if self.phase == "choose_action":
            if action == 0:
                # Fortify
                if current_reserve >= 15:
                    # Cannot fortify if reserve is 15 or more
                    self.done = True
                    reward = -10
                    return self._get_observation(), reward, True, False, info
                else:
                    self.player_reserves[self.current_player] += 1
                    # End of turn, switch player
                    self._switch_player()
                    return self._get_observation(), reward, False, False, info
            else:
                # Attack with action tokens
                if action > current_reserve:
                    # Cannot commit more tokens than available
                    self.done = True
                    reward = -10
                    return self._get_observation(), reward, True, False, info
                else:
                    self.attacker_commit = action
                    # Switch to defend phase
                    self.phase = "defend"
                    return self._get_observation(), reward, False, False, info
        elif self.phase == "defend":
            defender_reserve = self.player_reserves[self.current_player]
            if action > defender_reserve:
                # Cannot commit more tokens than available
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, info
            else:
                defender_commit = action
                # Both players commit tokens
                attacker = self.opponent_player
                defender = self.current_player
                attacker_reserve = self.player_reserves[attacker]

                # Reveal committed tokens
                if self.attacker_commit > defender_commit:
                    # Attacker wins
                    difference = self.attacker_commit - defender_commit
                    tokens_transferred = min(difference, self.player_reserves[defender])
                    self.player_reserves[attacker] += tokens_transferred
                    self.player_reserves[defender] -= tokens_transferred
                else:
                    # Defender wins (tie or defender_commit > attacker_commit)
                    difference = defender_commit - self.attacker_commit
                    tokens_transferred = min(difference, self.player_reserves[attacker])
                    self.player_reserves[defender] += tokens_transferred
                    self.player_reserves[attacker] -= tokens_transferred

                # Spent tokens are removed from reserves (spent tokens are removed from the game)
                self.player_reserves[attacker] -= self.attacker_commit
                self.player_reserves[defender] -= defender_commit

                # Ensure reserves are not negative
                self.player_reserves[attacker] = max(0, self.player_reserves[attacker])
                self.player_reserves[defender] = max(0, self.player_reserves[defender])

                # Check for game end condition
                if self.player_reserves[defender] == 0:
                    # Defender has 0 tokens, attacker wins
                    self.done = True
                    if self.current_player == self.current_player:
                        # Current player lost
                        reward = -1  # Opponent wins
                    else:
                        reward = 1  # Current player wins
                    return self._get_observation(), reward, True, False, info
                elif self.player_reserves[attacker] == 0:
                    # Attacker has 0 tokens, defender wins
                    self.done = True
                    if self.current_player == self.current_player:
                        reward = 1  # Current player wins
                    else:
                        reward = -1  # Current player lost
                    return self._get_observation(), reward, True, False, info
                else:
                    # Continue game
                    # Reset attacker_commit
                    self.attacker_commit = 0
                    # Switch to next player's turn
                    self.phase = "choose_action"
                    self._switch_player()
                    return self._get_observation(), reward, False, False, info
        else:
            # Invalid phase
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, info

    def render(self):
        output = "---- Token Tactics ----\n"
        output += f"Player 1 Tokens: {self.player_reserves[1]}\n"
        output += f"Player 2 Tokens: {self.player_reserves[2]}\n"
        output += f"Current Player: Player {self.current_player}\n"
        output += f"Phase: {self.phase}\n"
        if self.phase == "defend":
            output += "Attacker committed tokens: (hidden)\n"
        return output

    def valid_moves(self):
        current_reserve = self.player_reserves[self.current_player]
        if self.phase == "choose_action":
            moves = []
            if current_reserve < 15:
                moves.append(0)  # Fortify
            if current_reserve >= 1:
                # Can attack with 1 to current_reserve tokens
                moves.extend(range(1, current_reserve + 1))
            return moves
        elif self.phase == "defend":
            # Can defend with 0 to current_reserve tokens
            return list(range(0, current_reserve + 1))
        else:
            return []

    def _switch_player(self):
        if self.current_player == 1:
            self.current_player = 2
            self.opponent_player = 1
        else:
            self.current_player = 1
            self.opponent_player = 2

    def _get_observation(self):
        phase_indicator = 0 if self.phase == "choose_action" else 1
        observation = np.array(
            [
                self.player_reserves[self.current_player],
                self.player_reserves[self.opponent_player],
                phase_indicator,
                self.current_player,
            ],
            dtype=np.int32,
        )
        return observation
