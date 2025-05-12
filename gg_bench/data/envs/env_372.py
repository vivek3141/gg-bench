import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to digits 1-9 (represented as 0-8 in action space)
        self.action_space = spaces.Discrete(9)

        # Observation space:
        # Index 0: Current player's life points
        # Index 1: Opponent's life points
        # Index 2-10: Current player's digits used (1 if used, 0 if not)
        # Index 11-19: Opponent's digits used
        # Index 20: Current phase (0 for attack, 1 for defense)
        # Index 21: Last action of opponent (attack value when defending)
        self.observation_space = spaces.Box(low=0, high=10, shape=(22,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_life_points = [10, 10]  # Life points for both players
        # Digits used by both players: 0 if unused, 1 if used
        self.player_digits_used = [
            np.zeros(9, dtype=np.int8),
            np.zeros(9, dtype=np.int8),
        ]
        self.current_player = 0  # Player 0 starts as attacker
        self.current_phase = "attack"  # 'attack' or 'defense'
        self.last_action_opponent = None  # Stores attack value during defense phase
        self.done = False
        self.info = {}
        self.last_attacker = None  # Tracks who was the attacker in the last turn

        observation = self._get_observation()
        return observation, self.info

    def _get_observation(self):
        observation = np.zeros(22, dtype=np.int8)
        observation[0] = self.player_life_points[self.current_player]
        opponent = 1 - self.current_player
        observation[1] = self.player_life_points[opponent]
        observation[2:11] = self.player_digits_used[self.current_player]
        observation[11:20] = self.player_digits_used[opponent]
        observation[20] = 0 if self.current_phase == "attack" else 1
        observation[21] = (
            self.last_action_opponent if self.last_action_opponent is not None else 0
        )
        return observation

    def step(self, action):
        if self.done:
            raise Exception("Game is over. Please reset the environment.")

        # Map action (0-8) to digit (1-9)
        digit_chosen = action + 1

        # Check if the digit has not been used by the current player
        if self.player_digits_used[self.current_player][action] == 1:
            # Invalid move, digit already used
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, self.done, False, self.info

        # Mark the digit as used
        self.player_digits_used[self.current_player][action] = 1

        opponent = 1 - self.current_player  # Opponent index

        # Process based on current phase
        if self.current_phase == "attack":
            # Attack phase: current player is attacker
            self.attack_value = digit_chosen
            # Record last attacker
            self.last_attacker = self.current_player
            # Switch to defense phase
            self.current_phase = "defense"
            # Switch current player to defender
            self.current_player = opponent
            # Record attack value for defender's observation
            self.last_action_opponent = self.attack_value
            reward = 0
            self.done = False
        elif self.current_phase == "defense":
            # Defense phase: current player is defender
            self.defense_value = digit_chosen
            # Compute the outcome
            if self.attack_value > self.defense_value:
                damage = self.attack_value - self.defense_value
                # Subtract damage from defender's life points
                self.player_life_points[self.current_player] -= damage
            # After subtracting damage, check if defender's life points <= 0
            if self.player_life_points[self.current_player] <= 0:
                # Defender (current player) loses
                reward = -1  # Current player loses
                self.done = True
            else:
                # Check if all digits are used
                if np.all(self.player_digits_used[0] == 1) and np.all(
                    self.player_digits_used[1] == 1
                ):
                    # All digits are used
                    current_player_life = self.player_life_points[self.current_player]
                    opponent_life = self.player_life_points[opponent]
                    if current_player_life > opponent_life:
                        reward = 1  # Current player wins
                    elif current_player_life < opponent_life:
                        reward = -1  # Current player loses
                    else:
                        # Life points equal, attacker in last turn wins
                        if self.last_attacker == self.current_player:
                            reward = 1  # Current player wins
                        else:
                            reward = -1  # Current player loses
                    self.done = True
                else:
                    # Game continues
                    reward = 0
                    self.done = False
            # Reset last action
            self.last_action_opponent = None
            # Switch roles for next turn
            self.current_player = opponent
            # Set phase to attack
            self.current_phase = "attack"
        else:
            raise Exception("Invalid game phase.")

        # Prepare observation
        observation = self._get_observation()
        return observation, reward, self.done, False, self.info

    def render(self):
        board_str = ""
        board_str += f"--- {'Attacker' if self.current_phase == 'attack' else 'Defender'} Turn ---\n"
        board_str += f"Player {self.current_player + 1}'s Turn ({'Attack' if self.current_phase == 'attack' else 'Defense'})\n"
        board_str += f"Life Points: Player 1 - {self.player_life_points[0]}, Player 2 - {self.player_life_points[1]}\n"
        board_str += f"Available Digits for Player {self.current_player + 1}: "
        available_digits = [
            str(i + 1)
            for i in range(9)
            if self.player_digits_used[self.current_player][i] == 0
        ]
        board_str += ", ".join(available_digits) + "\n"
        if self.current_phase == "defense":
            board_str += f"Attack Value to defend against: {self.attack_value}\n"
        return board_str

    def valid_moves(self):
        return [
            i for i in range(9) if self.player_digits_used[self.current_player][i] == 0
        ]
