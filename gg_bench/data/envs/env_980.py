import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 (pass/not defend) or 1-10 (tokens)
        self.action_space = spaces.Discrete(11)

        # Observation space:
        # [Player's Base HP, Opponent's Base HP,
        #  Player's tokens availability (10 elements),
        #  Opponent's tokens availability (10 elements),
        #  Phase indicator (0 for attack, 1 for defense),
        #  Current player (1 or -1),
        #  Attack token (0 if not set, else token value normalized)]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(25,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.base_hp = {1: 15, -1: 15}  # Player 1 and Player -1 Base HP
        self.tokens = {
            1: {i: True for i in range(1, 11)},  # Player 1's tokens availability
            -1: {i: True for i in range(1, 11)},  # Player -1's tokens availability
        }
        self.current_player = 1  # Player 1 starts
        self.phase = "attack"  # Start with attack phase
        self.attack_token = 0  # No attack token yet
        self.done = False
        self.info = {}
        observation = self._get_observation()
        return observation, self.info  # Return observation and info

    def step(self, action):
        # Check if game is already over
        if self.done:
            return self._get_observation(), 0, True, False, self.info

        reward = 0

        # Attack Phase
        if self.phase == "attack":
            # Validate action
            if action < 1 or action > 10:
                # Invalid action during attack phase
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, self.info

            # Check if token is available
            if not self.tokens[self.current_player].get(action, False):
                # Token not available
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, self.info

            # Valid attack
            self.attack_token = action
            # Remove token from current player's available tokens
            self.tokens[self.current_player][action] = False

            # Switch to defense phase, switch current player
            self.phase = "defense"
            self.current_player *= -1  # Switch player

            observation = self._get_observation()
            return observation, reward, False, False, self.info

        # Defense Phase
        elif self.phase == "defense":
            # Validate action
            if action < 0 or action > 10:
                # Invalid action during defense phase
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, self.info

            if action == 0:
                # Defender chooses not to defend
                damage = self.attack_token
            else:
                # Check if token is available
                if not self.tokens[self.current_player].get(action, False):
                    # Token not available
                    self.done = True
                    reward = -10
                    return self._get_observation(), reward, True, False, self.info

                # Valid defense
                defense_token = action
                # Remove defense token
                self.tokens[self.current_player][defense_token] = False

                # Resolve attack
                if defense_token >= self.attack_token:
                    # Attack fully deflected
                    damage = 0
                else:
                    # Partial defense
                    damage = self.attack_token - defense_token
                # Add defense token to info for rendering
                self.info["defense_token"] = defense_token
            # Apply damage
            self.base_hp[self.current_player] -= damage

            # Check for game end
            if self.base_hp[self.current_player] <= 0:
                self.done = True
                reward = 1  # Current player loses, so opponent wins
                # Switch current player back to attacker for correct reward assignment
                self.current_player *= -1
                return self._get_observation(), reward, True, False, self.info

            # Reset attack token
            self.attack_token = 0

            # Switch to attack phase, switch current player
            self.phase = "attack"
            self.current_player *= -1  # Switch player

            observation = self._get_observation()
            return observation, reward, False, False, self.info

        else:
            # Invalid phase
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, self.info

    def _get_observation(self):
        # Normalize Base HPs
        player_hp = self.base_hp[self.current_player] / 15.0
        opponent_hp = self.base_hp[-self.current_player] / 15.0

        # Player's tokens availability
        player_tokens = [
            1.0 if self.tokens[self.current_player][i + 1] else 0.0 for i in range(10)
        ]

        # Opponent's tokens availability
        opponent_tokens = [
            1.0 if self.tokens[-self.current_player][i + 1] else 0.0 for i in range(10)
        ]

        # Phase indicator: 0 for attack, 1 for defense
        phase_indicator = 0.0 if self.phase == "attack" else 1.0

        # Current player indicator: 1.0 for player 1, 0.0 for player -1
        current_player_indicator = 1.0 if self.current_player == 1 else 0.0

        # Attack token normalized (0.0 if not set)
        attack_token_norm = self.attack_token / 10.0 if self.attack_token > 0 else 0.0

        observation = np.array(
            [player_hp]
            + [opponent_hp]
            + player_tokens
            + opponent_tokens
            + [phase_indicator]
            + [current_player_indicator]
            + [attack_token_norm],
            dtype=np.float32,
        )

        return observation

    def render(self):
        player = "Player 1" if self.current_player == 1 else "Player 2"
        opponent = "Player 2" if self.current_player == 1 else "Player 1"
        s = f"Current turn: {player}\n"
        s += f"{player}'s Base HP: {self.base_hp[self.current_player]}\n"
        s += f"{opponent}'s Base HP: {self.base_hp[-self.current_player]}\n"

        s += f"{player}'s Available Tokens: {[i+1 for i in range(10) if self.tokens[self.current_player][i+1]]}\n"
        s += f"{opponent}'s Available Tokens: {[i+1 for i in range(10) if self.tokens[-self.current_player][i+1]]}\n"

        if self.phase == "attack":
            s += "Phase: Attack\n"
            s += "Select a token (1-10) to attack with.\n"
        elif self.phase == "defense":
            s += "Phase: Defense\n"
            s += f"Attack token: {self.attack_token}\n"
            s += "Select 0 to not defend, or a token (1-10) to defend with.\n"
        else:
            s += "Invalid Phase\n"

        return s

    def valid_moves(self):
        if self.phase == "attack":
            # Valid moves are available tokens numbered 1-10
            return [i for i in range(1, 11) if self.tokens[self.current_player][i]]
        elif self.phase == "defense":
            # Valid moves are 0 (not defend) and available tokens 1-10
            available_tokens = [
                i for i in range(1, 11) if self.tokens[self.current_player][i]
            ]
            return [0] + available_tokens
        else:
            return []
