import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define the prime numbers available for attacks
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19]

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(len(self.primes))
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_hp = [100, 100]  # [Player 1 HP, Player 2 HP]
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        if self.done:
            # If game has ended, return the current state
            return self._get_obs(), -10, True, False, {}

        # Map the action to the corresponding prime number attack
        prime_attack = self.primes[action]

        attacker = self.current_player
        opponent = 1 - self.current_player

        attacker_hp = self.player_hp[attacker]
        opponent_hp = self.player_hp[opponent]

        # Initialize reward
        reward = -10  # Negative reward for each valid move

        # Determine if the attack is successful or reflected
        if opponent_hp % prime_attack != 0:
            # Attack is successful
            self.player_hp[opponent] -= prime_attack
        else:
            # Attack is reflected
            self.player_hp[attacker] -= prime_attack

        # Update HP after attack
        attacker_hp = self.player_hp[attacker]
        opponent_hp = self.player_hp[opponent]

        # Check for win conditions
        if opponent_hp <= 0 and attacker_hp <= 0:
            # Both players have 0 or less HP; attacking player wins
            self.done = True
            reward = 1  # Reward for winning the game
        elif opponent_hp <= 0:
            # Opponent has 0 or less HP; attacking player wins
            self.done = True
            reward = 1  # Reward for winning the game
        elif attacker_hp <= 0:
            # Attacker has 0 or less HP; opponent wins
            self.done = True
            # Reward remains -10 as per instructions
        else:
            # Game continues; switch turns
            self.current_player = opponent

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        # Return a string representation of the current game state
        return (
            f"---------------------------------------\n"
            f"Player {self.current_player + 1}'s Turn\n\n"
            f"Player 1 HP: {self.player_hp[0]}\n"
            f"Player 2 HP: {self.player_hp[1]}\n"
            f"Available Primes: {self.primes}\n"
            f"---------------------------------------\n"
        )

    def valid_moves(self):
        # All primes are always available as valid moves
        return list(range(len(self.primes)))

    def _get_obs(self):
        # Observation is the current player's HP and opponent's HP
        attacker = self.current_player
        opponent = 1 - self.current_player
        return np.array(
            [self.player_hp[attacker], self.player_hp[opponent]], dtype=np.int32
        )
