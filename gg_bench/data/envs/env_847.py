import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space

        # Action space: Discrete(5), representing the 5 powers
        # Actions 0-4 correspond to powers [1, 2, 4, 8, 16]
        self.action_space = spaces.Discrete(5)

        # Observation space: Contains the state of the game
        # Indexes 0-4: Player's available powers (1 if available, 0 if used)
        # Indexes 5-9: Opponent's available powers
        # Index 10: Player's duel victories
        # Index 11: Opponent's duel victories
        self.observation_space = spaces.Box(low=0, high=3, shape=(12,), dtype=np.int32)

        self.powers = np.array([1, 2, 4, 8, 16], dtype=np.int32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the game state
        self.player_powers = np.ones(5, dtype=np.int32)
        self.opponent_powers = np.ones(5, dtype=np.int32)
        self.player_used_powers = np.zeros(5, dtype=np.int32)
        self.opponent_used_powers = np.zeros(5, dtype=np.int32)
        self.player_duel_victories = 0
        self.opponent_duel_victories = 0
        self.done = False
        self.in_tiebreaker = False
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            # Game is over
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        if self.in_tiebreaker:
            # Tiebreaker duel, players can select any previously used power
            player_power = self.powers[action]
            # Opponent selects power (randomly from all powers)
            opponent_action = self.np_random.choice(range(5))
            opponent_power = self.powers[opponent_action]
        else:
            # Player selects power
            player_power = self.powers[action]
            self.player_powers[action] = 0  # Mark power as used
            self.player_used_powers[action] = 1  # Mark power as used

            # Opponent selects power (randomly from available powers)
            opponent_valid_moves = [
                i for i, available in enumerate(self.opponent_powers) if available == 1
            ]
            opponent_action = self.np_random.choice(opponent_valid_moves)
            opponent_power = self.powers[opponent_action]
            self.opponent_powers[opponent_action] = 0  # Mark power as used
            self.opponent_used_powers[opponent_action] = 1  # Mark used

        # Determine duel outcome
        duel_winner = None

        if player_power == opponent_power:
            # Tie
            duel_winner = None
        elif player_power == 2 * opponent_power:
            # Exact Double Rule applies, opponent wins
            duel_winner = "opponent"
        elif opponent_power == 2 * player_power:
            # Exact Double Rule applies, player wins
            duel_winner = "player"
        elif player_power > opponent_power:
            duel_winner = "player"
        else:
            duel_winner = "opponent"

        # Update duel victories
        reward = 0
        if duel_winner == "player":
            if self.in_tiebreaker:
                # Tiebreaker duel, player wins the game
                self.player_duel_victories += 1
                reward = 1
                self.done = True
            else:
                self.player_duel_victories += 1
                reward = 1
        elif duel_winner == "opponent":
            if self.in_tiebreaker:
                # Tiebreaker duel, opponent wins the game
                self.opponent_duel_victories += 1
                self.done = True
            else:
                self.opponent_duel_victories += 1
        else:
            # Tie, no points awarded
            pass

        # Check for game over conditions
        if self.player_duel_victories >= 3 or self.opponent_duel_victories >= 3:
            self.done = True
        elif (
            not self.in_tiebreaker
            and np.sum(self.player_powers) == 0
            and np.sum(self.opponent_powers) == 0
        ):
            # All powers have been used
            if self.player_duel_victories > self.opponent_duel_victories:
                self.done = True
            elif self.player_duel_victories < self.opponent_duel_victories:
                self.done = True
            else:
                # Tie in duel victories, initiate tiebreaker duel
                self.in_tiebreaker = True
                # Reset powers to allow selection from previously used powers
                self.player_powers = self.player_used_powers.copy()
                self.opponent_powers = self.opponent_used_powers.copy()
                self.player_used_powers = np.zeros(5, dtype=np.int32)
                self.opponent_used_powers = np.zeros(5, dtype=np.int32)

        observation = self._get_observation()
        return observation, reward, self.done, False, {}

    def render(self):
        # Visual representation of the game state
        output = ""
        if self.in_tiebreaker:
            output += "Tiebreaker Duel!\n"
        output += "Player's unused powers: "
        output += ", ".join(
            [str(self.powers[i]) for i in range(5) if self.player_powers[i] == 1]
        )
        output += "\nOpponent's unused powers: "
        output += ", ".join(
            [str(self.powers[i]) for i in range(5) if self.opponent_powers[i] == 1]
        )
        output += f"\nDuel Victories - Player: {self.player_duel_victories}, Opponent: {self.opponent_duel_victories}\n"
        return output

    def valid_moves(self):
        if self.done:
            return []
        if self.in_tiebreaker:
            # In tiebreaker, all previously used powers are available
            return [i for i in range(5)]
        else:
            return [
                i for i, available in enumerate(self.player_powers) if available == 1
            ]

    def _get_observation(self):
        # Create the observation array
        observation = np.concatenate(
            (
                self.player_powers,
                self.opponent_powers,
                np.array(
                    [self.player_duel_victories, self.opponent_duel_victories],
                    dtype=np.int32,
                ),
            )
        )
        return observation
