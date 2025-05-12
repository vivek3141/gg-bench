import numpy as np
import gymnasium as gym
from gymnasium import spaces
import itertools


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 125 possible guesses (digits 1-5 for each of 3 positions)
        self.action_space = spaces.Discrete(125)

        # Define observation space
        # Observation consists of:
        # - Player 1's secret code: 3 digits (values 1-5)
        # - Player 2's secret code: 3 digits (values 1-5)
        # - Current player indicator: 1 or 2
        # - Last N turns (N=10), each consisting of:
        #   - Player 1's guess: 3 digits (values 1-5)
        #   - Feedback to Player 1: 2 numbers (hits 0-3, blows 0-3)
        #   - Player 2's guess: 3 digits (values 1-5)
        #   - Feedback to Player 2: 2 numbers (hits 0-3, blows 0-3)
        # Total observation length: 3 + 3 + 1 + 10 * 10 = 107
        self.N_turns = 10
        self.observation_space = spaces.Box(low=0, high=5, shape=(107,), dtype=np.int32)

        # Pre-compute all possible guesses
        self.possible_digits = [1, 2, 3, 4, 5]
        self.all_possible_guesses = list(
            itertools.product(self.possible_digits, repeat=3)
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly select secret codes for both players
        self.player1_code = self._random_code()
        self.player2_code = self._random_code()

        # Initialize game history
        self.history = []
        self.done = False
        self.current_player = 1  # Player 1 starts

        # Prepare initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}

        # Map action to guess
        guess = self.all_possible_guesses[action]

        # Process the guess
        if self.current_player == 1:
            # Player 1's turn
            feedback = self._compute_feedback(guess, self.player2_code)
            # Update history
            turn_data = {
                "player1_guess": guess,
                "player1_feedback": feedback,
                "player2_guess": None,
                "player2_feedback": None,
            }
            # Check for win
            if feedback["hits"] == 3:
                self.done = True
                reward = 1  # Player 1 wins
            else:
                reward = -10  # Valid move, continue game
        else:
            # Player 2's turn
            feedback = self._compute_feedback(guess, self.player1_code)
            # Update history
            if len(self.history) == 0 or self.history[-1]["player2_guess"] is not None:
                # Start a new turn
                turn_data = {
                    "player1_guess": None,
                    "player1_feedback": None,
                    "player2_guess": guess,
                    "player2_feedback": feedback,
                }
                self.history.append(turn_data)
            else:
                # Update existing turn
                self.history[-1]["player2_guess"] = guess
                self.history[-1]["player2_feedback"] = feedback
            # Check for win
            if feedback["hits"] == 3:
                self.done = True
                reward = 1  # Player 2 wins
            else:
                reward = -10  # Valid move, continue game

        # If the game is not over, switch players
        if not self.done:
            self.current_player = 2 if self.current_player == 1 else 1
        else:
            self.current_player = None  # No current player when game is over

        # Add turn data to history if not already added
        if self.current_player == 1 and "player1_guess" not in self.history[-1]:
            self.history[-1]["player1_guess"] = guess
            self.history[-1]["player1_feedback"] = feedback
            self.history.append(turn_data)

        observation = self._get_observation()
        return observation, reward, self.done, False, {}

    def render(self):
        output = "=== Code Breaker Duel ===\n\n"
        output += f"Player 1's Secret Code: {self.player1_code}\n"
        output += f"Player 2's Secret Code: {self.player2_code}\n"
        output += f"Current Player: {'Player ' + str(self.current_player) if self.current_player else 'None'}\n"
        output += "\nGame History:\n"
        for idx, turn in enumerate(self.history[-self.N_turns :], 1):
            output += f"Turn {idx}:\n"
            if turn["player1_guess"]:
                output += f"  Player 1 guessed: {turn['player1_guess']}, Feedback: {turn['player1_feedback']['hits']} Hits, {turn['player1_feedback']['blows']} Blows\n"
            if turn["player2_guess"]:
                output += f"  Player 2 guessed: {turn['player2_guess']}, Feedback: {turn['player2_feedback']['hits']} Hits, {turn['player2_feedback']['blows']} Blows\n"
        return output

    def valid_moves(self):
        # All moves are valid in this game (0 to 124)
        return list(range(125))

    def _random_code(self):
        return tuple(self.np_random.choice(self.possible_digits, size=3))

    def _compute_feedback(self, guess, code):
        hits = sum(g == c for g, c in zip(guess, code))
        code_counts = {d: code.count(d) for d in set(code)}
        guess_counts = {d: guess.count(d) for d in set(guess)}
        blows = (
            sum(min(code_counts.get(d, 0), guess_counts.get(d, 0)) for d in set(guess))
            - hits
        )
        return {"hits": hits, "blows": blows}

    def _get_observation(self):
        obs = np.zeros(107, dtype=np.int32)
        obs_idx = 0
        # Player 1's secret code
        obs[obs_idx : obs_idx + 3] = self.player1_code
        obs_idx += 3
        # Player 2's secret code
        obs[obs_idx : obs_idx + 3] = self.player2_code
        obs_idx += 3
        # Current player indicator
        obs[obs_idx] = self.current_player if self.current_player else 0
        obs_idx += 1
        # Last N turns
        for turn in self.history[-self.N_turns :]:
            # Player 1's guess
            if turn.get("player1_guess"):
                obs[obs_idx : obs_idx + 3] = turn["player1_guess"]
            obs_idx += 3
            # Feedback to Player 1
            if turn.get("player1_feedback"):
                obs[obs_idx] = turn["player1_feedback"]["hits"]
                obs[obs_idx + 1] = turn["player1_feedback"]["blows"]
            obs_idx += 2
            # Player 2's guess
            if turn.get("player2_guess"):
                obs[obs_idx : obs_idx + 3] = turn["player2_guess"]
            obs_idx += 3
            # Feedback to Player 2
            if turn.get("player2_feedback"):
                obs[obs_idx] = turn["player2_feedback"]["hits"]
                obs[obs_idx + 1] = turn["player2_feedback"]["blows"]
            obs_idx += 2

        return obs
