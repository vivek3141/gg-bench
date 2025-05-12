import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Agent can choose numbers from 1 to 5, represented as 0 to 4 in action space
        self.action_space = spaces.Discrete(5)

        # Observation space consists of:
        # [score_player1, score_player2, last_move_player1, last_move_player2]
        # Scores range from 0 to 5
        # Moves range from 1 to 5 (0 used as placeholder for initial moves)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([5, 5, 5, 5]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.score_player1 = 0
        self.score_player2 = 0
        self.last_move_player1 = 0
        self.last_move_player2 = 0
        self.done = False
        self.current_player = 1  # Agent is always player 1
        observation = np.array(
            [
                self.score_player1,
                self.score_player2,
                self.last_move_player1,
                self.last_move_player2,
            ],
            dtype=np.int32,
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if game is already over
        if self.done:
            return (
                self._get_observation(),
                0,
                True,
                False,
                {},
            )

        # Validate action
        if action not in [0, 1, 2, 3, 4]:
            self.done = True
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )

        # Agent's move (Player 1)
        agent_move = action + 1  # Convert to number between 1 and 5

        # Opponent's move (Player 2), randomly selected
        opponent_action = self.np_random.integers(0, 5)
        opponent_move = opponent_action + 1

        # Update last moves
        self.last_move_player1 = agent_move
        self.last_move_player2 = opponent_move

        # Determine round outcome
        reward = 0  # Default reward

        if agent_move == opponent_move:
            # Round is a draw; no points awarded
            pass
        elif abs(agent_move - opponent_move) == 1 and not (
            agent_move == 1
            and opponent_move == 5
            or agent_move == 5
            and opponent_move == 1
        ):
            # Numbers are consecutive
            if agent_move < opponent_move:
                # Agent wins the round
                self.score_player1 += 1
                reward = 1
            else:
                # Opponent wins the round
                self.score_player2 += 1
        else:
            # Numbers are not consecutive and not the same
            if agent_move > opponent_move:
                # Agent wins the round
                self.score_player1 += 1
                reward = 1
            else:
                # Opponent wins the round
                self.score_player2 += 1

        # Check for game over
        if self.score_player1 >= 5 or self.score_player2 >= 5:
            self.done = True

        observation = self._get_observation()

        return observation, reward, self.done, False, {}

    def render(self):
        # Return a string representation of the game state
        render_str = "-----------------------------\n"
        render_str += (
            f"Scores:\nPlayer 1: {self.score_player1}\nPlayer 2: {self.score_player2}\n"
        )
        render_str += f"Last Moves:\nPlayer 1 chose: {self.last_move_player1}\nPlayer 2 chose: {self.last_move_player2}\n"
        render_str += "-----------------------------\n"
        if self.done:
            if self.score_player1 >= 5:
                render_str += "*** Player 1 wins the game! ***\n"
            else:
                render_str += "*** Player 2 wins the game! ***\n"
        return render_str

    def valid_moves(self):
        # Return list of valid moves (0 to 4 corresponding to numbers 1 to 5)
        if self.done:
            return []
        else:
            return [0, 1, 2, 3, 4]

    def _get_observation(self):
        return np.array(
            [
                self.score_player1,
                self.score_player2,
                self.last_move_player1,
                self.last_move_player2,
            ],
            dtype=np.int32,
        )
