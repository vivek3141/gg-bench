import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: agent can choose move cards 1 to 5 (indices 0 to 4)
        self.action_space = spaces.Discrete(5)

        # Define observation space:
        # - Marker position: -5 to +5 (represented by an integer)
        # - Agent's remaining move cards: 5 elements (1 if available, 0 if used)
        # - Opponent's remaining move cards: 5 elements (1 if available, 0 if used)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(11,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Marker starts at position 0
        self.marker_position = 0

        # Agent's move cards: [1, 1, 1, 1, 1] (cards 1 to 5 available)
        self.agent_cards = np.ones(5, dtype=np.int8)

        # Opponent's move cards: [1, 1, 1, 1, 1]
        self.opponent_cards = np.ones(5, dtype=np.int8)

        self.done = False
        self.info = {}

        # Construct the initial observation
        observation = self._get_obs()
        return observation, self.info  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, ignore further actions
            return self._get_obs(), 0, self.done, False, self.info

        # Convert action index (0 to 4) to move card value (1 to 5)
        agent_move_card = action + 1

        # Check if the agent's move card is valid (not yet used)
        if self.agent_cards[action] == 0:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            return self._get_obs(), reward, self.done, False, self.info

        # Agent's move card is valid; remove it from agent's remaining cards
        self.agent_cards[action] = 0

        # Opponent selects a move card randomly from their available cards
        opponent_available_indices = np.where(self.opponent_cards == 1)[0]
        if len(opponent_available_indices) == 0:
            # No more opponent cards left; game should end
            opponent_move_card = None
        else:
            opponent_action = self.np_random.choice(opponent_available_indices)
            opponent_move_card = opponent_action + 1
            # Remove the opponent's used move card
            self.opponent_cards[opponent_action] = 0

        # Compute net movement
        if opponent_move_card is None:
            # Opponent has no cards left, agent wins by default
            self.done = True
            reward = 1
            return self._get_obs(), reward, self.done, False, self.info

        if agent_move_card == opponent_move_card:
            # Net movement is 0; marker does not move
            net_movement = 0
        else:
            higher_card = max(agent_move_card, opponent_move_card)
            lower_card = min(agent_move_card, opponent_move_card)
            net_movement = higher_card - lower_card

            # Direction of movement
            if agent_move_card > opponent_move_card:
                # Agent wins this round; marker moves towards +5
                self.marker_position += net_movement
                if self.marker_position >= 5:
                    self.marker_position = 5
                    self.done = True
                    reward = 1  # Agent wins
            else:
                # Opponent wins this round; marker moves towards -5
                self.marker_position -= net_movement
                if self.marker_position <= -5:
                    self.marker_position = -5
                    self.done = True
                    reward = 0  # Agent loses
        # After movement, check if the game has ended
        if not self.done:
            # Check if all move cards have been used
            if np.all(self.agent_cards == 0) and np.all(self.opponent_cards == 0):
                # All move cards used; determine winner
                if self.marker_position > 0:
                    # Marker on positive side; agent wins
                    self.done = True
                    reward = 1
                elif self.marker_position < 0:
                    # Marker on negative side; opponent wins
                    self.done = True
                    reward = 0
                else:
                    # Marker at 0; tiebreaker applies
                    # Check who last moved the marker towards their opponent's end
                    # Since opponent acted last (agent is current player), agent loses
                    self.done = True
                    reward = 0  # Agent loses
            else:
                # Game continues
                reward = 0
        else:
            if reward != 1:
                reward = 0  # Agent loses

        return self._get_obs(), reward, self.done, False, self.info

    def render(self):
        # Create a visual representation of the number line and marker position
        line = ""
        for position in range(-5, 6):
            if position == self.marker_position:
                line += " M "  # M represents the marker
            else:
                line += " - "
        agent_cards_str = " ".join(
            [str(i + 1) for i in range(5) if self.agent_cards[i] == 1]
        )
        opponent_cards_str = "X" * np.sum(self.opponent_cards)
        render_str = (
            f"Number Line:\n{line}\n"
            f"Marker Position: {self.marker_position}\n"
            f"Your Remaining Cards: {agent_cards_str}\n"
            f"Opponent's Remaining Cards: {opponent_cards_str}\n"
        )
        return render_str

    def valid_moves(self):
        # Return a list of indices (0 to 4) of the agent's available move cards
        return [i for i in range(5) if self.agent_cards[i] == 1]

    def _get_obs(self):
        # Assemble the observation array
        observation = np.zeros(11, dtype=np.int8)
        # First element is the marker position
        observation[0] = self.marker_position
        # Next 5 elements are the agent's remaining move cards
        observation[1:6] = self.agent_cards
        # Next 5 elements are the opponent's remaining move cards
        observation[6:11] = self.opponent_cards
        return observation
