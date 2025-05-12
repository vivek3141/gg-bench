import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space
        # Actions:
        # 0: Move forward 1
        # 1: Move forward 2
        # 2: Move forward 3
        # 3: Rest (only at own base)
        # 4: Play battle card 1
        # 5: Play battle card 2
        # 6: Play battle card 3
        # 7: Play battle card 4
        # 8: Play battle card 5
        self.action_space = spaces.Discrete(9)

        # Define observation space
        # Observation includes:
        # - Player's hero position (integer 0-10)
        # - Opponent's hero position (integer 0-10)
        # - Player's available battle cards (5 values, 1.0 for available, 0.0 for used)
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(12,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_player = 1  # Agent is Player 1
        self.done = False
        self.in_battle = False
        self.player_positions = {1: 0, 2: 0}  # Positions of Player 1 and Player 2
        self.player_cards = {
            1: {
                1: True,
                2: True,
                3: True,
                4: True,
                5: True,
            },  # True indicates card is available
            2: {1: True, 2: True, 3: True, 4: True, 5: True},
        }
        self.battle_card_selection = {}  # Store selected battle cards during battle

        # Initialize observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done, False, {}

        reward = 0
        info = {}

        if self.in_battle:
            # Agent must select a battle card
            if action < 4 or action > 8:
                # Invalid action
                self.done = True
                reward = -10
                return self._get_observation(), reward, self.done, False, info

            card_number = action - 3  # Map action 4-8 to card number 1-5
            if not self.player_cards[1][card_number]:
                # Card not available
                self.done = True
                reward = -10
                return self._get_observation(), reward, self.done, False, info

            # Agent selects battle card
            self.battle_card_selection[1] = card_number
            # Simulate opponent's battle card selection
            opponent_cards = [
                card for card, available in self.player_cards[2].items() if available
            ]
            if opponent_cards:
                opponent_card = np.random.choice(opponent_cards)
                self.battle_card_selection[2] = opponent_card
            else:
                # Opponent has no available cards, default to minimum card
                opponent_card = 1
                self.battle_card_selection[2] = opponent_card

            # Process battle outcome
            player_card = self.battle_card_selection[1]
            opponent_card = self.battle_card_selection[2]

            # Discard used cards
            self.player_cards[1][player_card] = False
            self.player_cards[2][opponent_card] = False

            if player_card > opponent_card:
                # Player wins the battle
                self.player_positions[2] = 0  # Opponent goes back to base
                # Player remains on position
            elif player_card < opponent_card:
                # Opponent wins the battle
                self.player_positions[1] = 0  # Player goes back to base
                # Opponent remains on position
            else:
                # Tie, both go back to base
                self.player_positions[1] = 0
                self.player_positions[2] = 0

            # Reset battle state
            self.in_battle = False
            self.battle_card_selection = {}

            # Check if game over
            if self.player_positions[1] == 10:
                self.done = True
                reward = 1  # Agent wins
                return self._get_observation(), reward, self.done, False, info
            elif self.player_positions[2] == 10:
                self.done = True
                reward = -1  # Agent loses
                return self._get_observation(), reward, self.done, False, info

            # Switch to opponent's turn
            self._opponent_turn()
            return self._get_observation(), reward, self.done, False, info

        else:
            # Agent must select action 0-3 (move/rest)
            if action < 0 or action > 3:
                # Invalid action
                self.done = True
                reward = -10
                return self._get_observation(), reward, self.done, False, info

            if action == 3:
                # Rest
                if self.player_positions[1] != 0:
                    # Cannot rest if not at base
                    self.done = True
                    reward = -10
                    return self._get_observation(), reward, self.done, False, info
                else:
                    # Restore battle cards
                    for card in self.player_cards[1]:
                        self.player_cards[1][card] = True
            else:
                # Move forward 1, 2, or 3 positions
                steps = action + 1  # action 0 -> 1 step, action 1 -> 2 steps, etc.
                new_position = self.player_positions[1] + steps
                if new_position > 10:
                    new_position = 10  # Can't move beyond position 10
                self.player_positions[1] = new_position

                # Check if player reaches opponent's tower
                if self.player_positions[1] == 10:
                    self.done = True
                    reward = 1  # Agent wins
                    return self._get_observation(), reward, self.done, False, info

                # Check for battle
                if self.player_positions[1] == self.player_positions[2]:
                    self.in_battle = True
                    # Agent will select battle card in next action

            # Switch to opponent's turn
            self._opponent_turn()

            # Check if game over after opponent's turn
            if self.done:
                if self.player_positions[2] == 10:
                    reward = -1  # Agent loses
                elif self.player_positions[1] == 10:
                    reward = 1  # Agent wins
                else:
                    reward = 0
                return self._get_observation(), reward, self.done, False, info

            return self._get_observation(), reward, self.done, False, info

    def _opponent_turn(self):
        if self.done:
            return

        if self.in_battle:
            # Opponent selects battle card
            opponent_cards = [
                card for card, available in self.player_cards[2].items() if available
            ]
            if opponent_cards:
                opponent_card = np.random.choice(opponent_cards)
                self.battle_card_selection[2] = opponent_card
            else:
                opponent_card = 1  # Default to card 1 if no cards available
                self.battle_card_selection[2] = opponent_card

            # Simulate agent's battle card selection randomly if not already selected
            if 1 not in self.battle_card_selection:
                agent_cards = [
                    card
                    for card, available in self.player_cards[1].items()
                    if available
                ]
                if agent_cards:
                    agent_card = np.random.choice(agent_cards)
                    self.battle_card_selection[1] = agent_card
                else:
                    agent_card = 1
                    self.battle_card_selection[1] = agent_card

            # Process battle outcome
            agent_card = self.battle_card_selection[1]
            opponent_card = self.battle_card_selection[2]

            # Discard used cards
            self.player_cards[1][agent_card] = False
            self.player_cards[2][opponent_card] = False

            if agent_card > opponent_card:
                # Agent wins the battle
                self.player_positions[2] = 0  # Opponent goes back to base
                # Agent remains on position
            elif agent_card < opponent_card:
                # Opponent wins the battle
                self.player_positions[1] = 0  # Agent goes back to base
                # Opponent remains on position
            else:
                # Tie, both go back to base
                self.player_positions[1] = 0
                self.player_positions[2] = 0

            # Reset battle state
            self.in_battle = False
            self.battle_card_selection = {}

            # Check if game over
            if self.player_positions[2] == 10:
                self.done = True
            elif self.player_positions[1] == 10:
                self.done = True

        else:
            # Opponent's normal action
            available_actions = []

            # Move forward actions
            for move in [1, 2, 3]:
                new_position = self.player_positions[2] + move
                if new_position <= 10:
                    available_actions.append(("move", move))
            # Rest action
            if self.player_positions[2] == 0:
                available_actions.append(("rest", 0))

            # Select an action randomly
            if available_actions:
                action_type, value = np.random.choice(available_actions)
                if action_type == "rest":
                    # Opponent rests and recovers battle cards
                    for card in self.player_cards[2]:
                        self.player_cards[2][card] = True
                elif action_type == "move":
                    # Opponent moves forward
                    self.player_positions[2] += value
                    if self.player_positions[2] > 10:
                        self.player_positions[2] = 10

                    # Check if opponent reaches agent's tower
                    if self.player_positions[2] == 10:
                        self.done = True
                        return

                    # Check for battle
                    if self.player_positions[2] == self.player_positions[1]:
                        self.in_battle = True
                        # Opponent will select battle card in next action
            else:
                # No available actions (should not happen)
                pass

    def render(self):
        board_str = "Battlefield Positions:\n"
        board = ["  " for _ in range(11)]
        board[self.player_positions[1]] = "P1"
        board[self.player_positions[2]] = (
            "P2" if board[self.player_positions[2]] == "  " else "B"
        )  # 'B' for Battle
        positions = " | ".join(board)
        board_str += positions + "\n"
        player_cards = [
            str(card) for card, available in self.player_cards[1].items() if available
        ]
        opponent_cards = [
            str(card) for card, available in self.player_cards[2].items() if available
        ]
        board_str += f"Your available battle cards: {', '.join(player_cards)}\n"
        board_str += f"Opponent's available battle cards: {', '.join(opponent_cards)}\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        if self.done:
            return valid_actions

        if self.in_battle:
            # Must select available battle card
            for action in range(4, 9):  # Actions 4 to 8 correspond to cards 1 to 5
                card_number = action - 3
                if self.player_cards[1][card_number]:
                    valid_actions.append(action)
        else:
            # Normal actions
            if self.player_positions[1] == 0:
                valid_actions.append(3)  # Can rest
            else:
                # Cannot rest
                pass
            for action in range(
                0, 3
            ):  # Actions 0 to 2 correspond to moving 1, 2, or 3 steps
                steps = action + 1
                new_position = self.player_positions[1] + steps
                if new_position <= 10:
                    valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Observation includes:
        # - Player's hero position (0-10)
        # - Opponent's hero position (0-10)
        # - Player's available battle cards (5 values)
        # - In battle (0 or 1)
        # - Opponent's distance to their tower (0-10)
        obs = np.zeros(12, dtype=np.float32)
        obs[0] = self.player_positions[1]
        obs[1] = self.player_positions[2]
        obs[2:7] = np.array(
            [
                1.0 if available else 0.0
                for card, available in self.player_cards[1].items()
            ]
        )
        obs[7] = 1.0 if self.in_battle else 0.0
        obs[8] = 10 - self.player_positions[2]  # Opponent's distance to their tower
        # Those are the most relevant features for the agent
        # You can include more information if necessary
        return obs
