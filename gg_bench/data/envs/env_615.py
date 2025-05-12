import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            5
        )  # Actions: 0 to 4 corresponding to Spells 1 to 5
        # Observation: [Current Player HP, Opponent HP, Current Player Blocking (0 or 1), Opponent Blocking (0 or 1)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([10, 10, 1, 1]),
            dtype=np.int32,
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_hp = [10, 10]  # Player 1 and Player 2 HP
        self.is_blocking = [False, False]  # Blocking status for Player 1 and Player 2
        self.will_block_next_turn = [False, False]  # For updating blocking status
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False

        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}
        if action not in [0, 1, 2, 3, 4]:
            self.done = True
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )  # Invalid action penalty

        opponent = 1 - self.current_player

        # Update opponent's blocking status at the beginning of the turn
        self.is_blocking[opponent] = self.will_block_next_turn[opponent]
        self.will_block_next_turn[opponent] = (
            False  # Reset the blocking status for next turn
        )

        # Process the spell action
        if action == 0:
            # Spell 1: Magic Missile
            # Damage: 1 HP to opponent (unblockable)
            self.player_hp[opponent] -= 1
        elif action == 1:
            # Spell 2: Fireball
            # Damage: 2 HP to opponent (blockable)
            if not self.is_blocking[opponent]:
                self.player_hp[opponent] -= 2
        elif action == 2:
            # Spell 3: Shield
            # Activate blocking status for next turn
            pass  # No damage dealt
        elif action == 3:
            # Spell 4: Heal
            # Restore 2 HP to current player (max 10)
            self.player_hp[self.current_player] += 2
            if self.player_hp[self.current_player] > 10:
                self.player_hp[self.current_player] = 10
        elif action == 4:
            # Spell 5: Lightning Strike
            # Damage: 4 HP to opponent (fails if opponent is blocking)
            if not self.is_blocking[opponent]:
                self.player_hp[opponent] -= 4
        else:
            # Invalid action (should not reach here)
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Ensure HP bounds
        if self.player_hp[opponent] < 0:
            self.player_hp[opponent] = 0
        if self.player_hp[self.current_player] > 10:
            self.player_hp[self.current_player] = 10

        # Check for victory
        if self.player_hp[opponent] <= 0:
            self.done = True
            reward = 1  # Current player wins
            return self._get_observation(), reward, True, False, {}
        else:
            reward = 0

        # At the end of current player's turn, update blocking status for next turn
        if action == 2:
            # Casted Shield
            self.will_block_next_turn[self.current_player] = True
        else:
            self.will_block_next_turn[self.current_player] = False

        # Switch current player
        self.current_player = opponent

        # Prepare observation
        return self._get_observation(), reward, False, False, {}

    def _get_observation(self):
        opponent = 1 - self.current_player
        observation = np.array(
            [
                self.player_hp[self.current_player],
                self.player_hp[opponent],
                int(self.is_blocking[self.current_player]),
                int(self.is_blocking[opponent]),
            ],
            dtype=np.int32,
        )
        return observation

    def render(self):
        opponent = 1 - self.current_player
        player_num = self.current_player + 1
        opponent_num = opponent + 1
        render_str = f"Player {player_num}'s Turn\n"
        render_str += "----------------\n"
        render_str += f"Your HP: {self.player_hp[self.current_player]}\n"
        render_str += f"Opponent's HP: {self.player_hp[opponent]}\n"
        render_str += f"Your Blocking Status: {'Active' if self.is_blocking[self.current_player] else 'Inactive'}\n"
        render_str += f"Opponent's Blocking Status: {'Active' if self.is_blocking[opponent] else 'Inactive'}\n"
        render_str += "----------------\n"
        render_str += "Available Spells:\n"
        render_str += "0 - Magic Missile\n"
        render_str += "1 - Fireball\n"
        render_str += "2 - Shield\n"
        render_str += "3 - Heal\n"
        render_str += "4 - Lightning Strike\n"
        return render_str

    def valid_moves(self):
        return [0, 1, 2, 3, 4]  # All spells are always valid to cast
