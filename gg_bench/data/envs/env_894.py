import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 10 possible actions
        # Actions 0-4: Attack with values 1-5
        # Actions 5-9: Defend with values 1-5
        self.action_space = spaces.Discrete(10)

        # Define observation space
        # Observation includes: player_hp, opponent_hp, last_opponent_action
        # last_opponent_action is -1 if no previous action
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1]),
            high=np.array([10, 10, 9]),
            shape=(3,),
            dtype=np.int32,
        )

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_hp = 10
        self.opponent_hp = 10
        self.done = False
        self.last_opponent_action = -1  # No previous action at the start
        observation = np.array(
            [self.player_hp, self.opponent_hp, self.last_opponent_action],
            dtype=np.int32,
        )
        return observation, {}

    def step(self, action):
        # Check for invalid action or if the game is already over
        if action not in self.valid_moves() or self.done:
            self.done = True
            reward = -10  # Penalty for invalid move
            observation = np.array(
                [self.player_hp, self.opponent_hp, self.last_opponent_action],
                dtype=np.int32,
            )
            return observation, reward, self.done, False, {}

        # Decode the agent's action
        agent_action_type, agent_action_value = self.decode_action(action)

        # Opponent selects an action
        opponent_action_index = self.opponent_policy()
        self.last_opponent_action = opponent_action_index  # Store for observation
        opponent_action_type, opponent_action_value = self.decode_action(
            opponent_action_index
        )

        # Resolve actions according to game rules
        self.resolve_actions(
            agent_action_type,
            agent_action_value,
            opponent_action_type,
            opponent_action_value,
        )

        # Check for victory conditions
        reward = 0
        if self.opponent_hp <= 0 and self.player_hp > 0:
            self.done = True
            reward = 1  # Agent wins
        elif self.player_hp <= 0:
            self.done = True
            reward = 0  # Agent loses; no additional penalty specified

        observation = np.array(
            [self.player_hp, self.opponent_hp, self.last_opponent_action],
            dtype=np.int32,
        )
        return observation, reward, self.done, False, {}

    def render(self):
        # Provide a visual representation of the current state
        state_str = f"Player HP: {self.player_hp}\nOpponent HP: {self.opponent_hp}\n"
        if self.last_opponent_action != -1:
            opponent_action_type, opponent_action_value = self.decode_action(
                self.last_opponent_action
            )
            state_str += f"Last opponent action: {opponent_action_type} {opponent_action_value}\n"
        else:
            state_str += "No previous opponent action.\n"
        return state_str

    def valid_moves(self):
        # List of all valid actions (0-9)
        return list(range(10))

    def decode_action(self, action_index):
        # Decode the action index into action type and value
        if action_index < 5:
            action_type = "attack"
            action_value = action_index + 1
        else:
            action_type = "defend"
            action_value = action_index - 4
        return action_type, action_value

    def opponent_policy(self):
        # Opponent selects a random valid action
        return self.np_random.randint(0, 10)

    def resolve_actions(
        self,
        agent_action_type,
        agent_action_value,
        opponent_action_type,
        opponent_action_value,
    ):
        # Resolve the actions based on game rules

        # Attack vs. Attack
        if agent_action_type == "attack" and opponent_action_type == "attack":
            self.player_hp -= opponent_action_value
            self.opponent_hp -= agent_action_value

        # Attack vs. Defend
        elif agent_action_type == "attack" and opponent_action_type == "defend":
            if opponent_action_value >= agent_action_value:
                pass  # Attack is blocked; no damage
            else:
                self.opponent_hp -= agent_action_value - opponent_action_value

        # Defend vs. Attack
        elif agent_action_type == "defend" and opponent_action_type == "attack":
            if agent_action_value >= opponent_action_value:
                pass  # Attack is blocked; no damage
            else:
                self.player_hp -= opponent_action_value - agent_action_value

        # Defend vs. Defend
        else:
            pass  # No damage dealt

        # Ensure HP does not drop below zero
        self.player_hp = max(self.player_hp, 0)
        self.opponent_hp = max(self.opponent_hp, 0)
