import math
from copy import deepcopy

# Internal Imports
from gg_bench.utils.inference.predict import get_prediction


class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state  # The game state at this node
        self.parent = parent  # Parent node
        self.move = move  # The move that led to this node
        self.children = []  # Child nodes
        self.visits = 0  # Number of times this node was visited
        self.wins = 0  # Number of wins from this node
        self.player = state.get_current_player()
        self.untried_moves = state.valid_moves()  # Moves not yet tried from this node

    def is_fully_expanded(self):
        """Check if all possible moves have been tried."""
        return len(self.untried_moves) == 0

    def ucb1(self, c_param=1.4):
        """Calculate the UCB1 value for this node."""
        if self.visits == 0:
            return float("inf")
        win_rate = self.wins / self.visits
        exploration = c_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return win_rate + exploration

    def best_child(self, c_param=1.4):
        """Select the child with the highest UCB1 value."""
        return max(self.children, key=lambda child: child.ucb1(c_param))

    def expand(self):
        """Expand a child node from an untried move."""
        move = self.untried_moves.pop()
        next_state = deepcopy(self.state)
        next_state.step(move)
        child_node = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node


def tree_policy(node):
    """Selection and expansion phases of MCTS."""
    while not node.state.game_over():
        if not node.is_fully_expanded():
            return node.expand()
        else:
            node = node.best_child()
    return node


def default_policy(state, agent):
    """Simulation phase of MCTS: play out the game randomly to the end."""
    current_state = deepcopy(state)
    while not current_state.game_over() and current_state.valid_moves():
        move = get_prediction(
            agent=agent,
            obs=current_state.get_obs(),
            valid_moves=current_state.valid_moves(),
        )
        current_state.step(move)

    return current_state.get_reward() * current_state.get_current_player()


def backup(node, reward):
    """Backpropagation phase of MCTS."""
    while node is not None:
        node.visits += 1
        if reward == node.player:
            node.wins += 1
        elif reward == 0:
            node.wins += 0.5  # Consider a draw as a half-win
        node = node.parent


def get_mcts_prediction(env, agent, max_iter=100):
    """Perform MCTS starting from the root_state."""
    root_node = MCTSNode(env)
    for _ in range(max_iter):
        node = root_node
        # Selection and Expansion
        node = tree_policy(node)
        # Simulation
        reward = default_policy(node.state, agent)
        # Backpropagation
        backup(node, reward)
    # Choose the move with the most visits
    best_child = max(root_node.children, key=lambda c: c.visits)
    return best_child.move
