import numpy as np
import json
import matplotlib.pyplot as plt
with open("grid_environment.json", "r") as f:
    data = json.load(f)
grid_size = data["grid_size"]
start_position = tuple(data["start_position"])
goal_position = tuple(data["end_position"])
obstacles = [tuple(obs) for obs in data["obstacles"]]
grid = np.zeros(grid_size)
for obs in obstacles:
    grid[obs] = -1
grid[goal_position] = 10
class GridEnvironment:
    def __init__(self, grid, start, goal, obstacles):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = start
        self.actions = ["up", "down", "left", "right"]
        self.grid_size = grid.shape
    def reset(self):
        self.state = self.start
        return self.state
    def step(self, action):
        x, y = self.state
        if action == "up":
            next_state = (x - 1, y)
        elif action == "down":
            next_state = (x + 1, y)
        elif action == "left":
            next_state = (x, y - 1)
        elif action == "right":
            next_state = (x, y + 1)
        else:
            raise ValueError("Invalid action")
        if (0 <= next_state[0] < self.grid_size[0] and 0 <= next_state[1] < self.grid_size[1] and next_state not in self.obstacles):
            self.state = next_state
        else:
            next_state = self.state
        if next_state == self.goal:
            reward = 10
            done = True
        elif next_state in self.obstacles:
            reward = -10
            done = False
        else:
            reward = -1
            done = False
        return next_state, reward, done
env = GridEnvironment(grid, start_position, goal_position, obstacles)
q_table = np.zeros((*grid_size, len(env.actions)))
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 5000
for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(len(env.actions))
        else:
            action_idx = np.argmax(q_table[state])
        action = env.actions[action_idx]
        next_state, reward, done = env.step(action)
        q_table[state][action_idx] = q_table[state][action_idx] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action_idx])
        state = next_state
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
print("Training completed!")
def visualize_path(env, q_table):
    state = env.reset()
    path = [state]
    done = False
    while not done:
        action_idx = np.argmax(q_table[state])
        action = env.actions[action_idx]
        state, _, done = env.step(action)
        path.append(state)
    plt.figure(figsize=(6, 6))
    plt.grid(True)
    for obs in env.obstacles:
        plt.scatter(obs[1], obs[0], c="red", s=200, label="Obstacle" if obs == env.obstacles[0] else "")
    plt.scatter(env.start[1], env.start[0], c="blue", s=200, label="Start")
    plt.scatter(env.goal[1], env.goal[0], c="green", s=200, label="Goal")
    path_x, path_y = zip(*path)
    plt.plot(path_y, path_x, c="black", label="Path")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.title("Agent's Path")
    plt.show()
visualize_path(env, q_table)
state = env.reset()
done = False
cumulative_reward = 0
while not done:
    action_idx = np.argmax(q_table[state])
    action = env.actions[action_idx]
    state, reward, done = env.step(action)
    cumulative_reward += reward
print("Cumulative reward for the trained agent:", cumulative_reward)
