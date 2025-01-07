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
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 5000
reward_trend = []
q_table = np.zeros((*grid_size, len(env.actions)))
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(len(env.actions))
        else:
            action_idx = np.argmax(q_table[state])
        action = env.actions[action_idx]
        next_state, reward, done = env.step(action)
        q_table[state][action_idx] = q_table[state][action_idx] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action_idx])
        state = next_state
        total_reward += reward
    reward_trend.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
plt.plot(reward_trend)
plt.title("Reward Trend over Training Episodes")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.show()
state = env.reset()
done = False
cumulative_reward = 0
while not done:
    action_idx = np.argmax(q_table[state])
    action = env.actions[action_idx]
    state, reward, done = env.step(action)
    cumulative_reward += reward
print("Cumulative reward for the trained agent:", cumulative_reward)
