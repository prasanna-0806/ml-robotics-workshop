import numpy as np
import gym
from gym.wrappers import StepAPICompatibility
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
env = StepAPICompatibility(gym.make("Taxi-v3"), new_step_api=True)
n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros((n_states, n_actions))  
alpha = 0.1  
gamma = 0.99  
epsilon = 1.0  
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000
for episode in range(episodes):
    reset_output = env.reset()
    if isinstance(reset_output, tuple):  
        state = reset_output[0]
    else:  
        state = reset_output
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample() 
        else:
            action = np.argmax(q_table[state])
        step_output = env.step(action)
        if isinstance(step_output, tuple): 
            next_state, reward, done, _, _ = step_output
        else:  
            next_state, reward, done, _ = step_output
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        state = next_state
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
print("Trained Q-Table:")
print(q_table)
