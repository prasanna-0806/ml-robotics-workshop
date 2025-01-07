Summary-

We fine-tuned the Q-learning algorithm to maximize an agent's rewards in a grid environment by experimenting with the following hyperparameters:

Learning Rate (α): The optimal value was 0.1, balancing between fast learning and stability

Discount Factor (γ): The best value was 0.99, prioritizing long-term rewards

Exploration Rate (ε): Starting at 1.0 and decaying with 0.995 allowed the agent to explore initially and gradually exploit learned policies

Over 5000 episodes, the agent improved its performance, as reflected in the reward trend, which showed an increase in rewards as the agent shifted from exploration to exploiting its learned knowledge

Output-

![download](https://github.com/user-attachments/assets/16b38840-d97c-4cfb-861e-57971a077918)
