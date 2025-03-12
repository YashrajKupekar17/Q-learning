import gym
import numpy as np
import pickle
from huggingface_hub import hf_hub_download


repo_id = "yashrajkupekar/Taxi_7200states"
filename = "q-learning.pkl"

# Download the model file from the Hub and load it.
model_path = hf_hub_download(repo_id=repo_id, filename=filename)
with open(model_path, "rb") as f:
    model = pickle.load(f)


q_table = model["qtable"]
# env_id = model["env_id"]

# Create the Taxi-v3 environment in human render mode.
env = gym.make("Taxi-v3", render_mode="human").env  

total_epochs = 0
total_penalties = 0
episodes = 100

for _ in range(episodes):
    state = env.reset()[0]  
    epochs, penalties = 0, 0
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info, _ = env.step(action)
        if reward == -10:
            penalties += 1
        epochs += 1
    total_epochs += epochs
    total_penalties += penalties

print("Results after {} episodes:".format(episodes))
print("Average time steps per episode: {:.2f}".format(total_epochs / episodes))
print("Average penalties per episode: {:.2f}".format(total_penalties / episodes))
