

import numpy as np
import gymnasium as gym
from huggingface_hub import hf_hub_download
import pickle
import time

def load_from_hub(repo_id, filename):
    """
    Load a model from Hugging Face Hub
    :param repo_id: id of the model repository from the HF Hub
    :param filename: name of the model file
    """
    # Download the model file from the Hub
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    # Load the model
    with open(local_path, "rb") as f:
        model = pickle.load(f)
    
    return model

def test_model_rendering(model, episodes=10, render_mode="human", delay=0.5):
    """
    Test the model with rendering
    :param model: The loaded model from HF Hub
    :param episodes: Number of episodes to run
    :param render_mode: The render mode ("human" or "rgb_array")
    :param delay: Time delay between steps (in seconds)
    """
    # Create the environment
    # env_kwargs = {}
    # if "map_name" in model:
    #     env_kwargs["map_name"] = model["map_name"]
    # if "slippery" in model:
    #     env_kwargs["is_slippery"] = model["slippery"]
    desc = [
     "SFFFFFFF",
      "FFFFFFFF",
      "FFFHFFFF",
      "FFFFFHFF",
      "FFFHFFFF",
      "FHHFFFHF",
      "FHFFHFHF",
      "FFFHFFFG",
    ]
    env = gym.make(model["env_id"],desc=desc, is_slippery=True,render_mode=render_mode)
    
    # Get the Q-table
    q_table = model["qtable"]
    
    # Initialize metrics
    total_epochs = 0
    total_rewards = 0
    total_successes = 0
    
    # Run episodes
    for ep in range(episodes):
        state, info = env.reset()
        epochs, rewards = 0, 0
        done = False
        truncated = False
        
        print(f"\nEpisode {ep+1}/{episodes}")
        
        while not (done or truncated):
            # Take action with highest Q-value
            action = np.argmax(q_table[state])
            
            # Map action to direction for better visualization
            action_mapping = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
            print(f"Taking action: {action_mapping[action]}")
            
            # Apply action and get new state
            new_state, reward, done, truncated, info = env.step(action)
            
            # Update metrics
            epochs += 1
            rewards += reward
            
            # Print step information
            print(f"Step {epochs}, State: {state}, New State: {new_state}, Reward: {reward}")
            
            # Update state
            state = new_state
            
            # Add delay for better visualization
            time.sleep(delay)
        
        # Update episode metrics
        total_epochs += epochs
        total_rewards += rewards
        if reward > 0:  # Typically reward is 1.0 when reaching goal in FrozenLake
            total_successes += 1
        
        print(f"Episode {ep+1} finished: Steps={epochs}, Reward={rewards}")
    
    # Close the environment
    env.close()
    
    # Print final statistics
    print("\n--- Final Results ---")
    print(f"Results after {episodes} episodes:")
    print(f"Average steps per episode: {total_epochs / episodes:.2f}")
    print(f"Average reward per episode: {total_rewards / episodes:.2f}")
    print(f"Success rate: {total_successes / episodes * 100:.2f}%")

# Usage example
if __name__ == "__main__":
    # Replace with your actual repo_id
    repo_id = "yashrajkupekar/FrozenLake-v1-8x8-Slippery"
    
    # Load the model
    model = load_from_hub(repo_id, "q-learning.pkl")
    
    # Test the model with rendering
    test_model_rendering(
        model,
        episodes=5,       # Number of episodes to run
        render_mode="human",  # Render mode (human or rgb_array)
        delay=0.5         # Delay between steps in seconds
    )