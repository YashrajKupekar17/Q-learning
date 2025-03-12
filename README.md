# **Q-learning Implementation** 

This repository contains a from-scratch implementation of **Q-learning**, a fundamental **reinforcement learning (RL)** algorithm. The implementation is tested in the **FrozenLake** and **Taxi Problem** environments provided by OpenAI's Gym.  

##  **Overview**  
Q-learning is a model-free RL algorithm that enables an agent to learn optimal policies through trial and error. The goal is to maximize cumulative rewards by updating Q-values using the Bellman equation.  

This project demonstrates Q-learning in two environments:  
 **FrozenLake (8×8, Slippery)** – A grid-world navigation problem.  
 **Custom Taxi Environment** – A modified version of the classic Taxi-v3 problem.  

---

## **Colab Notebooks**  
You can explore the implementations interactively on Google Colab:  

| Environment | Colab Link |
|------------|------------|
| **Q-learning Implementation** | [![Colab](https://img.shields.io/badge/Open%20in-Colab-blue?logo=googlecolab)](https://colab.research.google.com/drive/13R5u03HAqNYZwpoAEomUag5-tNZAoFdu?usp=sharing) |
| **FrozenLake (8×8, Slippery)** | [![Colab](https://img.shields.io/badge/Open%20in-Colab-blue?logo=googlecolab)](https://colab.research.google.com/drive/16rgXpTv3PSTfuf9MfajgscBNKsLLecXp?usp=sharing) |
| **Custom Taxi Environment** | [![Colab](https://img.shields.io/badge/Open%20in-Colab-blue?logo=googlecolab)](https://colab.research.google.com/drive/11jncjjDSmsWRLMWCaaA6OdAYnwqWUPB6?usp=sharing) |

---

##  **Trained Models**  
The trained Q-learning models are available on **Hugging Face** for evaluation.  

| Model | Hugging Face Link | Preview |
|-------|------------------|---------|
| **FrozenLake (4×4, Non-Slippery)** | [🔗 Model](https://huggingface.co/yashrajkupekar/q-FrozenLake-v1-4x4-noSlippery) | ![FrozenLake_4X4_Non-slippery](Testing/Videos/ezgif.com-video-to-gif-converter-2.gif) |
| **FrozenLake (8×8, Slippery)** | [🔗 Model](https://huggingface.co/yashrajkupekar/FrozenLake-v1-8x8-Slippery) | ![FrozenLake_8X8_slippery](Testing/Videos/FrozenLake_8X8_slippery-ezgif.com-video-to-gif-converter.gif) |
| **Taxi (500 states)** | [🔗 Model](https://huggingface.co/yashrajkupekar/Taxi_500states) | ![Taxi_500states)](Testing/Videos/ezgif.com-video-to-gif-converter.gif) |
| **Taxi (7200 states)** | [🔗 Model](https://huggingface.co/yashrajkupekar/Taxi_7200states) |![Taxi_500states)](Testing/Videos/Custom_taxi-ezgif.com-video-to-gif-converter.gif)  |


---
##  **References**  
- [Q-learning Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Q-learning)  
- [OpenAI Gym Documentation](https://www.gymlibrary.dev/)  

---
