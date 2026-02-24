# PyTorch Learning Lab: From Deep RL to Embodied AI

This repository is a comprehensive, from-scratch engineering portfolio bridging **Deep Reinforcement Learning (DRL)** and **Foundation Models**. Built with a focus on **Embodied AI** and physical simulation, it demonstrates the translation of theoretical ML algorithms into robust, research-ready PyTorch code. 

Key highlights include implementing core Transformer attention mechanisms, building local RAG systems, and training advanced continuous-control PPO agents in MuJoCo physics environments (featuring GAE, online observation normalization, and Domain Randomization for sim-to-real transfer).

---

## 📋 Table of Contents

- [Project 1: Q-Learning Maze Solver](#-project-1-q-learning-maze-solver)
- [Project 2: Transformer Basics](#-project-2-transformer-basics)
- [Project 3: PPO Reacher Agent](#-project-3-ppo-reacher-agent)
- [Project 4: RAG Assistant](#-project-4-rag-assistant)
- [Project 5: PPO Ant Walker (MuJoCo)](#-project-5-ppo-ant-walker-mujoco)
- [Project Structure](#%EF%B8%8F-project-structure)
- [Setup & Installation](#%EF%B8%8F-setup--installation)
- [References](#-references)
- [License](#-license)

---

## 🏗️ Project Structure

```
pytorch-learning-lab/
├── README.md
├── requirements.txt
├── .gitignore
│
├── 01_q_learning_maze/
│   ├── maze_env.py              # SimpleMaze environment (4×4 grid world)
│   ├── q_learning_agent.py      # Tabular Q-Learning agent
│   ├── train.py                 # Training loop + reward curve + agent GIF
│   └── results/
│       ├── training_reward_curve.png
│       └── maze_agent.gif
│
├── 02_transformer_basics/
│   ├── transformer_block.py      # SimpleSelfAttention + TransformerBlock
│   ├── sentiment_analysis.py     # Sentiment analysis with Hugging Face
│   └── text_generation.py        # Text generation with GPT-2
│
├── 03_ppo_reacher/
│   ├── networks.py             # PolicyNetwork (actor) and ValueNetwork (critic)
│   ├── ppo.py                  # PPO algorithm: rollout, GAE, PPO-Clip update
│   ├── train.py                # Training loop with argparse and result saving
│   ├── evaluate.py             # Model evaluation and GIF recording
│   ├── explore_env.py          # Environment exploration and reward analysis
│   └── results/
│       ├── trained_model.pth
│       ├── training_reward_curve.png
│       └── trained_agent.gif
│
├── 04rag-assistant/
│   ├── main.py                # Entry point - run full RAG pipeline
│   ├── README.md
│   ├── data/
│   │   └── sample.txt         # Zombie survival guide (knowledge base)
│   └── src/
│       ├── step1_load.py      # Document loading
│       ├── step2_split.py     # Text splitting
│       ├── step3_embed.py     # Embedding & vector storage
│       ├── step4_retrieve.py  # Similarity search retrieval
│       └── step5_generate.py  # LLM answer generation
│
├── 05_mujoco_ant/
│   ├── networks.py            # ActorCritic network (actor + critic heads)
│   ├── ppo_buffer.py          # PPO rollout buffer with GAE
│   ├── ppo_agent.py           # PPO agent (select_action, update)
│   ├── obs_normalizer.py      # Running observation normalizer (Welford)
│   ├── reward_normalizer.py   # Running reward normalizer
│   ├── domain_random.py       # Domain randomization for sim-to-real
│   ├── train.py               # Training loop with checkpoint & resume
│   ├── record.py              # Record trained agent as MP4 video
│   ├── plot_training_curve.py # Plot training reward curve from log
│   ├── training_log.txt       # Training output log
│   └── results/
│       ├── training_reward_curve.png
│       └── ant_walking.gif
│
└── assets/
```

---

## 🐭 Project 1: Q-Learning Maze Solver

### Overview

A Q-Learning agent that learns to navigate a 4×4 maze to find the cheese.  
The agent starts at the top-left corner **(0, 0)** and must reach the goal at **(3, 3)**, avoiding two walls.

```
┌───┬───┬───┬───┐
│ S │   │   │   │     S = Start (0, 0)
├───┼───┼───┼───┤
│   │ X │   │   │     X = Wall (impassable)
├───┼───┼───┼───┤
│   │   │ X │   │     G = Goal / Cheese (3, 3)
├───┼───┼───┼───┤
│   │   │   │ G │
└───┴───┴───┴───┘
```

### Key Concepts

| Concept | Description |
|---|---|
| **Q-Table** | Dictionary mapping `(state, action) → expected reward` |
| **Bellman Equation** | `Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]` |
| **Epsilon-Greedy** | Explore randomly with probability ε, exploit best action otherwise |
| **Epsilon Decay** | ε shrinks each episode so the agent exploits more as it learns |

### Results

| Training Reward Curve | Trained Agent Navigation |
|:---:|:---:|
| ![Reward Curve](01_q_learning_maze/results/training_reward_curve.png) | ![Agent GIF](01_q_learning_maze/results/maze_agent.gif) |

### How to Run

```bash
cd 01_q_learning_maze
python train.py
```

Output files are saved to `01_q_learning_maze/results/`.

---

## 🤖 Project 2: Transformer Basics

### Overview

Exploring the building blocks of modern large language models using PyTorch and Hugging Face.  
Includes a from-scratch self-attention implementation as well as pre-trained model inference.

### Key Concepts

| Concept | File |
|---|---|
| **Scaled Dot-Product Self-Attention** | `transformer_block.py` — `SimpleSelfAttention` |
| **Full Transformer Encoder Block** | `transformer_block.py` — `TransformerBlock` |
| **Sentiment Analysis** | `sentiment_analysis.py` — DistilBERT via HF pipeline |
| **Text Generation (GPT-2)** | `text_generation.py` — GPT-2 via HF pipeline |

### Self-Attention at a Glance

```
Q = X W_Q
K = X W_K                      (B, T, D)
V = X W_V
                                   ↓
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

### How to Run

```bash
cd 02_transformer_basics

# Test the from-scratch Transformer block
python transformer_block.py

# Run sentiment analysis (downloads DistilBERT on first run)
python sentiment_analysis.py

# Run text generation (downloads GPT-2 on first run)
python text_generation.py
```

---

## 🤖 Project 3: PPO Reacher Agent

### Overview

A PPO agent trained from scratch using PyTorch to control a 2-DOF robotic arm in the MuJoCo Reacher-v5 environment. The agent learns to coordinate two joint torques to move its fingertip to a randomly placed target.

### Key Concepts

| Concept | Description |
|---|---|
| **Proximal Policy Optimization (PPO-Clip)** | Clipped surrogate objective for stable on-policy updates |
| **Generalized Advantage Estimation (GAE)** | Bias–variance trade-off in advantage computation |
| **Continuous action space** | Gaussian policy over joint torques |
| **Actor-Critic architecture** | Separate policy and value networks |

### Results

| Training Reward Curve | Trained Agent Demo |
|:---:|:---:|
| ![Reward Curve](03_ppo_reacher/results/training_reward_curve.png) | ![Agent GIF](03_ppo_reacher/results/trained_agent.gif) |

### How to Run

```bash
cd 03_ppo_reacher

# Explore the environment
python explore_env.py

# Train the agent
python train.py --num_iterations 200 --steps_per_iter 2048

# Evaluate and save a GIF
python evaluate.py --model_path results/trained_model.pth --save_gif
```

---

## 🧟 Project 4: RAG Assistant

### Overview

A Retrieval-Augmented Generation (RAG) system built from scratch using LangChain. It reads a zombie survival guide, splits it into chunks, embeds them into a vector database, and uses a local LLM (TinyLlama-1.1B) to answer questions based on retrieved context - all running locally without any API keys.

### Key Concepts

| Concept | Description |
|---|---|
| **Retrieval-Augmented Generation (RAG)** | Combine document retrieval with LLM generation for grounded answers |
| **Text Embeddings** | Convert text into numerical vectors that capture semantic meaning |
| **Vector Store (ChromaDB)** | Database that enables fast similarity search over embedded documents |
| **Prompt Engineering** | Structured instructions that guide LLM output quality and format |
| **Local LLM Inference** | Run a language model (TinyLlama-1.1B) entirely on-device without API calls |

### Architecture

```
Document ---> Split ---> Embed ---> Store (ChromaDB)
                     |
User Query ---> Embed ---> Similarity Search ---> Retrieve Top-K Chunks
                     |
Prompt + Context ---> LLM ---> Answer
```

### Sample Output

```text
=== RAG Assistant: Zombie Survival Guide ===
📥 Loading documents...
✅ 1 document loaded
✂️ Splitting into chunks...
✅ 20 chunks created
🧠 Creating vector store...
✅ 20 vectors stored
🤖 Loading LLM...
✅ TinyLlama ready

--- Query 1/4 ---
Q: What is the best weapon for close combat?
A: The best weapon for close combat is a sturdy crowbar. It is quiet, nearly unbreakable, and provides excellent balance and control for striking and prying.

--- Query 2/4 ---
Q: How do I purify water?
A: Purify by boiling for at least one full minute, filtering through a proper field filter, or using purification tablets exactly as directed.

✅ Done! Total execution time: 51.8s
```

### How to Run

```bash
# Create and activate conda environment
conda create -n rag_env python=3.11
conda activate rag_env

# Install dependencies
pip install langchain langchain-community langchain-core langchain-text-splitters \
  langchain-huggingface chromadb sentence-transformers transformers torch accelerate pypdf

# Run the full pipeline
python 04rag-assistant/main.py
```

> **Note:** First run downloads TinyLlama (~2 GB) and the embedding model (~90 MB). Subsequent runs use cached models.

---

## 🐜 Project 5: PPO Ant Walker (MuJoCo)

### Overview

A PPO agent trained from scratch to control a quadruped ant (Ant-v5) in MuJoCo. The agent learns to coordinate 8 joint torques to walk forward, achieving a peak reward of +632 after 5M steps of training. Includes observation normalization, reward normalization, learning rate annealing, and domain randomization for sim-to-real transfer.

### Key Concepts

| Concept | Description |
|---|---|
| **Proximal Policy Optimization (PPO-Clip)** | Clipped surrogate objective for stable on-policy updates |
| **Actor-Critic Architecture** | Shared observation space with separate actor and critic heads |
| **Generalized Advantage Estimation (GAE)** | Bias–variance trade-off in advantage computation |
| **Observation Normalization** | Welford's online algorithm for running mean/variance of 105-dim observations |
| **Reward Normalization** | Running variance scaling to prevent large penalties from dominating learning |
| **Learning Rate Annealing** | Linear decay from 3e-4 to 0 for stable late-stage training |
| **Domain Randomization** | Randomize gravity, friction, and body mass for sim-to-real transfer |
| **Checkpoint & Resume** | Save/load full training state (network, optimizer, normalizers, progress) |

### Results

| Training Reward Curve | Trained Agent Demo |
|:---:|:---:|
| ![Reward Curve](05_mujoco_ant/results/training_reward_curve.png) | ![Agent GIF](05_mujoco_ant/results/ant_walking.gif) |

### How to Run

```bash
cd 05_mujoco_ant

# Train the agent (5M steps, ~38 min)
python train.py --timesteps 5000000

# Resume from checkpoint
python train.py --timesteps 5000000 --resume checkpoints/ant_ppo_final.pt

# Record the trained agent
python record.py --checkpoint checkpoints/ant_ppo_4096000.pt --episodes 1

# Plot training curve
python plot_training_curve.py --log training_log.txt
```

---

## 🛠️ Setup & Installation

**Prerequisites:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/pytorch-learning-lab.git
cd pytorch-learning-lab

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `torch` installation varies by platform. Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the correct command for your CUDA version.

---

## 📚 References

- Sutton & Barto — [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2018)
- Vaswani et al. — [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Schulman et al. — [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) (2017)
- Schulman et al. — [*High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438) (2016)
- Tobin et al. — [*Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*](https://arxiv.org/abs/1703.06907) (2017)
- Lewis et al. — [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401) (2020)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [LangChain Documentation](https://python.langchain.com/docs/introduction/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Hugging Face sentence-transformers](https://www.sbert.net/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Gymnasium Reacher-v5 Documentation](https://gymnasium.farama.org/environments/mujoco/reacher/)
- [Gymnasium Ant-v5 Documentation](https://gymnasium.farama.org/environments/mujoco/ant/)

---

## 📝 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
Feel free to use, modify, and distribute for educational purposes.
