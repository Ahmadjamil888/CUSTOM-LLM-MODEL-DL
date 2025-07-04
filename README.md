
# CUSTOM GPT-LIKE LLM WITH REINFORCEMENT LEARNING

A custom-built GPT-style language model developed from scratch using PyTorch. This project includes a training loop with reinforcement learning from human feedback (RLHF), a Flask-based web interface for interaction, and uses a small conversational dataset for training and testing.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Setup Guide](#setup-guide)
- [Dataset Loading](#dataset-loading)
- [Model Training](#model-training)
- [Running the Chatbot](#running-the-chatbot)
- [Project Structure](#project-structure)
<img src="https://raw.githubusercontent.com/Ahmadjamil888/CUSTOM-LLM-MODEL-DL/refs/heads/main/Screenshot%202025-07-04%20163634.png">
---

## Project Overview

This project demonstrates the full lifecycle of building a custom large language model (LLM) using PyTorch and training it on dialogue data with reinforcement learning. A user-friendly interface is provided using Flask to simulate a ChatGPT-like experience.

---

## Requirements

- Python 3.10 or higher
- pip
- git

---

## Setup Guide

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Ahmadjamil888/CUSTOM-LLM-MODEL-DL.git
    cd CUSTOM-LLM-MODEL-DL
    ```

2. **Create and activate virtual environment:**

    ```bash
    python -m venv venv
    venv\Scripts\activate   # On Windows
    # Or on macOS/Linux:
    # source venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

---

## Dataset Loading

This project uses the `daily_dialog` dataset from Hugging Face. It is small (~45MB) and appropriate for initial experiments.

The dataset will be automatically downloaded during training. If required manually:

```python
from datasets import load_dataset
dataset = load_dataset("daily_dialog", split="train[:1000]")
```

---

## Model Training

To train the model using the `daily_dialog` dataset:

```bash
cd model
python train.py
```

This trains the model and saves it to `trained_model/gpt_custom.pth`.

---

## Running the Chatbot

To start the Flask chatbot web interface:

```bash
cd ..
python run.py
```

Then, open your browser and go to:  
`http://127.0.0.1:5000`

---

## Project Structure

```
youtube-ai-recommender/
│
├── app/
│   ├── static/
│   ├── templates/
│   ├── __init__.py
│   ├── routes.py
│   ├── utils.py
│   └── chat_history.json
│
├── model/
│   ├── model.py
│   ├── train.py
│   ├── feedback_env.py
│   ├── tokenizer.py
│   ├── ppo_trainer.py
│   └── reward_model.py
│
├── datasets/
│   └── (optional .txt data)
│
├── requirements.txt
├── run.py
└── README.md
```

---

## License

This project is licensed under the MIT License.
