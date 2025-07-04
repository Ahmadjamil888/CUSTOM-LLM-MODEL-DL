import gym
from gym import spaces
import random

class ChatFeedbackEnv(gym.Env):
    """
    Custom Gym environment for chat-based reinforcement learning.

    States: Natural language prompts
    Actions: Model-generated responses (strings)
    Rewards: Simulated or real feedback (e.g., 0–1 scale)
    """

    def __init__(self, prompts=None):
        super(ChatFeedbackEnv, self).__init__()

        self.prompts = prompts or [
            "What is AI?",
            "Explain gravity.",
            "How to bake a cake?",
            "What is quantum computing?",
            "Tell me a joke.",
        ]

        self.max_steps = len(self.prompts)
        self.step_count = 0
        self.current_prompt = None

        # Observation is a text prompt (string)
        # Action is a text response (string)
        self.observation_space = spaces.Discrete(self.max_steps)  # For simplicity
        self.action_space = spaces.Discrete(1)  # Dummy, actual action is handled as text externally

    def reset(self):
        self.step_count = 0
        self.current_prompt = self.prompts[0]
        return self.current_prompt

    def step(self, action_text):
        """
        Args:
            action_text (str): AI-generated response to the prompt
        Returns:
            next_prompt (str), reward (float), done (bool), info (dict)
        """
        reward = self._simulate_reward(self.current_prompt, action_text)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "prompt": self.current_prompt,
            "response": action_text,
            "reward": reward
        }

        self.current_prompt = self.prompts[self.step_count % len(self.prompts)] if not done else None
        return self.current_prompt, reward, done, info

    def _simulate_reward(self, prompt, response):
        """
        Placeholder: Simulate feedback based on response quality.
        Replace with real reward model in production.
        """
        length = len(response.strip())
        if "?" in response or length < 10:
            return 0.1
        elif "because" in response or length > 50:
            return 0.9
        else:
            return 0.5
