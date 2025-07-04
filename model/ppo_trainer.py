import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from model.reward_model import RewardModel
from model.feedback_env import ChatFeedbackEnv
from model.inference import generate_response  # or your model directly

class PPOFeedbackEnv(ChatFeedbackEnv):
    """
    Extend your original ChatFeedbackEnv to compute reward using RewardModel.
    """
    def __init__(self, prompts, reward_model):
        super().__init__(prompts)
        self.reward_model = reward_model

    def step(self, action_text):
        # Compute reward using the reward model
        reward = self.reward_model.compute_reward(self.current_prompt, action_text)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "prompt": self.current_prompt,
            "response": action_text,
            "reward": reward
        }

        self.current_prompt = self.prompts[self.step_count % len(self.prompts)] if not done else None
        return self.current_prompt, reward, done, info

def main():
    print("🚀 Initializing PPO RLHF Training")

    # Sample prompts
    prompts = [
        "What is AI?",
        "Explain quantum computing.",
        "Tell me a fun fact.",
        "How do I bake a cake?",
        "What is the future of technology?"
    ]

    # Load reward model
    reward_model = RewardModel()
    reward_model.eval()

    # Create environment
    env = PPOFeedbackEnv(prompts, reward_model)
    check_env(env, warn=True)

    # Train a PPO agent (dummy MLP policy to start)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Save PPO policy
    model.save("ppo_rlhf_agent")
    print("✅ PPO Agent Trained and Saved")

if __name__ == "__main__":
    main()
