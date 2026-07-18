from src.training.reward_model import RewardModel

reward = RewardModel()

responses = [
    "Paris is the capital of France.",
    "I don't know.",
    "The answer is 42."
]

scores = reward.score(responses)

print(scores)