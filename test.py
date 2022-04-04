# import ray
# import ray.rllib.agents.ppo as ppo
# from ray import serve

# def train_ppo_model():
#     trainer = ppo.PPOTrainer(
#         config={"framework": "torch", "num_workers": 0},
#         env="CartPole-v0",
#     )
#     # Train for one iteration
#     trainer.train()
#     trainer.save("/Users/liwenyu/Downloads/ray_results/PPO_Hopper-v2_0c9e7_00000_0_2022-03-30_13-26-47/checkpoint_001400/checkpoint-1400")
#     return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"


# checkpoint_path = train_ppo_model()
import numpy
print(numpy.version.version)