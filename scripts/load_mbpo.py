from mbrl.planning.core import load_agent
import gym

env = gym.make("Hopper-v2")
agent = load_agent("/app/data/mbpo/default/gym___Hopper-v2/2022.04.01/100721",env)

agent.act(env.reset())

# import omegaconf
# from sklearn import gaussian_process
# import torch
# import mbrl.util.env
# import mbrl.util.common
# import mbrl.planning
# import gym
# import time
# import numpy
# # from numpy import average
# # from sympy import total_degree
# import wandb
# import argparse
# #path = "exp/pets/default/cartpole_continuous/2022.02.21/134508/"
# # path = "exp/pets/default/cartpole_continuous/2022.02.21/134508/"
# parser = argparse.ArgumentParser(description='Evaluate trained model')
# parser.add_argument("--modelpath", required=True, help="Filepath to trained checkpoint",
#                     default="/app/data/mbpo/default/gym___Hopper-v2/2022.04.01/100715/")
# # parser.add_argument("--trainseed", required=True, help="Training seed.",
# #                     default='2')
# parser.add_argument("--algorithm", required=True, help="Algorithm used", default="mbpo")
# parser.add_argument("--gymenv", required=True, help="Environment.",
#                     default='Hopper-v2')
# # parser.add_argument("--checkpoint", required=True, help="checkpoint to evaluate",
# #                     default="1")
# parser.add_argument("--evaseed", required=True, help="Evaluation seed.",
#                     default=1)
# args = vars(parser.parse_args())
# print("Input of argparse:", args)
# # cfg = omegaconf.OmegaConf.load(path+".hydra/config.yaml")
# path = args["modelpath"]

# cfg = omegaconf.OmegaConf.load(path+".hydra/config.yaml")
# cfg["device"] = "cpu"
# torch_generator = torch.Generator(device=cfg.device)

# env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
# obs_shape = env.observation_space.shape
# act_shape = env.action_space.shape
# dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
# dynamics_model.load(path)
# model_env = mbrl.models.ModelEnv(env, dynamics_model, term_fn, reward_fn)
# agent = mbrl.planning.create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles)



