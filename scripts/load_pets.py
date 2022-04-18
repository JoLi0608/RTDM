import omegaconf
import torch
import mbrl.util.env
import mbrl.util.common
import mbrl.planning
import numpy as np 

path = "/app/data/pets/HalfCheetah-v2/102236/"

cfg = omegaconf.OmegaConf.load(path+".hydra/config.yaml")
#cfg["device"] = "cpu"
torch_generator = torch.Generator(device=cfg.device)

env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
obs_shape = env.observation_space.shape
act_shape = env.action_space.shape
dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
dynamics_model.load(path)
model_env = mbrl.models.ModelEnv(env, dynamics_model, term_fn, reward_fn)
agent = mbrl.planning.create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles)
action = agent.act(env.reset())
action = np.clip(action, -1.0, 1.0)  # to account for the noise

