from spinup.utils.test_policy import load_policy_and_env, run_policy
import time
import ray.rllib.agents.ars as ars
from mbrl.planning.core import load_agent
import gym

import omegaconf
import torch
import mbrl.util.env
import mbrl.util.common
import mbrl.planning
import numpy as np
import rtrl

def run_env(agent,env,num_steps=1000,conc_prev=False):
    prev_action = np.zeros(env.action_space.shape[0])
    t1 = time.time()
    obs = env.reset()
    for i in range(num_steps):
        if conc_prev:
            obs = (obs,prev_action)
        action = agent(obs)  # Get action
        obs,_ ,_ , _ = env.step(action)
        prev_action = action
    return (time.time()-t1)/1000

def load_agent(algo="PPO",path=,device="cpu"):

print("Benchmark mbpo")


env = gym.make("Hopper-v2")
ag = load_agent("/app/data/mbpo/default/gym___Hopper-v2/2022.04.01/100721", env,"cpu")

agent = lambda obs: ag.act(obs)

t = run_env(agent,env)
print("Average CPU inference time: ",t)

ag = load_agent("/app/data/mbpo/default/gym___Hopper-v2/2022.04.01/100721", env,"cuda")

agent = lambda obs: ag.act(obs)

t = run_env(agent,env)
print("Average GPU inference time: ",t)

print("Benchmark pets")

path = "/app/data/pets/Hopper-v2/091637/"

cfg = omegaconf.OmegaConf.load(path+".hydra/config.yaml")
#cfg["device"] = "cpu"
torch_generator = torch.Generator(device=cfg.device)

env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
obs_shape = env.observation_space.shape
act_shape = env.action_space.shape
dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
dynamics_model.load(path)
model_env = mbrl.models.ModelEnv(env, dynamics_model, term_fn, reward_fn)
ag = mbrl.planning.create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles)
action = ag.act(env.reset())
action = np.clip(action, -1.0, 1.0)  # to account for the noise
agent = lambda obs: np.clip(ag.act(obs),-1.0,1.0)

t = run_env(agent,env,num_steps=20)
print("Average GPU inference time: ",t)



cfg = omegaconf.OmegaConf.load(path+".hydra/config.yaml")
cfg["device"] = "cpu"
torch_generator = torch.Generator(device=cfg.device)

env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
obs_shape = env.observation_space.shape
act_shape = env.action_space.shape
dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
dynamics_model.load(path)
dynamics_model.to("cpu")
model_env = mbrl.models.ModelEnv(env, dynamics_model, term_fn, reward_fn)
ag = mbrl.planning.create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles)
action = ag.act(env.reset())
action = np.clip(action, -1.0, 1.0)  # to account for the noise
agent = lambda obs: np.clip(ag.act(obs),-1.0,1.0)

t = run_env(agent,env,num_steps=20)
print("Average CPU inference time: ",t)



print("Benchmark rllib")
path = "/app/data/ray_results/1/ARS_Hopper-v2_47175_00000_0_2022-04-07_11-24-24/checkpoint_015300/checkpoint-15300"
trainer = ars.ARSTrainer(
    config={
        "framework": "torch",
    },
    env="Hopper-v2",
)
trainer.restore(path)
agent = lambda obs: trainer.compute_single_action(obs)
env = gym.make("HalfCheetah-v2")
t = run_env(agent,env)
print("Average CPU inference time: ",t)


trainer = ars.ARSTrainer(
    config={
        "framework": "torch",
        "num_gpus":0.2,
    },
    env="HalfCheetah-v2",
)
trainer.restore(path)
agent = lambda obs: trainer.compute_single_action(obs)
env = gym.make("HalfCheetah-v2")
t = run_env(agent,env)
print("Average GPU inference time: ",t)


print("Benchmark RTRL --------")
path = "/app/data/rtrl_2/exp-1-Hopper-v2-RTAC/"
r = rtrl.load(path+"state")
r.agent.model.to("cpu")
agent = lambda obs: r.agent.act(obs,[],[],[],train=False)[0]
env = gym.make("Hopper-v2")

t = run_env(agent,env,conc_prev=True)
print("Average CPU inference time: ",t)

r.agent.model.to("cuda")
t = run_env(agent,env,conc_prev=True)
print("Average GPU inference time: ",t)


# Start with spinup
print("Benchmark spinup -------")
env,policy = load_policy_and_env("/app/data/Hopper-v2/cmd_sac_pytorch/cmd_sac_pytorch_s1",device="cpu")
t = run_env(policy,env)
print("Average CPU inference time: ",t)

env, policy = load_policy_and_env("/app/data/Hopper-v2/cmd_sac_pytorch/cmd_sac_pytorch_s1", device="gpu")
t = run_env(policy,env)
print("Average GPU inference time: ",t)



