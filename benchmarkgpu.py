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

def load(algo,env_name="Hopper-v2",gpu=False):
    if algo == "mbpo":
        env = gym.make(env_name)
        device = "cuda" if gpu else "cpu"
        ag = load_agent("/app/data/mbpo/default/Hopper-v2/2022.04.01/034518", env,device)
        agent = lambda obs: ag.act(obs)
    elif algo == "pets":
        path = "/app/data/pets/Hopper-v2/091637/"
        cfg = omegaconf.OmegaConf.load(path+".hydra/config.yaml")
        if not gpu:
            cfg["device"] = "cpu"
        torch_generator = torch.Generator(device=cfg.device)
        env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
        dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, env.observation_space.shape, env.action_space.shape)
        dynamics_model.load(path)
        if gpu == False:
            dynamics_model.to("cpu")
        model_env = mbrl.models.ModelEnv(env, dynamics_model, term_fn, reward_fn)
        ag = mbrl.planning.create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles)
        action = ag.act(env.reset())
        action = np.clip(action, -1.0, 1.0)  # to account for the noise
        agent = lambda obs: np.clip(ag.act(obs),-1.0,1.0)
    elif algo == "rtrl":
        path = "/app/data/rtrl_2/exp-1-Hopper-v2-RTAC/"
        r = rtrl.load(path+"state")
        if not gpu:
            r.agent.model.to("cpu")
        else:
            r.agent.model.to("cuda")
        agent = lambda obs: r.agent.act(obs,[],[],[],train=False)[0]
        env = gym.make(env_name)

    elif algo == "sac":
        device = "cuda" if gpu else "cpu"
        env,agent = load_policy_and_env("/app/data/spinup/sac/Hopper-v2/cmd_sac_pytorch/cmd_sac_pytorch_s1",device=device)
    elif algo == "ppo":
        device = "cuda" if gpu else "cpu"
        env,agent = load_policy_and_env("/app/data/spinup/ppo/Hopper-v2/cmd_ppo_pytorch/cmd_ppo_pytorch_s1",device=device)
    else:
        print("Algo not known", algo)
    return agent,env
    
def run_env(agent,env,num_steps=100,conc_prev=False):
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


inf_time = {}
for gpu in [False,True]:
    for algo in ["mbpo","pets","rtrl","sac","ppo"]:
        agent,env = load(algo,gpu=gpu)
        print("Done loading")
        if algo == "rtrl":
            inf_time[algo+"_"+str(gpu)] = run_env(agent,env,conc_prev=True)
        else:   
            inf_time[algo+"_"+str(gpu)] = run_env(agent,env)
            
print(inf_time)

        
#elif algo == "ars":
#     path = "/app/data/ray_results/1/ARS_Hopper-v2_47175_00000_0_2022-04-07_11-24-24/checkpoint_015300/checkpoint-15300"
#     num_gpus = 0.2 if gpu else 0
#     trainer = ars.ARSTrainer(config={
#             "framework": "torch",
#             "num_gpus": num_gpus,},env="Hopper-v2")
#     trainer.restore(path)
#     agent = lambda obs: trainer.compute_single_action(obs)
#     env = gym.make("Hopper-v2")
