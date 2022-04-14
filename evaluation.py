from cmath import inf
import gym
import time
import wandb
from ray import serve
from pydoc import doc
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ars as ars
import ray.rllib.agents.sac as sac
import math
import rtrl
import os
import time
from rtrl import Training, run
from rtrl.wrappers import StatsWrapper
import numpy as np
import omegaconf
import torch
import mbrl.util.env
import mbrl.util.common
import mbrl.planning
import mbrl.env.pets_pusher as pusher
import mbrl.env.humanoid_truncated_obs as humanoid
import mbrl.env.cartpole_continuous as cart
import argparse
from mbrl.planning.core import load_agent
import abc
import pathlib
from typing import Any, Union

import gym
import hydra
import numpy as np
import omegaconf

import mbrl.models
import mbrl.types


# Input arguments from command line.
parser = argparse.ArgumentParser(description='Evaluate trained model')
parser.add_argument("--modeltype", required=True, help="type of model: rtrl mbrl or rllib",
                    default="rllib")
parser.add_argument("--modelpath", required=True, help="Filepath to trained checkpoint",
                    default="/app/data/ray_results/2/ARS_CartPole-v0_661d3_00000_0_2022-03-31_10-07-40/checkpoint_000100/checkpoint-100")
parser.add_argument("--trainseed", required=True, help="Training seed.",
                    default='2')
parser.add_argument("--algorithm", required=True, help="Algorithm used", default="ARS")
parser.add_argument("--envir", required=True, help="Environment.",
                    default='CartPole-v0')
parser.add_argument("--checkpoint", required=False, help="checkpoint to evaluate",
                    default="1")
parser.add_argument("--evaseed", required=True, help="Evaluation seed.",
                    default=1)
args = vars(parser.parse_args())
print("Input of argparse:", args)

def play(env, trainer, times, gap, type, level = 0):
    print('difficulty level:', level)
    prev_action = np.zeros(env.action_space.shape[0])
    total_rewards = []
    iter_ep = 20
    total_ep = level/gap*iter_ep  

    for k in range(iter_ep):
        # print("here")
        obs = env.reset()
        total_reward = 0
        total_ep += 1
        wandb.log({"episode": total_ep, "difficulty_level": level})

        for i in range(times):
            t1 = time.time()
            if type == 'rllib':
                action = trainer.compute_single_action(obs)
            elif type == 'rtrl':
                action = trainer.act((obs,prev_action),[],[],[],train=False)[0]
            elif type == 'mbrl':
                action = trainer.act(obs, deterministic=True)
            # print(action)
            t2 = time.time()
            compute_time = (t2 - t1)
            wandb.log({"computation_time": compute_time})
            compute_times.append(compute_time)
            obs, reward, done, info = env.step(action)
            # print(reward,done)
            repeat = int(level * 1 * compute_time)
            total_reward += reward
            if repeat:
                for j in range(repeat):
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    if done:
                        total_rewards.append(total_reward)  
                        break 
            else:        
                if done:
                    total_rewards.append(total_reward)
                    obs = env.reset()
                    break
            if done:
                break         
    
        env.close()

    #print(total_rewards)
    # print(total_rewards)
    reward_ave = sum(total_rewards)/len(total_rewards) if len(total_rewards) else sum(total_rewards)/(len(total_rewards)+1)
    
    wandb.log({"average_rewards": reward_ave, "difficulty_level": level})
    return reward_ave



wandb.init(project="RTDM", entity="rt_dm")
wconfig = wandb.config
wconfig.model_type = args["modeltype"]
wconfig.algorithm = args["algorithm"]
wconfig.eva_seed = args["evaseed"]
wconfig.train_seed = args["trainseed"]
wconfig.env = args["envir"]
wconfig.checkpoint = args["checkpoint"]

serve.start()

record = []
begin = 0
gap = 500
end = 10000
x = np.arange(begin, end, gap)
seed = int(args["evaseed"])
environment = args["envir"]
compute_times = []

type = args["modeltype"]
if type == 'mbrl':
    if environment == 'pets_pusher':
        env = pusher.PusherEnv()
    elif environment == 'humanoid_truncated_obs':
        env = humanoid.HumanoidTruncatedObsEnv()
    elif environment == 'cartpole_continuous':
        env = cart.CartPoleEnv()
    # trainer = load_agent(args["modelpath"],env)
    agent_path = pathlib.Path(args["modelpath"])
    cfg = omegaconf.OmegaConf.load(agent_path / "config.yaml")

    if cfg.algorithm.agent._target_ == "mbrl.third_party.pytorch_sac_pranz24.sac.SAC":
        import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac

        from .sac_wrapper import SACAgent

        complete_agent_cfg(env, cfg.algorithm.agent)
        agent: pytorch_sac.SAC = hydra.utils.instantiate(cfg.algorithm.agent)
        agent.load_checkpoint(ckpt_path=agent_path / "sac.pth")
        trainer = SACAgent(agent)
    else:
        raise ValueError("Invalid agent configuration.")

elif type == 'rtrl':
    path = args["modelpath"]
    r = rtrl.load(path+"/store")
    trainer = r.agent
    env = gym.make(environment)


elif type == 'rllib':
    algorithm = args["algorithm"]
    trained_model = args["modelpath"]
    env = gym.make(environment)
    if algorithm == "ARS":
        trainer = ars.ARSTrainer(
            config={
                "framework": "torch",
                # "num_workers": 4,
            },
            env=environment,
        )
    elif algorithm == "PPO":
        trainer = ppo.PPOTrainer(
            config={
                "framework": "torch",
                # "num_workers": 4,
            },
            env=environment,
        )
    elif algorithm == "SAC":
        trainer = sac.SACTrainer(
            config={
                "framework": "torch",
                # "num_workers": 4,
            },
            env=environment,
        )
    trainer.restore(trained_model)


env.seed(seed)
reward_ave = play(env, trainer, 100000, gap = gap, type = type)
record.append(reward_ave)
for level in x[1:]:
    reward_ave = play(env, trainer, 100000, gap = gap, type = type, level = level)
    record.append(reward_ave)
time_ave = sum(compute_times)/len(compute_times)
wandb.log({'average_compute_time':time_ave})