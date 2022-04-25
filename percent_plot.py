from cmath import inf
import gym
import time
import wandb
# from ray import serve
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
from spinup.utils.test_policy import load_policy_and_env, run_policy


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

def play(env, trainer, times, flag, gap, type, algorithm, level = 0):
    initial_reward = 0
    percent = 0
    total_rewards = []
    if algorithm == 'pets':
        iter_ep = 5
    else:
        iter_ep = 20
    total_ep = level/gap*iter_ep  
    if type == 'rtrl':
        prev_action = np.zeros(env.action_space.shape[0])

    for repeat in range(11):
        print('action repeated:', repeat)
        
        for k in range(iter_ep):

            obs = env.reset()
            total_reward = 0
            total_ep += 1
            # wandb.log({"episode": total_ep, "difficulty_level": level})

            for i in range(times):

                if type == 'rllib':
                    action = trainer.compute_single_action(obs)
                elif type == 'rtrl':
                    action = trainer.act((obs,prev_action),[],[],[],train=False)[0]
                elif type == 'mbrl':
                    action = trainer.act(obs, deterministic=True)
                    if algorithm == 'pets':
                        action = np.clip(action, -1.0, 1.0) 
                elif type == 'spinup':
                    action = trainer(obs)
                obs, reward, done, info = env.step(action)
                print(1,done, reward)
                rest = repeat-1
                total_reward += reward
                if rest:
                    for j in range(rest):
                        obs, reward, done, info = env.step(action)
                        print(2,done, reward)
                        total_reward += reward
                        if done:
                            total_rewards.append(total_reward)  
                            # print(total_reward)
                            break 
                else:        
                    if done:
                        total_rewards.append(total_reward)
                        print(3,done, reward)
                        # print(total_reward)
                        obs = env.reset()
                        break
                if done:
                    break 
            env.close()
        print(total_rewards)    
        reward_ave = sum(total_rewards)/len(total_rewards) if len(total_rewards) else sum(total_rewards)/(len(total_rewards)+1)
        if repeat == 0:
            # print('here')
            # print(reward_ave)
            initial_reward = reward_ave
        print(initial_reward)
        percent = reward_ave/initial_reward
        wandb.log({"percent": percent, "action_repeated": repeat})



    
        env.close()


    return 

type = args["modeltype"]
seed = int(args["evaseed"])
environment = args["envir"]
path = args["modelpath"]
algorithm = args["algorithm"]

wandb.init(project="RTDM", entity="rt_dm")
wconfig = wandb.config
wconfig.model_type = type
wconfig.algorithm = algorithm
wconfig.eva_seed = args["evaseed"]
wconfig.train_seed = args["trainseed"]
wconfig.env = environment
wconfig.checkpoint = args["checkpoint"]

# serve.start()

record = []
begin = 0
gap = 500
end = 10000
x = np.arange(begin, end, gap)
compute_times = []

if environment == 'pets_pusher':
    env = pusher.PusherEnv()
elif environment == 'humanoid_truncated_obs':
    env = humanoid.HumanoidTruncatedObsEnv()
elif environment == 'cartpole_continuous':
    env = cart.CartPoleEnv()
else:
    env = gym.make(environment)

if type == 'mbrl':
    if algorithm == 'mbpo':
        trainer = load_agent(path,env,"cuda")
    elif algorithm == 'pets':
        cfg = omegaconf.OmegaConf.load(path+"/.hydra/config.yaml")
        #cfg["device"] = "cpu"
        torch_generator = torch.Generator(device=cfg.device)
        env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
        dynamics_model.load(path)
        model_env = mbrl.models.ModelEnv(env, dynamics_model, term_fn, reward_fn)
        trainer = mbrl.planning.create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles)




elif type == 'rtrl':
    r = rtrl.load(path+"/store")
    trainer = r.agent


elif type == 'rllib':
    algorithm = args["algorithm"]
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
    trainer.restore(path)

elif type == 'spinup':
    env,trainer = load_policy_and_env(path,device="cpu")

flag = 0
times = 100000

if environment == 'Pusher-v2' or environment == 'pets_pusher':
    flag = 1
    times = 100


env.seed(seed)
play(env, trainer, times, flag, gap = gap, type = type, algorithm = algorithm)
# time_ave = sum(compute_times)/len(compute_times)
# wandb.log({'average_compute_time':time_ave})