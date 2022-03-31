
import omegaconf
import torch
import mbrl.util.env
import mbrl.util.common
import mbrl.planning

import gym
import time
from numpy import average
from stable_baselines3 import A2C
from sympy import total_degree
import wandb

#path = "exp/pets/default/cartpole_continuous/2022.02.21/134508/"
# path = "exp/pets/default/cartpole_continuous/2022.02.21/134508/"

# cfg = omegaconf.OmegaConf.load(path+".hydra/config.yaml")
path = "exp/pets/default/gym___Hopper-v2/2022.02.21/134508/"

cfg = omegaconf.OmegaConf.load(path+".hydra/config.yaml")
cfg["device"] = "cpu"
torch_generator = torch.Generator(device=cfg.device)

env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
obs_shape = env.observation_space.shape
act_shape = env.action_space.shape
dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
dynamics_model.load(path)
model_env = mbrl.models.ModelEnv(env, dynamics_model, term_fn, reward_fn)
agent = mbrl.planning.create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles)

#######################################################################
wandb.init(project="RTDM", entity="rt_dm")
env = gym.make("Hopper-v2")

seed = 10
compute_times = []


config = wandb.config
config.learning_timestep = 10000
config.algorithm = 'A2C'
config.policy = 'MlpPolicy'
config.seed = seed


def play(env, model, times, asy = 0, level = 0):
    print('difficulty level:', level)
    total_rewards = []
    iter_ep = 3
    total_ep = level*iter_ep
    
    
    

    for k in range(iter_ep):
        print(k)
        obs = env.reset()
        total_reward = 0
        total_ep += 1
        wandb.log({"episode": total_ep, "difficulty_level": level})

        for i in range(times):


            t1 = time.time()
            action = model.act(obs, deterministic=True)
            # print(action)
            t2 = time.time()
            compute_time = 1000 * (t2 - t1)
            wandb.log({"computation_time": compute_time})
            compute_times.append(compute_time)
            obs, reward, done, info = env.step(action)
            repeat = int(level * compute_time)
            total_reward += reward
            if asy and repeat:
                for j in range(repeat):
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    if done:
                        total_rewards.append(total_reward)  
                        break          
            else:
                if done:
                    total_rewards.append(total_reward)
                    break
                    
            if done:
                obs = env.reset()
                break
    
        env.close()

    #print(total_rewards)
    reward_ave = sum(total_rewards)/len(total_rewards)
    wandb.log({"average_rewards": reward_ave, "difficulty_level": level})
    return reward_ave

record = []
reward_ave = play(env, agent, 800, asy = 0)
record.append(reward_ave)
x = range(0,20)
for level in x[1:]:
    print('here')
    reward_ave = play(env, agent, 5, asy = 1, level = level)
    record.append(reward_ave)
time_ave = sum(compute_times)/len(compute_times)
wandb.log({'average_compute_time':time_ave})
#print('final result:' , record)



