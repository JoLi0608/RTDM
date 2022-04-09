
import omegaconf
from sklearn import gaussian_process
import torch
import mbrl.util.env
import mbrl.util.common
import mbrl.planning
import gym
import time
import mbrl.env.pets_pusher as pusher
import numpy
# from numpy import average
# from sympy import total_degree
import wandb
import argparse
from mbrl.planning.core import load_agent
import gym

#path = "exp/pets/default/cartpole_continuous/2022.02.21/134508/"
# path = "exp/pets/default/cartpole_continuous/2022.02.21/134508/"
parser = argparse.ArgumentParser(description='Evaluate trained model')
parser.add_argument("--modelpath", required=True, help="Filepath to trained checkpoint",
                    default="/app/data/mbpo/default/gym___Hopper-v2/2022.04.01/100715/")
# parser.add_argument("--trainseed", required=True, help="Training seed.",
#                     default='2')
parser.add_argument("--algorithm", required=True, help="Algorithm used", default="mbpo")
parser.add_argument("--gymenv", required=True, help="Environment.",
                    default='Hopper-v2')
parser.add_argument("--trainseed", required=True, help="Training seed.",
                    default='100715')
parser.add_argument("--evaseed", required=True, help="Evaluation seed.",
                    default=1)
args = vars(parser.parse_args())
print("Input of argparse:", args)


#######################################################################
wandb.init(project="RTDM", entity="rt_dm")
# env = mbrl.util.env(args["gymenv"])
env = pusher.PusherEnv()
agent = load_agent(args["modelpath"],env)

seed = int(args["evaseed"])
env.seed(seed)
compute_times = []


wconfig = wandb.config
wconfig.algorithm = args["algorithm"]
wconfig.eva_seed = seed
wconfig.train_seed = args["trainseed"]
wconfig.env = args["gymenv"]


def play(env, agent, times, gap, level = 0):
    print('difficulty level:', level)
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
            action = agent.act(obs, deterministic=True)
            # print(action)
            t2 = time.time()
            compute_time = (t2 - t1)
            wandb.log({"computation_time": compute_time})
            compute_times.append(compute_time)
            obs, reward, done, info = env.step(action)
            repeat = int(level * 1 * compute_time)
            total_reward += reward
            if repeat:
                for j in range(repeat):
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    if done:
                        total_rewards.append(total_reward)  
                        obs = env.reset()
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
    reward_ave = sum(total_rewards)/len(total_rewards) if len(total_rewards) else sum(total_rewards)/(len(total_rewards)+1)
    
    wandb.log({"average_rewards": reward_ave, "difficulty_level": level})
    return reward_ave

record = []
begin = 0
gap = 500
end = 10000
x = numpy.arange(begin, end, gap)
reward_ave = play(env, agent, 100000, gap = gap)
record.append(reward_ave)
for level in x[1:]:
    # print('here')
    reward_ave = play(env, agent, 1000000, gap = gap, level = level)
    record.append(reward_ave)
time_ave = sum(compute_times)/len(compute_times)
wandb.log({'average_compute_time':time_ave})
#print('final result:' , record)



