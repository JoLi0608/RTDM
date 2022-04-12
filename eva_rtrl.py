import rtrl
import argparse
import os
from rtrl import Training, run
from rtrl.wrappers import StatsWrapper
import gym
import numpy as np
import wandb

# Input arguments from command line.


parser = argparse.ArgumentParser(description='Evaluate trained model')

parser.add_argument("--modelpath",  help="Filepath to trained checkpoint",
                    default="/app/data/rtrl/1/checkpoint/Hopper-v2/04-01-2022-13-10-27")
parser.add_argument("--algorithm", required=True, help="Algorithm used", default="rtrl")
parser.add_argument("--trainseed", required=True, help="Training seed.",
                    default='1')
parser.add_argument("--gymenv", required=True, help="Environment.",
                    default='Hopper-v2')
parser.add_argument("--evaseed", required=True, help="Evaluation seed.",
                    default=1)

args = vars(parser.parse_args())

path = args["modelpath"]
path_split = path.split(path,"/")
r = rtrl.load(path+"store")
agent = r.agent

env = gym.make(args["gymenv"])

wandb.init(project="RTDM", entity="rt_dm")
seed = int(args["evaseed"])
env.seed(seed)
compute_times = []


wconfig = wandb.config
wconfig.algorithm = args["algorithm"]
wconfig.eva_seed = seed
wconfig.train_seed = args["trainseed"]
wconfig.env = args["gymenv"]
wconfig.eva_seed = args["evaseed"]


def play(env, agent, times, gap, level = 0):
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
            action = r.agent.act((obs,prev_action),[],[],[]),train=False)[0]
            # print(action)
            t2 = time.time()
            compute_time = (t2 - t1)
            wandb.log({"computation_time": compute_time})
            compute_times.append(compute_time)
            obs, reward, done, info = env.step(action)
            prev_action = action
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
x = np.arange(begin, end, gap)
reward_ave = play(env, agent, 100000, gap = gap)
record.append(reward_ave)
for level in x[1:]:
    # print('here')
    reward_ave = play(env, agent, 1000000, gap = gap, level = level)
    record.append(reward_ave)
time_ave = sum(compute_times)/len(compute_times)
wandb.log({'average_compute_time':time_ave})
#print('final result:' , record)

# for step in range(10000):
#     action = r.agent.act((obs,prev_action),[],[],[]),train=False)[0]
#     obs,reward,done,info =env.step(action)
#     prev_action = action
