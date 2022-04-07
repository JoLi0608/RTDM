import rtrl
import argparse
import os
from rtrl import Training, run
from rtrl.wrappers import StatsWrapper
import gym

# Input arguments from command line.


parser = argparse.ArgumentParser(description='Evaluate trained model')

parser.add_argument("--path",  help="Filepath to trained checkpoint",
                    default="/app/data/rtrl/5/checkpoint/Pusher-v2/04-05-2022-07-10-57/")

args = vars(parser.parse_args())

path = args["path"]
path_split = path.split(path,"/")
r = rtrl.load(path+"store")

env = gym.make("Pusher-v2")
obs = e.reset()
prev_action = np.zeros(env.action_space.shape[0])

for step in range(10000):
    action = r.agent.act((obs,prev_action),[],[],[]),train=False)[0]
    obs,reward,done,info =env.step(action)
    prev_action = action
