import wandb 
import pickle
import numpy as np


d = pickle.load(open("/app/data/inference_time/data.pkl","rb"))

for algo in ["ars","mbpo","sac","ppo","rtrl"]:
    for env in ["Hopper-v2","HalfCheetah-v2","continuous_CartPole-v0","Humanoid-v2","Pusher-v2"]:
        run = wandb.init(project="RTDM_inference", entity="pierthodo")
        for cpu in [4,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02]:
            wconfig = wandb.config
            wconfig.algorithm = algo
            wconfig.env = env
            wandb.log({"inference time":np.median(d["cpu_"+cpu+"_"+algo+"_"+env+"_gpu_0"]),"cpu":cpu})
        run.finish()
