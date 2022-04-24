import pandas as pd
import wandb
import time
import argparse
import pickle
import os
# Input arguments from command line.


parser = argparse.ArgumentParser(description='Evaluate trained model')

parser.add_argument("--path",  help="Filepath to trained checkpoint")
parser.add_argument("--algo",  help="Filepath to trained checkpoint")


def change_key(d,old_k,new_k):
    for i in range(len(d)):
        d[i][new_k] = d[i].pop(old_k)
    return d

def planet(path):
    data = pd.read_csv(path + "results.csv").to_dict('records')
    data = change_key(data,"env_step","step")
    data = change_key(data,"train_episode_reward","reward")
    return {"env":path.split("/")[5],"seed":float(path.split("/")[7]),"algo":"PLANET",
            "data":data}

def dreamer(path):
    data = pd.read_json(path+"metrics.jsonl", lines=True).to_dict('records')
    data = change_key(data,"train_return","reward")
    return {"env":path.split("/")[5],"seed":float(path.split("/")[7]),"algo":"DREAMER",
            "data":data}

def sac(path):
    data = pd.read_csv(path + "progress.txt", sep="\t").to_dict('records')
    data = change_key(data,"TotalEnvInteracts","step")
    data = change_key(data,"AverageEpRet","reward")
    return {"env":path.split("/")[5],"seed":float(path[-2]),"algo":"SAC",
            "data":data}

def ppo(path):
    data = pd.read_csv(path + "progress.txt", sep="\t").to_dict('records')
    data = change_key(data,"TotalEnvInteracts","step")
    data = change_key(data,"AverageEpRet","reward")
    return {"env":path.split("/")[5],"seed":float(path[-2]),"algo":"PPO",
            "data":data}

def pets(path):
    data = pd.read_csv(path + "results.csv").to_dict('records')
    data = change_key(data,"env_step","step")
    data = change_key(data,"episode_reward","reward")
    return {"env":path.split("/")[4],"seed": float(path.split("/")[5]),"algo":"PETS",
            "data":data}

def mbpo(path):
    data = pd.read_csv(path + "results.csv").to_dict('records')
    data = change_key(data,"env_step","step")
    data = change_key(data,"episode_reward","reward")
    return {"env":path.split("/")[5],"seed": float(path.split("/")[7]),"algo":"MBPO",
            "data":data}

def rtrl(path):
    data = pickle.load(open(path+"stats", "rb")).to_dict('records')
    data["step"] = data.index * 1000
    data = change_key(data,"train_episode_reward","reward")
    name = path.split("/")[-1]
    return {"env":name[2:-5],"seed": float(name.split("/")[0]),"algo":"RTRL",
            "data":data}


def ars(path):
    data = pd.read_csv(path + "progress.csv", low_memory=False).to_dict('records')
    data = change_key(data,"timesteps_total","step")
    data = change_key(data,"episode_reward_mean","reward")
    env = path.split("/")[5].split("_")[1]
    if env == "continuous":
        env = "continuous_CartPole-v0"
    return {"env":env,"seed": float(path.split("/")[4]),"algo":"ARS",
            "data":data}


args = vars(parser.parse_args())

data = globals()[args["algo"]](args["path"])


wandb.init(project="RTDM_train", entity="pierthodo")
wconfig = wandb.config
wconfig.seed = data["seed"]
wconfig.algo = data["algo"]
wconfig.env = data["env"]

for i in range(len(data["data"])):
    try:
        wandb.log(data["data"][i],step=int(data["data"][i]["step"]),commit=False)
    except:
        print("Error at step",i)

wandb.log({"Done":True},commit=True)

