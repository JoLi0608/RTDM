import pandas as pd
import wandb
import time
import argparse
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
            "data":tmp}

args = vars(parser.parse_args())

data = locals()[args["algo"]](args["path"])



wandb.init(project="RTDM_train", entity="pierthodo")
wconfig = wandb.config
wconfig.seed = data["seed"]
wconfig.algo = data["algo"]
wconfig.env = data["env"]

for i in data["data"]:
    if (data["env"] == "Humanoid-v2") and (i["env_step"] > 200000) and (data["algo"] == "MBPO"):
        break
    wandb.log(i,step=int(i["env_step"]),commit=False)


wandb.log({"Done":True},commit=True)

