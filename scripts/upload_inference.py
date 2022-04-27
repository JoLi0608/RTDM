import wandb 
import pickle
import numpy as np


d = pickle.load(open("/app/data/inference_time/data.pkl","rb"))
for k in d.keys():
    cpu = d.split("_")[1]
    algorithm = d.split("_")[2]
    env_name = d.split("_")[3]
    if env_name == "continuous":
        env_name =  d.split("_")[3]+"_"+ d.split("_")[4]
        gpu =  d.split("_")[5]
    else:
        gpu = 4
    time = np.median(d[k])
wandb.init(project="RTDM_inference", entity="pierthodo")
