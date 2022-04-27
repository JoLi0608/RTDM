import wandb 
import pickle
import numpy as np


d = pickle.load(open("/app/data/inference_time/data.pkl","rb"))

   
for k in d.keys()
    run = wandb.init(project="RTDM_inference", entity="pierthodo")
    wconfig = wandb.config
    wconfig.algorithm = k.split("_")[2]
    wconfig.cpu = k.split("_")[1]
    if "continuous" in k:
        wconfig.env = "continuous-CartPole-v0"
        wconfig.gpu = k.split("_")[6]
    else
        wconfig.env = k.split("_")[3]
        wconfig.gpu = k.split("_")[5]
    wandb.log({"inference time":np.median(d["cpu_"+str( wconfig.cpu )+"_"+wconfig.algorithm +"_"+wconfig.env+"_gpu_"+k.split("_")[6]]),"cpu":cpu})
    run.finish()
