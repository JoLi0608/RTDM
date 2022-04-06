import gym
import time
import wandb
from ray import serve
from pydoc import doc
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ars as ars
import ray.rllib.agents.sac as sac
import argparse


# Input arguments from command line.
parser = argparse.ArgumentParser(description='Evaluate trained model')
parser.add_argument("--modelpath", required=True, help="Filepath to trained checkpoint",
                    default="/app/data/ray_results/2/ARS_CartPole-v0_661d3_00000_0_2022-03-31_10-07-40/checkpoint_000100/checkpoint-100")
parser.add_argument("--algorithm", required=True, help="Algorithm used", default="ARS")
parser.add_argument("--gymenv", required=True, help="Environment.",
                    default='CartPole-v0')
parser.add_argument("--checkpoint", required=True, help="checkpoint to evaluate",
                    default="1")
parser.add_argument("--trainseed", required=True, help="Training seed.",
                    default='2')
parser.add_argument("--evaseed", required=True, help="Evaluation seed.",
                    default=1)
args = vars(parser.parse_args())
print("Input of argparse:", args)

def play(env, trainer, times, asy = 0, level = 0):
    print('difficulty level:', level)
    total_rewards = []
    iter_ep = 20
    total_ep = level*iter_ep    

    for k in range(iter_ep):
        # print("here")
        obs = env.reset()
        total_reward = 0
        total_ep += 1
        wandb.log({"episode": total_ep, "difficulty_level": level})

        for i in range(times):
            t1 = time.time()
            action = trainer.compute_single_action(obs)
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
    reward_ave = sum(total_rewards)/len(total_rewards) if len(total_rewards) else sum(total_rewards)/(len(total_rewards)+1)
    
    wandb.log({"average_rewards": reward_ave, "difficulty_level": level})
    return reward_ave

seed = args["evaseed"]
algorithm = args["algorithm"]
trained_model = args["modelpath"]
env = gym.make(args["gymenv"])


wandb.init(project="RTDM", entity="rt_dm")
wconfig = wandb.config
wconfig.algorithm = args["algorithm"]
wconfig.eva_seed = args["evaseed"]
wconfig.train_seed = args["trainseed"]
wconfig.env = args["gymenv"]
wconfig.check = args["checkpoint"]


compute_times = []
serve.start()
if algorithm == "ARS":
    trainer = ars.ARSTrainer(
        config={
            "framework": "torch",
            # "num_workers": 4,
        },
        env=args["gymenv"],
    )
elif algorithm == "PPO":
    trainer = ppo.PPOTrainer(
        config={
            "framework": "torch",
            # "num_workers": 4,
        },
        env=args["gymenv"],
    )
elif algorithm == "SAC":
    trainer = sac.SACTrainer(
        config={
            "framework": "torch",
            # "num_workers": 4,
        },
        env=args["gymenv"],
    )

trainer.restore(trained_model)
record = []
reward_ave = play(env, trainer, 500, asy = 0)
record.append(reward_ave)
x = range(0,20)
for level in x[1:]:
    # print('here')
    reward_ave = play(env, trainer, 500, asy = 1, level = level)
    record.append(reward_ave)
time_ave = sum(compute_times)/len(compute_times)
wandb.log({'average_compute_time':time_ave})
#print('final result:' , record)