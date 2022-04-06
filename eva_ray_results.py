# import ray
import gym
import time
import wandb
from ray import serve
from pydoc import doc
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ars as ars
import ray.rllib.agents.sac as sac
# from ray.rllib.agents.sac import SACTrainer
# from starlette.requests import Request
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

env_seeds = 1
train_seed = 2

# for env_seed in env_seeds:
gym_env = "Hopper-v2"
check_point = 1
algorithm = "ARS"
trained_model = "/Users/liwenyu/Downloads/ray_results/ARS_Hopper-v2_303ed_00000_0_2022-03-30_13-27-47/checkpoint_000100/checkpoint-100"
wandb.init(project="RTDM", entity="rt_dm")
env = gym.make(gym_env)

compute_times = []
wconfig = wandb.config
# wconfig.learning_timestep = 10000
wconfig.algorithm = algorithm
wconfig.env_seed = env_seeds
wconfig.train_seed = train_seed
wconfig.env = gym_env
wconfig.check = check_point
serve.start()
trainer = ars.ARSTrainer(
    config={
        "framework": "torch",
        # "num_workers": 4,
    },
    env=gym_env,
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