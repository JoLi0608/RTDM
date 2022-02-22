import gym
import time
from stable_baselines3 import PPO

import wandb

wandb.init(project="RTDM", entity="rt_dm")
env = gym.make("CartPole-v1")

seed = 666
compute_times = []

model = PPO("MlpPolicy", env, verbose=1, seed = seed)
model.learn(total_timesteps=10000)

config = wandb.config
config.learning_timestep = 10000
config.algorithm = 'PPO'
config.policy = 'MlpPolicy'
config.seed = seed

def play(env, model, times, asy = 0, level = 0):
    #print('difficulty level:', level)
    total_rewards = []
    iter_ep = 20
    total_ep = level*iter_ep
    
    
    

    for k in range(iter_ep):
        obs = env.reset()
        total_reward = 0
        total_ep += 1
        wandb.log({"episode": total_ep, "difficulty_level": level})

        for i in range(times):


            t1 = time.time()
            action, _states = model.predict(obs, deterministic=True)
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
    reward_ave = sum(total_rewards)/len(total_rewards)
    wandb.log({"average_rewards": reward_ave, "difficulty_level": level})
    return reward_ave

record = []
reward_ave = play(env, model, 800, asy = 0)
record.append(reward_ave)
x = range(0,20)
for level in x[1:]:
    reward_ave = play(env, model, 800, asy = 1, level = level)
    record.append(reward_ave)
time_ave = sum(compute_times)/len(compute_times)
wandb.log({'average_compute_time':time_ave})
#print('final result:' , record)