import gym
import time
from stable_baselines3 import PPO

import wandb

wandb.init(project="PPO", entity="lwyjo")
env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

config = wandb.config
config.learning_timestep = 10000
config.algorithm = 'PPO'

def play(env, model, times, asy = 0, level = 0):
    print('difficulty level:', level)
    total_rewards = []

    for k in range(20):
        obs = env.reset()
        total_reward = 0

        for i in range(times):


            t1 = time.time()
            action, _states = model.predict(obs, deterministic=True)
            t2 = time.time()
            compute_time = 1000 * (t2 - t1)
            wandb.log({"computation_time": compute_time})
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
    print(total_rewards)
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

print('final result:' , record)