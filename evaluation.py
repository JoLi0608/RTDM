import gym
import time
import wandb
from ray.rllib.agents.ppo import PPOTrainer


wandb.init(project="RTDM", entity="rt_dm")
algorithm = 'PPO'
environment = "CartPole-v1"
env = gym.make(environment)


seed = 99
compute_times = []
wconfig = wandb.config
# wconfig.learning_timestep = 10000
wconfig.algorithm = algorithm
# wconfig.policy = 'MlpPolicy'
# wconfig.seed = seed
wconfig.seed = seed
wconfig.env = environment

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": environment,
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 2,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch"

}

# Create our RLlib Trainer.
trainer = PPOTrainer(config=config)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for _ in range(3):
    trainer.train()

print(trainer.compute_action)
# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
# trainer.evaluate()


def play(env, trainer, times, asy = 0, level = 0):
    print('difficulty level:', level)
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

record = []
reward_ave = play(env, trainer, 800, asy = 0)
record.append(reward_ave)
x = range(0,20)
for level in x[1:]:
    print('here')
    reward_ave = play(env, trainer, 800, asy = 1, level = level)
    record.append(reward_ave)
time_ave = sum(compute_times)/len(compute_times)
wandb.log({'average_compute_time':time_ave})
#print('final result:' , record)


