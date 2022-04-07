from mbrl.planning.core import load_agent
import gym

env = gym.make("Hopper-v2")
agent = load_agent("/app/data/mbpo/default/gym___Hopper-v2/2022.04.01/100721",env)

agent.act(env.reset())
