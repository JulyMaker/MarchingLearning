import numpy as np
import gym

env = gym.make('CartPole-v0')

def run_random(env):
    for _ in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            #print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            #if done:
            #    print("Episode finished after {} timesteps".format(t+1))
            #    break
run_random(env)

mu = [0., 0., 0., 0.]  # first means
sigma = [1., 1., 1., 1.]  # first standard deviations
episodies = 100
iterations = 10

def run_episode(env, weights, render=False, max_reward=200):
    observation = env.reset()
    totalreward = 0
    for _ in range(max_reward):
        if render:
            env.render()
        action = 0 if np.matmul(weights, observation) < 0 else 1  # this line is our agent (just one neuron!!)
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

for iteration in range(iterations):

    weights = []
    for m, s in zip(mu, sigma):
        weights.append(np.random.normal(m, s, episodies))
    weights = np.transpose(weights)

    rewards = []
    number_of_goals = 0

    for w in weights:
        r = run_episode(env, w, max_reward=500)
        rewards.append(r)
        if r == 500:
            number_of_goals += 1

    # We combine in a list parameteres+rewards and sort it by rewards.
    # To do that we use a lambda function
    l = sorted(zip(weights, rewards), key=lambda pair: pair[1])
    # We get the last ten (they will be those with the higher reward), but
    # only first component (parameters) is needed. 
    l = list(zip(*l[-10:])[0])

    mu = np.mean(l, 0)
    sigma = np.std(l, 0)

    print ("------------")
    print ("Iteration:", iteration)
    print ("Mean:", mu)
    print ("Standard deviation:", sigma)
    print ("# goals:", number_of_goals)

    run_episode(env, mu, render=True, max_reward=7500)