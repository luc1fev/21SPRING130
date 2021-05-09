import retro
import gym
import numpy as np

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1', record=True)
env.reset()
done = False

combos=[['RIGHT'], ['RIGHT', 'A']]

buttons = env.unwrapped.buttons
print(buttons)

arr = np.array(env.action_space.n)
print(arr)

decode_discrete_action = []

for combo in combos:
    arr = np.array([False] * env.action_space.n)
    for button in combo:
        arr[buttons.index(button)] = True
    decode_discrete_action.append(arr)

env.action_space = gym.spaces.Discrete(len(decode_discrete_action))

print(decode_discrete_action,"dda")

for row in decode_discrete_action:
    print(row)

episodes = 1000
for e in range(episodes):
    state = env.reset()
    while True :
        env.render()
        ob, rew, done, info = env.step(decode_discrete_action[1])
        state = ob
        print(f"\n{rew},\n{done},{info}")