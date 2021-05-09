import retro
import gym
import numpy as np

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1', record=True)
env.reset()
done = False

from enum import Enum,unique
@unique
class Button_List(Enum):
    RIT = [0,0,0,0,0,0,0,1,0]
    RIT_A = [0,0,0,0,0,0,0,1,1]


combos=[['RIGHT'], ['RIGHT', 'A']]
RIT = [0,0,0,0,0,0,0,1,0]
RIT_A = [0,0,0,0,0,0,0,1,1]

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

episodes = 1000
for e in range(episodes):
    state = env.reset()
    while True :
        env.render()
        ob, rew, done, info = env.step(decode_discrete_action[1])
        rand_act = [RIT,RIT_A][np.random.randint(0,high=2)]
        print (rand_act)
        ob, rew, done, info = env.step(rand_act)
        state = ob
    #   print(f"\n{rew},\n{done},{info}")


# class SkipFrame(gym.Wrapper):
#     def __init__(self, env, skip):
#         """Return only every `skip`-th frame"""
#         super().__init__(env)
#         self._skip = skip

#     def step(self, action):
#         """Repeat action, and sum reward"""
#         total_reward = 0.0
#         done = False
#         for i in range(self._skip):
#             # Accumulate reward and repeat the same action
#             obs, reward, done, info = self.env.step(action)
#             total_reward += reward
#             if done:
#                 break
#         return obs, total_reward, done, info


# class GrayScaleObservation(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         obs_shape = self.observation_space.shape[:2]
#         self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

#     def permute_orientation(self, observation):
#         # permute [H, W, C] array to [C, H, W] tensor
#         observation = np.transpose(observation, (2, 0, 1))
#         observation = torch.tensor(observation.copy(), dtype=torch.float)
#         return observation

#     def observation(self, observation):
#         observation = self.permute_orientation(observation)
#         transform = T.Grayscale()
#         observation = transform(observation)
#         return observation


# class ResizeObservation(gym.ObservationWrapper):
#     def __init__(self, env, shape):
#         super().__init__(env)
#         if isinstance(shape, int):
#             self.shape = (shape, shape)
#         else:
#             self.shape = tuple(shape)

#         obs_shape = self.shape + self.observation_space.shape[2:]
#         self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

#     def observation(self, observation):
#         transforms = T.Compose(
#             [T.Resize(self.shape), T.Normalize(0, 255)]
#         )
#         observation = transforms(observation).squeeze(0)
#         return observation


# # Apply Wrappers to environment
# env = SkipFrame(env, skip=4)
# env = GrayScaleObservation(env)
# env = ResizeObservation(env, shape=84)
# env = FrameStack(env, num_stack=4)



# class Agent:
#     def __init__():
#         pass

#     def __init__():
#         pass

#     def act(self, state):
#         """Given a state, choose an epsilon-greedy action"""
#         pass

#     def cache(self, experience):
#         """Add the experience to memory"""
#         pass

#     def recall(self):
#         """Sample experiences from memory"""
#         pass

#     def learn(self):
#         """Update online action value (Q) function with a batch of experiences"""
#         passd



# ######################################################################
# # Agent
# # """""""""
# #
# # We create a class ``Mario`` to represent our agent in the game. Mario
# # should be able to:
# #
# # -  **Act** according to the optimal action policy based on the current
# #    state (of the environment).
# #
# # -  **Remember** experiences. Experience = (current state, current
# #    action, reward, next state). Mario *caches* and later *recalls* his
# #    experiences to update his action policy.
# #
# # -  **Learn** a better action policy over time
# #


# class Mario:
#     def __init__():
#         self.state_dim = (3,14,16) #state_dim #
#         self.action_dim = 2 # action_dim
#         self.save_dir = save_dir

#         self.use_cuda = torch.cuda.is_available()

#         # # Mario's DNN to predict the most optimal action - we implement this in the Learn section
#         # self.net = MarioNet(self.state_dim, self.action_dim).float()
#         # if self.use_cuda:
#         #     self.net = self.net.to(device="cuda")

#         self.exploration_rate = 1
#         self.exploration_rate_decay = 0.99999975
#         self.exploration_rate_min = 0.1
#         self.curr_step = 0

#         self.save_every = 5e5  # no. of experiences between saving Mario Net
#         self.memory = deque(maxlen=100000)
#         self.batch_size = 32

#     def act(self, state):
#         """
#         Given a state, choose an epsilon-greedy action and update value of step.

#         Inputs:
#         state(LazyFrame): A single observation of the current state, dimension is (state_dim)
#         Outputs:
#         action_idx (int): An integer representing which action Mario will perform
#         """
#         # EXPLORE
#         if np.random.rand() < self.exploration_rate:
#             action_idx = np.random.randint(self.action_dim)

#         # EXPLOIT
#         else:
#             state = state.__array__()
#             if self.use_cuda:
#                 state = torch.tensor(state).cuda()
#             else:
#                 state = torch.tensor(state)
#             state = state.unsqueeze(0)
#             action_values = self.net(state, model="online")
#             action_idx = torch.argmax(action_values, axis=1).item()

#         # decrease exploration_rate
#         self.exploration_rate *= self.exploration_rate_decay
#         self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

#         # increment step
#         self.curr_step += 1
#         return action_idx

#     def cache(self, state, next_state, action, reward, done):
#         """
#         Store the experience to self.memory (replay buffer)

#         Inputs:
#         state (LazyFrame),
#         next_state (LazyFrame),
#         action (int),
#         reward (float),
#         done(bool))
#         """
#         state = state.__array__()
#         next_state = next_state.__array__()

#         if self.use_cuda:
#             state = torch.tensor(state).cuda()
#             next_state = torch.tensor(next_state).cuda()
#             action = torch.tensor([action]).cuda()
#             reward = torch.tensor([reward]).cuda()
#             done = torch.tensor([done]).cuda()
#         else:
#             state = torch.tensor(state)
#             next_state = torch.tensor(next_state)
#             action = torch.tensor([action])
#             reward = torch.tensor([reward])
#             done = torch.tensor([done])

#         self.memory.append((state, next_state, action, reward, done,))

#     def recall(self):
#         """
#         Retrieve a batch of experiences from memory
#         """
#         batch = random.sample(self.memory, self.batch_size)
#         state, next_state, action, reward, done = map(torch.stack, zip(*batch))
#         return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

#     def learn(self):
#         """Update online action value (Q) function with a batch of experiences"""
#         pass

# Class AgentQ():
#     def __init__(self,action_range,state_range,):
#         pass

#     def


# class MarioNet(nn.Module):
#     """mini cnn structure
#   input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
#   """

#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         c, h, w = input_dim

#         if h != 84:
#             raise ValueError(f"Expecting input height: 84, got: {h}")
#         if w != 84:
#             raise ValueError(f"Expecting input width: 84, got: {w}")

#         self.online = nn.Sequential(
#             nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(3136, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_dim),
#         )

#         self.target = copy.deepcopy(self.online)

#         # Q_target parameters are frozen.
#         for p in self.target.parameters():
#             p.requires_grad = False

#     def forward(self, input, model):
#         if model == "online":
#             return self.online(input)
#         elif model == "target":
#             return self.target(input)

# class Mario(Mario):
#     def __init__(self, state_dim, action_dim, save_dir):
#         super().__init__(state_dim, action_dim, save_dir)
#         self.gamma = 0.9
#         self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
#         self.loss_fn = torch.nn.SmoothL1Loss()

#     def td_estimate(self, state, action):
#         # state = state_dim
#         # act = action_dim
#         # net object , np and action as subscript
#         # output of net is the init Q table
#         current_Q = self.net(state, model="online")[
#             np.arange(0, self.batch_size), action
#         ]  # Q_online(s,a)
#         return current_Q

#     @torch.no_grad()
#     def td_target(self, reward, next_state, done):
#         # next nn as input next state Q
#         next_state_Q = self.net(next_state, model="online")
#         # find max in next
#         best_action = torch.argmax(next_state_Q, axis=1)

#         # next predict Q value
#         next_Q = self.net(next_state, model="target")[
#             np.arange(0, self.batch_size), best_action
#         ]
#         # reward + gama * (max_next ) -Q

#         # reward + no value if die
#         return (reward + (1 - done.float()) * self.gamma * next_Q).float()

#     def update_Q_online(self, td_estimate, td_target):
#         # nn.loss =  smoothA
#         loss = self.loss_fn(td_estimate, td_target)

#         # adam
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         return loss.item()


#     def sync_Q_target(self):
#         # update Q by nn
#         self.net.target.load_state_dict(self.net.online.state_dict())


# ######################################################################
# # Save checkpoint
# # ~~~~~~~~~~~~~~~~~~
# #


# class Mario(Mario):
#     def save(self):
#         save_path = (
#             self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
#         )
#         torch.save(
#             dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
#             save_path,
#         )
#         print(f"MarioNet saved to {save_path} at step {self.curr_step}")


# ######################################################################
# # Putting it all together
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~
# #


# class Mario(Mario):
#     def __init__(self, state_dim, action_dim, save_dir):
#         super().__init__(state_dim, action_dim, save_dir)
#         self.burnin = 1e4  # min. experiences before training
#         self.learn_every = 3  # no. of experiences between updates to Q_online
#         self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

#     def learn(self):
#         if self.curr_step % self.sync_every == 0:
#             self.sync_Q_target()

#         if self.curr_step % self.save_every == 0:
#             self.save()

#         if self.curr_step < self.burnin:
#             return None, None

#         if self.curr_step % self.learn_every != 0:
#             return None, None

#         # Sample from memory
#         state, next_state, action, reward, done = self.recall()

#         # Get TD Estimate
#         td_est = self.td_estimate(state, action)

#         # Get TD Target
#         td_tgt = self.td_target(reward, next_state, done)

#         # Backpropagate loss through Q_online
#         loss = self.update_Q_online(td_est, td_tgt)

#         return (td_est.mean().item(), loss)


# ######################################################################
# # Logging
# # --------------
# #

# import numpy as np
# import time, datetime
# import matplotlib.pyplot as plt


# class MetricLogger:
#     def __init__(self, save_dir):
#         self.save_log = save_dir / "log"
#         with open(self.save_log, "w") as f:
#             f.write(
#                 f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
#                 f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
#                 f"{'TimeDelta':>15}{'Time':>20}\n"
#             )
#         self.ep_rewards_plot = save_dir / "reward_plot.jpg"
#         self.ep_lengths_plot = save_dir / "length_plot.jpg"
#         self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
#         self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

#         # History metrics
#         self.ep_rewards = []
#         self.ep_lengths = []
#         self.ep_avg_losses = []
#         self.ep_avg_qs = []

#         # Moving averages, added for every call to record()
#         self.moving_avg_ep_rewards = []
#         self.moving_avg_ep_lengths = []
#         self.moving_avg_ep_avg_losses = []
#         self.moving_avg_ep_avg_qs = []

#         # Current episode metric
#         self.init_episode()

#         # Timing
#         self.record_time = time.time()

#     def log_step(self, reward, loss, q):
#         self.curr_ep_reward += reward
#         self.curr_ep_length += 1
#         if loss:
#             self.curr_ep_loss += loss
#             self.curr_ep_q += q
#             self.curr_ep_loss_length += 1

#     def log_episode(self):
#         "Mark end of episode"
#         self.ep_rewards.append(self.curr_ep_reward)
#         self.ep_lengths.append(self.curr_ep_length)
#         if self.curr_ep_loss_length == 0:
#             ep_avg_loss = 0
#             ep_avg_q = 0
#         else:
#             ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
#             ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
#         self.ep_avg_losses.append(ep_avg_loss)
#         self.ep_avg_qs.append(ep_avg_q)

#         self.init_episode()

#     def init_episode(self):
#         self.curr_ep_reward = 0.0
#         self.curr_ep_length = 0
#         self.curr_ep_loss = 0.0
#         self.curr_ep_q = 0.0
#         self.curr_ep_loss_length = 0

#     def record(self, episode, epsilon, step):
#         mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
#         mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
#         mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
#         mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
#         self.moving_avg_ep_rewards.append(mean_ep_reward)
#         self.moving_avg_ep_lengths.append(mean_ep_length)
#         self.moving_avg_ep_avg_losses.append(mean_ep_loss)
#         self.moving_avg_ep_avg_qs.append(mean_ep_q)

#         last_record_time = self.record_time
#         self.record_time = time.time()
#         time_since_last_record = np.round(self.record_time - last_record_time, 3)

#         print(
#             f"Episode {episode} - "
#             f"Step {step} - "
#             f"Epsilon {epsilon} - "
#             f"Mean Reward {mean_ep_reward} - "
#             f"Mean Length {mean_ep_length} - "
#             f"Mean Loss {mean_ep_loss} - "
#             f"Mean Q Value {mean_ep_q} - "
#             f"Time Delta {time_since_last_record} - "
#             f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
#         )

#         with open(self.save_log, "a") as f:
#             f.write(
#                 f"{episode:8d}{step:8d}{epsilon:10.3f}"
#                 f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
#                 f"{time_since_last_record:15.3f}"
#                 f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
#             )

#         for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
#             plt.plot(getattr(self, f"moving_avg_{metric}"))
#             plt.savefig(getattr(self, f"{metric}_plot"))
#             plt.clf()


# ######################################################################
# # Let’s play!
# # """""""""""""""
# #
# # In this example we run the training loop for 10 episodes, but for Mario to truly learn the ways of
# # his world, we suggest running the loop for at least 40,000 episodes!
# #
# use_cuda = torch.cuda.is_available()
# print(f"Using CUDA: {use_cuda}")
# print()

# save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
# save_dir.mkdir(parents=True)

# mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

# logger = MetricLogger(save_dir)
# episodes = 10
# for e in range(episodes):

#     state = env.reset()

#     # Play the game!
#     while True:
#         env.render()

#         # Run agent on the state
#         action = mario.act(state)

#         # Agent performs action
#         next_state, reward, done, info = env.step(action)

#         # Remember
#         mario.cache(state, next_state, action, reward, done)

#         # Learn
#         q, loss = mario.learn()

#         # Logging
#         logger.log_step(reward, loss, q)

#         # Update state
#         state = next_state

#         # Check if end of game
#         if done or info["flag_get"]:
#             break

#     logger.log_episode()

#     if e % 20 == 0:
#         logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
