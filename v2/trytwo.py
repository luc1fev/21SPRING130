import retro
import gym
import numpy as np
from torch import nn
import torch
import cv2

import matplotlib.pyplot as plt





# class Visualizer(QtWidgets.QWidget):
#     def __init__(self, parent, size, config: Config, nn_viz: NeuralNetworkViz):
#         super().__init__(parent)
#         self.size = size
#         self.config = config
#         self.nn_viz = nn_viz
#         self.ram = None
#         self.x_offset = 150
#         self.tile_width, self.tile_height = self.config.Graphics.tile_size
#         self.tiles = None
#         self.enemies = None
#         self._should_update = True

#     def _draw_region_of_interest(self, painter: QPainter) -> None:
#         # Grab mario row/col in our tiles
#         mario = SMB.get_mario_location_on_screen(self.ram)
#         mario_row, mario_col = SMB.get_mario_row_col(self.ram)
#         x = mario_col

#         color = QColor(255, 0, 217)
#         painter.setPen(QPen(color, 3.0, Qt.SolidLine))
#         painter.setBrush(QBrush(Qt.NoBrush))

#         start_row, viz_width, viz_height = self.config.NeuralNetwork.input_dims
#         painter.drawRect(x*self.tile_width + 5 + self.x_offset, start_row*self.tile_height + 5, viz_width*self.tile_width, viz_height*self.tile_height)


#     def draw_tiles(self, painter: QPainter):
#         if not self.tiles:
#             return
#         for row in range(15):
#             for col in range(16):
#                 painter.setPen(QPen(Qt.black,  1, Qt.SolidLine))
#                 painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
#                 x_start = 5 + (self.tile_width * col) + self.x_offset
#                 y_start = 5 + (self.tile_height * row)

#                 loc = (row, col)
#                 tile = self.tiles[loc]

#                 if isinstance(tile, (StaticTileType, DynamicTileType, EnemyType)):
#                     rgb = ColorMap[tile.name].value
#                     color = QColor(*rgb)
#                     painter.setBrush(QBrush(color))
#                 else:
#                     pass

#                 painter.drawRect(x_start, y_start, self.tile_width, self.tile_height)

#     def paintEvent(self, event):
#         painter = QPainter()
#         painter.begin(self)

#         if self._should_update:
#             draw_border(painter, self.size)
#             if not self.ram is None:
#                 self.draw_tiles(painter)
#                 self._draw_region_of_interest(painter)
#                 self.nn_viz.show_network(painter)
#         else:
#             # draw_border(painter, self.size)
#             painter.setPen(QColor(0, 0, 0))
#             painter.setFont(QtGui.QFont('Times', 30, QtGui.QFont.Normal))
#             txt = 'Display is hidden.\nHit Ctrl+V to show\nConfig: {}'.format(args.config)
#             painter.drawText(event.rect(), Qt.AlignCenter, txt)
#             pass

#         painter.end()

#     def _update(self):
#         self.update()



# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self, config: Optional[Config] = None):
#         super().__init__()
#         global args
#         self.config = config
#         self.top = 150
#         self.left = 150
#         self.width = 1100
#         self.height = 700

#         self.title = 'Super Mario Bros AI'
#         self.current_generation = 0
#         # This is the generation that is actual 0. If you load individuals then you might end up starting at gen 12, in which case
#         # gen 12 would be the true 0
#         self._true_zero_gen = 0

#         self._should_display = True
#         self._timer = QTimer(self)
#         self._timer.timeout.connect(self._update)
#         # Keys correspond with B, NULL, SELECT, START, U, D, L, R, A
#         # index                0  1     2       3      4  5  6  7  8
#         self.keys = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)

#         # I only allow U, D, L, R, A, B and those are the indices in which the output will be generated
#         # We need a mapping from the output to the keys above
#         self.ouput_to_keys_map = {
#             0: 4,  # U
#             1: 5,  # D
#             2: 6,  # L
#             3: 7,  # R
#             4: 8,  # A
#             5: 0   # B
#         }

#         # Initialize the starting population
#         individuals: List[Individual] = []

#         # Load any individuals listed in the args.load_inds
#         num_loaded = 0
#         if args.load_inds:
#             # Overwrite the config file IF one is not specified
#             if not self.config:
#                 try:
#                     self.config = Config(os.path.join(args.load_file, 'settings.config'))
#                 except:
#                     raise Exception(f'settings.config not found under {args.load_file}')

#             set_of_inds = set(args.load_inds)

#             for ind_name in os.listdir(args.load_file):
#                 if ind_name.startswith('best_ind_gen'):
#                     ind_number = int(ind_name[len('best_ind_gen'):])
#                     if ind_number in set_of_inds:
#                         individual = load_mario(args.load_file, ind_name, self.config)
#                         # Set debug stuff if needed
#                         if args.debug:
#                             individual.name = f'm{num_loaded}_loaded'
#                             individual.debug = True
#                         individuals.append(individual)
#                         num_loaded += 1

#             # Set the generation
#             self.current_generation = max(set_of_inds) + 1  # +1 becauase it's the next generation
#             self._true_zero_gen = self.current_generation

#         # Load any individuals listed in args.replay_inds
#         if args.replay_inds:
#             # Overwrite the config file IF one is not specified
#             if not self.config:
#                 try:
#                     self.config = Config(os.path.join(args.replay_file, 'settings.config'))
#                 except:
#                     raise Exception(f'settings.config not found under {args.replay_file}')

#             for ind_gen in args.replay_inds:
#                 ind_name = f'best_ind_gen{ind_gen}'
#                 fname = os.path.join(args.replay_file, ind_name)
#                 if os.path.exists(fname):
#                     individual = load_mario(args.replay_file, ind_name, self.config)
#                     # Set debug stuff if needed
#                     if args.debug:
#                         individual.name= f'm_gen{ind_gen}_replay'
#                         individual.debug = True
#                     individuals.append(individual)
#                 else:
#                     raise Exception(f'No individual named {ind_name} under {args.replay_file}')
#         # If it's not a replay then we need to continue creating individuals
#         else:
#             num_parents = max(self.config.Selection.num_parents - num_loaded, 0)
#             for _ in range(num_parents):
#                 individual = Mario(self.config)
#                 # Set debug stuff if needed
#                 if args.debug:
#                     individual.name = f'm{num_loaded}'
#                     individual.debug = True
#                 individuals.append(individual)
#                 num_loaded += 1

#         self.best_fitness = 0.0
#         self._current_individual = 0
#         self.population = Population(individuals)

#         self.mario = self.population.individuals[self._current_individual]

#         self.max_distance = 0  # Track farthest traveled in level
#         self.max_fitness = 0.0
#         self.env = retro.make(game='SuperMarioBros-Nes', state=f'Level{self.config.Misc.level}')

#         # Determine the size of the next generation based off selection type
#         self._next_gen_size = None
#         if self.config.Selection.selection_type == 'plus':
#             self._next_gen_size = self.config.Selection.num_parents + self.config.Selection.num_offspring
#         elif self.config.Selection.selection_type == 'comma':
#             self._next_gen_size = self.config.Selection.num_offspring

#         # If we aren't displaying we need to reset the environment to begin with
#         if args.no_display:
#             self.env.reset()
#         else:
#             self.init_window()

#             # Set the generation in the label if needed
#             if args.load_inds:
#                 txt = "<font color='red'>" + str(self.current_generation + 1) + '</font>'  # +1 because we switch from 0 to 1 index
#                 self.info_window.generation.setText(txt)

#             # if this is a replay then just set current_individual to be 'replay' and set generation
#             if args.replay_file:
#                 self.info_window.current_individual.setText('Replay')
#                 txt = f"<font color='red'>{args.replay_inds[self._current_individual] + 1}</font>"
#                 self.info_window.generation.setText(txt)

#             self.show()

#         if args.no_display:
#             self._timer.start(1000 // 1000)
#         else:
#             self._timer.start(1000 // 60)

#     def init_window(self) -> None:
#         self.centralWidget = QtWidgets.QWidget(self)
#         self.setCentralWidget(self.centralWidget)
#         self.setWindowTitle(self.title)
#         self.setGeometry(self.top, self.left, self.width, self.height)

#         self.game_window = GameWindow(self.centralWidget, (514, 480), self.config)
#         self.game_window.setGeometry(QRect(1100-514, 0, 514, 480))
#         self.game_window.setObjectName('game_window')
#         # # Reset environment and pass the screen to the GameWindow
#         screen = self.env.reset()
#         self.game_window.screen = screen

#         self.viz = NeuralNetworkViz(self.centralWidget, self.mario, (1100-514, 700), self.config)

#         self.viz_window = Visualizer(self.centralWidget, (1100-514, 700), self.config, self.viz)
#         self.viz_window.setGeometry(0, 0, 1100-514, 700)
#         self.viz_window.setObjectName('viz_window')
#         self.viz_window.ram = self.env.get_ram()

#         self.info_window = InformationWidget(self.centralWidget, (514, 700-480), self.config)
#         self.info_window.setGeometry(QRect(1100-514, 480, 514, 700-480))

#     def keyPressEvent(self, event):
#         k = event.key()
#         # m = {
#         #     Qt.Key_Right : 7,
#         #     Qt.Key_C : 8,
#         #     Qt.Key_X: 0,
#         #     Qt.Key_Left: 6,
#         #     Qt.Key_Down: 5
#         # }
#         # if k in m:
#         #     self.keys[m[k]] = 1
#         # if k == Qt.Key_D:
#         #     tiles = SMB.get_tiles(self.env.get_ram(), False)
#         modifier = int(event.modifiers())
#         if modifier == Qt.CTRL:
#             if k == Qt.Key_V:
#                 self._should_display = not self._should_display

#     def keyReleaseEvent(self, event):
#         k = event.key()
#         m = {
#             Qt.Key_Right : 7,
#             Qt.Key_C : 8,
#             Qt.Key_X: 0,
#             Qt.Key_Left: 6,
#             Qt.Key_Down: 5
#         }
#         if k in m:
#             self.keys[m[k]] = 0


#     def next_generation(self) -> None:
#         self._increment_generation()
#         self._current_individual = 0

#         if not args.no_display:
#             self.info_window.current_individual.setText('{}/{}'.format(self._current_individual + 1, self._next_gen_size))

#         # Calculate fitness
#         # print(', '.join(['{:.2f}'.format(i.fitness) for i in self.population.individuals]))

#         if args.debug:
#             print(f'----Current Gen: {self.current_generation}, True Zero: {self._true_zero_gen}')
#             fittest = self.population.fittest_individual
#             print(f'Best fitness of gen: {fittest.fitness}, Max dist of gen: {fittest.farthest_x}')
#             num_wins = sum(individual.did_win for individual in self.population.individuals)
#             pop_size = len(self.population.individuals)
#             print(f'Wins: {num_wins}/{pop_size} (~{(float(num_wins)/pop_size*100):.2f}%)')

#         if self.config.Statistics.save_best_individual_from_generation:
#             folder = self.config.Statistics.save_best_individual_from_generation
#             best_ind_name = 'best_ind_gen{}'.format(self.current_generation - 1)
#             best_ind = self.population.fittest_individual
#             save_mario(folder, best_ind_name, best_ind)

#         if self.config.Statistics.save_population_stats:
#             fname = self.config.Statistics.save_population_stats
#             save_stats(self.population, fname)

#         self.population.individuals = elitism_selection(self.population, self.config.Selection.num_parents)

#         random.shuffle(self.population.individuals)
#         next_pop = []

#         # Parents + offspring
#         if self.config.Selection.selection_type == 'plus':
#             # Decrement lifespan
#             for individual in self.population.individuals:
#                 individual.lifespan -= 1

#             for individual in self.population.individuals:
#                 config = individual.config
#                 chromosome = individual.network.params
#                 hidden_layer_architecture = individual.hidden_layer_architecture
#                 hidden_activation = individual.hidden_activation
#                 output_activation = individual.output_activation
#                 lifespan = individual.lifespan
#                 name = individual.name

#                 # If the indivdual would be alve, add it to the next pop
#                 if lifespan > 0:
#                     m = Mario(config, chromosome, hidden_layer_architecture, hidden_activation, output_activation, lifespan)
#                     # Set debug if needed
#                     if args.debug:
#                         m.name = f'{name}_life{lifespan}'
#                         m.debug = True
#                     next_pop.append(m)

#         num_loaded = 0

#         while len(next_pop) < self._next_gen_size:
#             selection = self.config.Crossover.crossover_selection
#             if selection == 'tournament':
#                 p1, p2 = tournament_selection(self.population, 2, self.config.Crossover.tournament_size)
#             elif selection == 'roulette':
#                 p1, p2 = roulette_wheel_selection(self.population, 2)
#             else:
#                 raise Exception('crossover_selection "{}" is not supported'.format(selection))

#             L = len(p1.network.layer_nodes)
#             c1_params = {}
#             c2_params = {}

#             # Each W_l and b_l are treated as their own chromosome.
#             # Because of this I need to perform crossover/mutation on each chromosome between parents
#             for l in range(1, L):
#                 p1_W_l = p1.network.params['W' + str(l)]
#                 p2_W_l = p2.network.params['W' + str(l)]
#                 p1_b_l = p1.network.params['b' + str(l)]
#                 p2_b_l = p2.network.params['b' + str(l)]

#                 # Crossover
#                 # @NOTE: I am choosing to perform the same type of crossover on the weights and the bias.
#                 c1_W_l, c2_W_l, c1_b_l, c2_b_l = self._crossover(p1_W_l, p2_W_l, p1_b_l, p2_b_l)

#                 # Mutation
#                 # @NOTE: I am choosing to perform the same type of mutation on the weights and the bias.
#                 self._mutation(c1_W_l, c2_W_l, c1_b_l, c2_b_l)

#                 # Assign children from crossover/mutation
#                 c1_params['W' + str(l)] = c1_W_l
#                 c2_params['W' + str(l)] = c2_W_l
#                 c1_params['b' + str(l)] = c1_b_l
#                 c2_params['b' + str(l)] = c2_b_l

#                 #  Clip to [-1, 1]
#                 np.clip(c1_params['W' + str(l)], -1, 1, out=c1_params['W' + str(l)])
#                 np.clip(c2_params['W' + str(l)], -1, 1, out=c2_params['W' + str(l)])
#                 np.clip(c1_params['b' + str(l)], -1, 1, out=c1_params['b' + str(l)])
#                 np.clip(c2_params['b' + str(l)], -1, 1, out=c2_params['b' + str(l)])


#             c1 = Mario(self.config, c1_params, p1.hidden_layer_architecture, p1.hidden_activation, p1.output_activation, p1.lifespan)
#             c2 = Mario(self.config, c2_params, p2.hidden_layer_architecture, p2.hidden_activation, p2.output_activation, p2.lifespan)

#             # Set debug if needed
#             if args.debug:
#                 c1_name = f'm{num_loaded}_new'
#                 c1.name = c1_name
#                 c1.debug = True
#                 num_loaded += 1

#                 c2_name = f'm{num_loaded}_new'
#                 c2.name = c2_name
#                 c2.debug = True
#                 num_loaded += 1

#             next_pop.extend([c1, c2])

#         # Set next generation
#         random.shuffle(next_pop)
#         self.population.individuals = next_pop

#     def _crossover(self, parent1_weights: np.ndarray, parent2_weights: np.ndarray,
#                    parent1_bias: np.ndarray, parent2_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         eta = self.config.Crossover.sbx_eta

#         # SBX weights and bias
#         child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, eta)
#         child1_bias, child2_bias =  SBX(parent1_bias, parent2_bias, eta)

#         return child1_weights, child2_weights, child1_bias, child2_bias


#     def _mutation(self, child1_weights: np.ndarray, child2_weights: np.ndarray,
#                   child1_bias: np.ndarray, child2_bias: np.ndarray) -> None:
#         mutation_rate = self.config.Mutation.mutation_rate
#         scale = self.config.Mutation.gaussian_mutation_scale

#         if self.config.Mutation.mutation_rate_type == 'dynamic':
#             mutation_rate = mutation_rate / math.sqrt(self.current_generation + 1)

#         # Mutate weights
#         gaussian_mutation(child1_weights, mutation_rate, scale=scale)
#         gaussian_mutation(child2_weights, mutation_rate, scale=scale)

#         # Mutate bias
#         gaussian_mutation(child1_bias, mutation_rate, scale=scale)
#         gaussian_mutation(child2_bias, mutation_rate, scale=scale)


#     def _increment_generation(self) -> None:
#         self.current_generation += 1
#         if not args.no_display:
#             txt = "<font color='red'>" + str(self.current_generation + 1) + '</font>'
#             self.info_window.generation.setText(txt)


#     def _update(self) -> None:
#         """
#         This is the main update method which is called based on the FPS timer.
#         Genetic Algorithm updates, window updates, etc. are performed here.
#         """
#         ret = self.env.step(self.mario.buttons_to_press)

#         if not args.no_display:
#             if self._should_display:
#                 self.game_window.screen = ret[0]
#                 self.game_window._should_update = True
#                 self.info_window.show()
#                 self.viz_window.ram = self.env.get_ram()
#             else:
#                 self.game_window._should_update = False
#                 self.info_window.hide()
#             self.game_window._update()

#         ram = self.env.get_ram()
#         tiles = SMB.get_tiles(ram)  # Grab tiles on the screen
#         enemies = SMB.get_enemy_locations(ram)

#         # self.mario.set_input_as_array(ram, tiles)
#         self.mario.update(ram, tiles, self.keys, self.ouput_to_keys_map)

#         if not args.no_display:
#             if self._should_display:
#                 self.viz_window.ram = ram
#                 self.viz_window.tiles = tiles
#                 self.viz_window.enemies = enemies
#                 self.viz_window._should_update = True
#             else:
#                 self.viz_window._should_update = False
#             self.viz_window._update()

#         if self.mario.is_alive:
#             # New farthest distance?
#             if self.mario.farthest_x > self.max_distance:
#                 if args.debug:
#                     print('New farthest distance:', self.mario.farthest_x)
#                 self.max_distance = self.mario.farthest_x
#                 if not args.no_display:
#                     self.info_window.max_distance.setText(str(self.max_distance))
#         else:
#             self.mario.calculate_fitness()
#             fitness = self.mario.fitness

#             if fitness > self.max_fitness:
#                 self.max_fitness = fitness
#                 max_fitness = '{:.2f}'.format(self.max_fitness)
#                 if not args.no_display:
#                     self.info_window.best_fitness.setText(max_fitness)
#             # Next individual
#             self._current_individual += 1

#             # Are we replaying from a file?
#             if args.replay_file:
#                 if not args.no_display:
#                     # Set the generation to be whatever best individual is being ran (+1)
#                     # Check to see if there is a next individual, otherwise exit
#                     if self._current_individual >= len(args.replay_inds):
#                         if args.debug:
#                             print(f'Finished replaying {len(args.replay_inds)} best individuals')
#                         sys.exit()

#                     txt = f"<font color='red'>{args.replay_inds[self._current_individual] + 1}</font>"
#                     self.info_window.generation.setText(txt)
#             else:
#                 # Is it the next generation?
#                 if (self.current_generation > self._true_zero_gen and self._current_individual == self._next_gen_size) or\
#                     (self.current_generation == self._true_zero_gen and self._current_individual == self.config.Selection.num_parents):
#                     self.next_generation()
#                 else:
#                     if self.current_generation == self._true_zero_gen:
#                         current_pop = self.config.Selection.num_parents
#                     else:
#                         current_pop = self._next_gen_size
#                     if not args.no_display:
#                         self.info_window.current_individual.setText('{}/{}'.format(self._current_individual + 1, current_pop))

#             if args.no_display:
#                 self.env.reset()
#             else:
#                 self.game_window.screen = self.env.reset()

#             self.mario = self.population.individuals[self._current_individual]

#             if not args.no_display:
#                 self.viz.mario = self.mario




env = retro.make(game='SuperMarioBros-Nes', state='Level1-1', record=True)
env.reset()
done = False

from enum import Enum,unique
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


@unique
class Button(Enum):
    """
    # key     B, NULL, SELECT, START, U, D, L, R, A
    # index   0  1     2       3      4  5  6  7  8
    """
    RIT = [0,0,0,0,0,0,0,1,0]
    RIT_A =   [0,0,0,0,0,0,0,1,1]
    Null =[0,1,0,0,0,0,0,0,0]

# # 3x224x24

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(224*240, 159*170),
#             nn.ReLU(),
#             nn.Linear(159*170, 113*120),
#             nn.ReLU(),
#             nn.Linear(512, 64),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits

# model = NeuralNetwork().to(device)
# print(model)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)

#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
class RAM():
    __init__:(self,ob, rew, done, info):
        self.ob = ob,
        self.rew = rew
        self.done = done
        self.info = info

    def obTOnp:
        img_table = np.array()


episodes = 1000
for e in range(episodes):
    state = env.reset()
    size =5
    while True :
        for i in range(size):
            env.render()
            # print(decode_discrete_action[1])
            # ob, rew, done, info = env.step(decode_discrete_action[1])

            rand_act = random.choice(list(Button))
            print(rand_act)
            ob, rew, done, info = env.step(rand_act.value)
            state = ob
            if i ==size:
                i=0


            im = np.array(ob)

            plt.imshow(im,'gray')

            #img = cv2.imread(ob)

            #w, h = img.shape[:2][::-1]

            # img_resize = cv2.resize(img,
            # (int(width*0.5),int(height*0.5)),interpolation=cv2.INTER_CUBIC)
            #print(img)
            #cv2.imshow("img",img)
            # print("img_reisze shape:{}".format(np.shape(img_resize)))

            cv2.waitKey(0)
            # print(ob)
            # print(np.shape(ob))
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
