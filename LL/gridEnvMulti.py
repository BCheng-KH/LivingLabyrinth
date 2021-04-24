from gym import Env
from gym.spaces import MultiDiscrete, Box, Discrete
import numpy as np
import random
import copy

class envGrid(Env):
    def __init__(self, size = 10, playerPos = [random.randrange(1, 9), 9], dronePos = [random.randrange(1, 9), 0]):
        #self.action_space = MultiDiscrete([5, 82, 82])
        self.action_space = Discrete(169)
        self.observation_space = Box(low=np.array([[0 for _ in range(265)]]),high=np.array([[2, 9, 9]+[3 for _ in range(100)]+[1 for _ in range(81)]+[1 for _ in range(81)]]))#2(x, y), 100(grid), 81(wallH), 81(wallV)

        self.size = size
        self.visionGrid = [[0 for ii in range(size)] for i in range(size)] # 0 = nothing, 1 = rigged, 2 = player was there, 3 = player is there, this is what ai sees
        self.playerPos = playerPos
        self.dronePos = dronePos
        self.wallGrid = [[0 for ii in range(size-1)] for i in range(size-1)]
        self.level = random.randrange(1, 6)
        self.turn = 0
        self.roll = 0
    def wipeAll(self):
        self.wallGrid = [[0 for ii in range(self.size-1)] for i in range(self.size-1)]
        self.visionGrid = [[0 for ii in range(self.size)] for i in range(self.size)]
        self.dronePos = [random.randrange(1, 9), 0]
        self.playerPos = [random.randrange(1, 9), 9]
        self.turn = 0
        self.score = 0
        self.roll = 0
    def CheckWin(self, coordxFirst, coordyFirst, coordxSecond, coordySecond): #Checks to see if the palyer is on the drone.
        if coordxFirst == coordxSecond and coordyFirst == coordySecond:
            return True

        else:
            return False
    def rigCells(self):
        ratio = self.level * 0.2
        riggedNum = int(self.size*self.size*ratio)

        for i in range(riggedNum):
            while True:
                x = random.randrange(0, 10)
                y = random.randrange(0, 10)
                if self.visionGrid[y][x] == 0:
                    self.visionGrid[y][x] = 1
                    break
            
    def placeWall(self, positioning, coordx, coordy):
        if self.wallGrid[coordy][coordx] == "V" or self.wallGrid[coordy][coordx] == "H":
            return -1
        
        elif positioning == "V":
            if coordy > 0 and self.wallGrid[coordy - 1][coordx] == "V":
                return -1
            
            elif coordy < 8 and self.wallGrid[coordy + 1][coordx] == "V":
                return -1

            else:
                newWallGrid = copy.deepcopy(self.wallGrid)
                newWallGrid[coordy][coordx] = "V"
                if self.canMove(newWallGrid) == 1:
                    self.wallGrid[coordy][coordx] = "V"
                    return 1
                else:
                    #print(-1)
                    return -1

        else:
            if coordx > 0 and self.wallGrid[coordy][coordx - 1] == "H":
                return -1

            elif coordx < 8 and self.wallGrid[coordy][coordx + 1] == "H":
                return -1

            else:
                newWallGrid = copy.deepcopy(self.wallGrid)
                newWallGrid[coordy][coordx] = "H"
                if self.canMove(newWallGrid) == 1:
                    self.wallGrid[coordy][coordx] = "H"
                    return 1
                else:
                    #print(-1)
                    return -1


    def moveEntity(self, item, direction):
        if item == "player":
            itemPos = self.playerPos
        else:
            itemPos = self.dronePos

        # Check for walls

        if direction == "left":
            indexChange = -1
            moveAmount = -1
            direction = "h"
        elif direction == "right":
            indexChange = 0
            moveAmount = 1
            direction = "h"
        elif direction == "up":
            indexChange = -1
            moveAmount = -1
            direction = "v"
        elif direction == "down":
            indexChange = 0
            moveAmount = 1
            direction = "v"
        moved = False
        if direction == "h":
            
            if not (itemPos[0] + moveAmount < 0 or itemPos[0] + moveAmount > 9):
                
                if itemPos[0] == 0 or self.wallGrid[itemPos[0] + indexChange][itemPos[1] - 1] != "V":
                    
                    if itemPos[1] >= 9 or self.wallGrid[itemPos[0] + indexChange][itemPos[1]] != "V":
                        
                        itemPos[0] += moveAmount
                        moved = True
        elif direction == "v":
            if not (itemPos[1] + moveAmount < 0 or itemPos[1] + moveAmount > 9):
                if itemPos[1] == 0 or self.wallGrid[itemPos[0] - 1][itemPos[1] + indexChange] != "H":
                    if itemPos[0] >= 9 or self.wallGrid[itemPos[0]][itemPos[1] + indexChange] != "H":
                        itemPos[1] += moveAmount
                        moved = True
        return moved
    

    def checkVision(self):
        x, y = self.playerPos
        for xi in range(self.size):
            for yi in range(self.size):
                if self.visionGrid[yi][xi] == 3:
                    self.visionGrid[yi][xi] = 2
                elif self.visionGrid[yi][xi] == 2:
                    self.visionGrid[yi][xi] = 1

        if self.visionGrid[y][x] == 1:
            self.visionGrid[y][x] = 3
        distancex = self.dronePos[0] - x
        if distancex < 0:
            distancex = distancex * -1
        distancey = self.dronePos[1] - y
        if distancey < 0:
            distancey = distancey * -1
        lookGrid = self.visionGrid[:]
        if distancex < 3 and distancey < 3:
            lookGrid[y][x] = 3
        finalGrid = []
        for y, things in enumerate(lookGrid):
            for x, thing in enumerate(things):
                finalGrid.append(thing)
        return finalGrid
    def getVision(self):
        obs = [self.roll]+self.dronePos[:]
        obs = obs+self.checkVision()
        wallH = [0 for i in range((self.size-1) * (self.size-1))]
        wallV = [0 for i in range((self.size-1) * (self.size-1))]
        for y, things in enumerate(self.wallGrid):
            for x, thing in enumerate(things):
                if thing == "H":
                    wallH[(self.size-1)*y+x] = 1
                elif thing == "V":
                    wallV[(self.size-1)*y+x] = 1
        obs = np.array(obs+wallH+wallV)
        #print(obs)
        #print(obs.shape)
        return obs
    def pathFind(self, visionGrid, wallGrid, theSpot):
        returnGrid = [[99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99, 99, 99]]
        
        returnGrid[theSpot[0]][theSpot[1]] = 0

        availables = [theSpot]

        while len(availables) > 0:
            theInfo = self.around(availables[0], wallGrid, returnGrid, returnGrid[availables[0][0]][availables[0][1]])
            returnGrid = theInfo[0]
            for i in theInfo[1]:
                if i not in availables:
                    availables.append(i)
            del(availables[0])
            #print(availables)
        return returnGrid

    def around(self, fromHere, wallGrid, givenGrid, distance):
        newSpace = []
        a = fromHere[0]
        b = fromHere[1]
        if 10 > b > -1 and 10 > a > -1:
            if wallGrid[self.coordFix(a - 1)][self.coordFix(b - 1)] != "V" and wallGrid[self.coordFix(a)][self.coordFix(b - 1)] != "V":
                if b - 1 > -1:
                    if givenGrid[a][b - 1] > distance + 1:
                        givenGrid[a][b - 1] = distance + 1
                        newSpace.append([a, b - 1])
            if wallGrid[self.coordFix(a - 1)][self.coordFix(b)] != "V" and wallGrid[self.coordFix(a)][self.coordFix(b)] != "V":
                if b + 1 < 10:
                    if givenGrid[a][b + 1] > distance + 1:
                        givenGrid[a][b + 1] = distance + 1
                        newSpace.append([a, b + 1])

            if wallGrid[self.coordFix(a - 1)][self.coordFix(b - 1)] != "H" and wallGrid[self.coordFix(a - 1)][self.coordFix(b)] != "H":
                if a - 1 > -1:
                    if givenGrid[a - 1][b] > distance + 1:
                        givenGrid[a - 1][b] = distance + 1
                        newSpace.append([a - 1, b])
            if wallGrid[self.coordFix(a)][self.coordFix(b - 1)] != "H" and wallGrid[self.coordFix(a)][self.coordFix(b)] != "H":
                if a + 1 < 10:
                    if givenGrid[a + 1][b] > distance + 1:
                        givenGrid[a + 1][b] = distance + 1
                        newSpace.append([a + 1, b])

        givenGrid2 = givenGrid[:]
        
        return givenGrid2, newSpace

    def coordFix(self, theNumber):
        if theNumber < 0:
            return 0
        elif theNumber > 8:
            return 8
        else:
            return theNumber
    def canMove(self, wallGrid):
        if self.pathFind(self.visionGrid, wallGrid, [self.dronePos[1], self.dronePos[0]])[self.playerPos[1]][self.playerPos[0]] == 99:
            return -1
        else:
            return 1
    def playerBestMove(self):
        distGrid = self.pathFind(self.visionGrid, self.wallGrid, self.dronePos)[:]
        x, y = self.playerPos
        if y > 0:
            up = distGrid[x][y-1]
        else:
            up = 99
        if y < 9:
            down = distGrid[x][y+1]
        else:
            down = 99
        if x > 0:
            left = distGrid[x-1][y]
        else:
            left = 99
        if x < 9:
            right = distGrid[x+1][y]
        else:
            right = 99
        if up < down and up < left and up < right:
            #print(up)
            return "up"
        elif down < left and down < right:
            #print(down)
            return "down"
        elif left < right:
            #print(left)
            return "left"
        else:
            #print(right)
            return "right"
    def step(self, action):
        #print(action)
        
        reward = 0
        done = False
        moved = True
        if self.roll == 0 and action > 4:
            action = 0
            reward -= 2
        elif self.roll == 1 and 5 > action > 86:
            action = 0
            reward -= 2
        elif self.roll == 2 and 87 > action:
            action = 0
            reward -= 2
        if action == 1:
            moved = self.moveEntity("drone", "up")
            reward += 2
        elif action == 2:
            moved = self.moveEntity("drone", "down")
            reward += 2
        elif action == 3:
            moved = self.moveEntity("drone", "left")
            reward += 2
        elif action == 4:
            moved = self.moveEntity("drone", "right")
            reward += 2
        if not moved:
            reward -= 2
        if action > 4:
            i = (action-5)//82
            action2 = (action-5)%82
            if i == 0:
                orientation = "H"
            else:
                orientation = "V"
            if action2 != 81:
                walled = self.placeWall(orientation, action2%9, action2//9)
                if walled == -1:
                    reward -= 2
                else:
                    reward += 2
                    
        if self.turn >= 300:
            done = True
            #reward += 100
        self.turn += 1
        if self.CheckWin(self.playerPos[0], self.playerPos[1], self.dronePos[0], self.dronePos[1]):
            done = True
            reward -= 10
        if not done and self.roll == 2:
            
            self.moveEntity("player", self.playerBestMove())
            if self.CheckWin(self.playerPos[0], self.playerPos[1], self.dronePos[0], self.dronePos[1]):
                done = True
                reward -= 10
            if not done:
                self.moveEntity("player", self.playerBestMove())
                if self.CheckWin(self.playerPos[0], self.playerPos[1], self.dronePos[0], self.dronePos[1]):
                    done = True
                    reward -= 10
                if not done:
                    reward += 0.5
        #print(self.playerPos)
        info = {}
        #print(done)
        if self.roll != 2:
            self.roll += 1
        else:
            self.roll = 0
        return self.getVision(), reward, done, info


    def render(self, **kargs):
        pass
    def reset(self):
        self.level = random.randrange(1, 6)
        self.wipeAll()
        return self.getVision()

env = envGrid()
#print(env.action_space.sample())
'''episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = env.action_space.sample()
        #print(action)
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
'''
env = envGrid()
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow
states = env.observation_space.shape
actions = 169
#actions2 = env.action_space.shape
#print(actions)
print(states)

def build_model(states, actions):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=states))
    model.add(Dense(32, activation='relu'))
    #model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.add(tensorflow.keras.layers.Reshape((169,)))
    return model

#del model

model = build_model(states, actions)
#model.load_weights('dyerM')
print(model.summary())



def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#dqn.load_weights('dqn_weights.h5f')

dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
dqn.save_weights('dqn_weights2.h5f', overwrite=True)
model.save_weights('dyerM2')
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
dqn.save_weights('dqn_weights2.h5f', overwrite=True)
model.save_weights('dyerM2')
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
dqn.save_weights('dqn_weights2.h5f', overwrite=True)
model.save_weights('dyerM2')
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
dqn.save_weights('dqn_weights2.h5f', overwrite=True)
model.save_weights('dyerM2')
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))

_ = dqn.test(env, nb_episodes=15, visualize=True)

dqn.save_weights('dqn_weights2.h5f', overwrite=True)
model.save_weights('dyerM2')
print(model.predict(np.array([env.getVision()])))
print(dqn.forward(env.getVision()))

