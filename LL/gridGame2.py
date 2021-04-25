import pygame, pygame.font, pygame.event, pygame.draw
import os
import random
import math
import random
import pygame
import time
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow
from gym import Env
from gym.spaces import MultiDiscrete, Box, Discrete
import numpy as np
import random
import copy


states = (1, 265)
actions = 169
def build_model(states, actions):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=states))
    model.add(Dense(64, activation='relu'))
    #model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.add(tensorflow.keras.layers.Reshape((169,)))
    return model

'''def build_model(states, actions):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=states))
    model.add(Dense(32, activation='relu'))
    #model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.add(tensorflow.keras.layers.Reshape((169,)))
    return model'''

model = build_model(states, actions)
print(model.summary())
'''def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn'''
#model.load_weights('dyerM')


#dyer = build_agent(model, actions)
#dyer.compile(Adam(lr=1e-3), metrics=['mae'])
#dyer.load_weights('dqn_weights.h5f')
model.load_weights('dyerM3')
print(model.summary())



def doMouseStuff():
    mousePos = pygame.mouse.get_pos()
    quadrant = [0, 0]
    for x in range(2):
        quadrant[x] = math.floor(mousePos[x] / 100)

    if quadrant[0] > 8:
        quadrant[0] = 8
    if quadrant[1] > 8:
        quadrant[1] = 8
    return quadrant



class grid(object):
    def __init__(self, size, playerPos = [random.randrange(1, 9), 9], dronePos = [random.randrange(1, 9), 0]):
        self.size = size
        
        self.visionGrid = [[0 for ii in range(self.size)] for i in range(self.size)]
        # 0 = nothing, 1 = rigged, 2 = player was there, 3 = player is there, this is what ai sees

        self.playerGrid = [[0 for ii in range(self.size)] for i in range(self.size)]
        # 0 = nothing, 1 = player, 2 = drone
        
        self.playerPos = playerPos
        self.dronePos = dronePos

        self.playerGrid[self.playerPos[0]][self.playerPos[1]] = 1
        self.playerGrid[self.dronePos[0]][self.dronePos[1]] = 2
        
        self.wallGrid = [[0 for ii in range(self.size-1)] for i in range(self.size-1)]
        self.level = 1
        self.scoreList = [0,0,0,0,0]
        self.score = 1000
        self.turn = 0
        self.roll = 0
        
    def wipeAll(self):
        self.wallGrid = [[0 for ii in range(self.size-1)] for i in range(self.size-1)]
        self.visionGrid = [[0 for ii in range(self.size)] for i in range(self.size)]
        self.playerGrid = [[0 for ii in range(self.size)] for i in range(self.size)]
        self.wallGrid = [[0 for ii in range(self.size-1)] for i in range(self.size-1)]
        self.visionGrid = [[0 for ii in range(self.size)] for i in range(self.size)]
        self.dronePos = [random.randrange(1, 9), 0]
        self.playerPos = [random.randrange(1, 9), 9]
        self.turn = 0
        self.roll = 0
        #self.level += 1
        
    def CheckWin(self, FirstCoords, SecondCoords):
        #Checks to see if the palyer is on the drone.
        if FirstCoords == SecondCoords:
            return True

        else:
            return False

    def Win(self):
        global finalScore
        if self.level == 5:
            
            self.scoreList[self.level - 1] = self.score
            print("Yay?")
            print(self.scoreList)
            for i in range(5):
                finalScore += mygrid.scoreList[i]
            return False

        else:
            self.level += 1
            self.wipeAll()
            self.rigCells()
            playerPos = [random.randrange(1, 9), 9]
            dronePos = [random.randrange(1, 9), 0]
            self.playerPos = playerPos
            self.dronePos = dronePos
            self.scoreList[self.level - 2] = self.score
            self.score = 1000
            self.playerGrid[self.playerPos[0]][self.playerPos[1]] = 1
            self.playerGrid[self.dronePos[0]][self.dronePos[1]] = 2
            return True

    def Lose(self):
        self.scoreList[self.level - 1] = self.score
        #print("Lose")
        #print(self.scoreList)
        
    def rigCells(self):
        ratio = self.level * 0.2
        riggedNum = self.size*self.size*ratio

        for i in range(int(riggedNum)):
            while True:
                x = random.randrange(0, 10)
                y = random.randrange(0, 10)
                if self.visionGrid[y][x] == 0:
                    self.visionGrid[y][x] = 1
                    break

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
            if not (itemPos[0] + moveAmount < 0 or itemPos[0] + moveAmount > 9):  # Check if will collide with wall
                if itemPos[1] == 0 or self.wallGrid[itemPos[1] - 1][itemPos[
                                                                        0] + indexChange] != "V":  # Has to be against top wall or node above moving thing can't be vertical
                    if itemPos[1] == 9 or self.wallGrid[itemPos[1]][itemPos[
                                                                        0] + indexChange] != "V":  # Has to be against bottom wall or node below can't be vertical
                        itemPos[0] += moveAmount
                        moved = True
        elif direction == "v":
            if not (itemPos[1] + moveAmount < 0 or itemPos[1] + moveAmount > 9):
                if itemPos[0] == 0 or self.wallGrid[itemPos[1] + indexChange][itemPos[0] - 1] != "H":
                    if itemPos[0] == 9 or self.wallGrid[itemPos[1] + indexChange][itemPos[0]] != "H":
                        itemPos[1] += moveAmount   
                        moved = True
        return moved  

    def canMoveEntity(self, item, direction):
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
            if not (itemPos[0] + moveAmount < 0 or itemPos[0] + moveAmount > 9):  # Check if will collide with wall
                if itemPos[1] == 0 or self.wallGrid[itemPos[1] - 1][itemPos[
                                                                        0] + indexChange] != "V":  # Has to be against top wall or node above moving thing can't be vertical
                    if itemPos[1] == 9 or self.wallGrid[itemPos[1]][itemPos[
                                                                        0] + indexChange] != "V":  # Has to be against bottom wall or node below can't be vertical
                        #itemPos[0] += moveAmount
                        moved = True
        elif direction == "v":
            if not (itemPos[1] + moveAmount < 0 or itemPos[1] + moveAmount > 9):
                if itemPos[0] == 0 or self.wallGrid[itemPos[1] + indexChange][itemPos[0] - 1] != "H":
                    if itemPos[0] == 9 or self.wallGrid[itemPos[1] + indexChange][itemPos[0]] != "H":
                        #itemPos[1] += moveAmount   
                        moved = True
        return moved  
    def droneBestMove(self):
        moved = False
        distGrid = self.pathFind(self.visionGrid, self.wallGrid, [self.playerPos[1], self.playerPos[0]])[:]
        print(distGrid)
        x, y = self.dronePos
        if y > 0:
            up = distGrid[y-1][x]
        else:
            up = 99
        if y < 9:
            down = distGrid[y+1][x]
        else:
            down = 99
        if x > 0:
            left = distGrid[y][x-1]
        else:
            left = 99
        if x < 9:
            right = distGrid[y][x+1]
        else:
            right = 99
        while not moved:
            if up < down and up < left and up < right:
                
                moved = self.canMoveEntity("drone", "up")
                if moved:
                    print("up")
                    return "up"
                else:
                    moved = False
                    up = 99
            elif down < left and down < right:
                
                moved = self.canMoveEntity("drone", "down")
                print(moved)
                if moved:
                    print("down")
                    return "down"
                else:
                    moved = False
                    down = 99
            elif left < right:
                
                moved = self.canMoveEntity("drone", "left")
                if moved:
                    print("left")
                    return "left"
                else:

                    moved = False
                    left = 99
            else:
                
                moved = self.canMoveEntity("drone", "right")
                if moved:
                    print("right")
                    return "right"
                else:
                    moved = False
                    right = 99
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
        obs = np.array([[obs+wallH+wallV]])
        #print(obs)
        #print(obs.shape)
        return obs
    def canMove(self, wallGrid):
        if self.pathFind(self.visionGrid, wallGrid, (self.dronePos[1], self.dronePos[0]))[self.playerPos[1]][self.playerPos[0]] == 99:
            return -1
        else:
            return 1
    def playerBestMove(self):
        moved = false
        distGrid = self.pathFind(self.visionGrid, self.wallGrid, [self.dronePos[1], self.dronePos[0]])[:]
        x, y = self.playerPos
        if y > 0:
            up = distGrid[y-1][x]
        else:
            up = 99
        if y < 9:
            down = distGrid[y+1][x]
        else:
            down = 99
        if x > 0:
            left = distGrid[y][x-1]
        else:
            left = 99
        if x < 9:
            right = distGrid[y][x+1]
        else:
            right = 99
        while not moved:
            if up < down and up < left and up < right:
                #print(up)
                moved = self.canMoveEntity("player", "up")
                if moved:
                    return "up"
                else:
                    moved = False
                    up = 99
            elif down < left and down < right:
                #print(down)
                moved = self.canMoveEntity("player", "down")
                if moved:
                    return "down"
                else:
                    moved = False
                    down = 99
            elif left < right:
                #print(left)
                moved = self.canMoveEntity("player", "left")
                if moved:
                    return "left"
                else:
                    moved = False
                    left = 99
            else:
                #print(right)
                moved = self.canMoveEntity("player", "right")
                if moved:
                    return "right"
                else:
                    moved = False
                    right = 99
    def step(self, action2):
        action = np.argmax(action2[0])
        
        print(action)
        if self.roll == 0 and action > 4:
            action = 0

        elif self.roll == 1 and 5 > action > 86:
            action = 0

        elif self.roll == 2 and 87 > action:
            action = 0

        if action == 1:
            moved = self.moveEntity("drone", "up")
        elif action == 2:
            moved = self.moveEntity("drone", "down")
        elif action == 3:
            moved = self.moveEntity("drone", "left")
        elif action == 4:
            moved = self.moveEntity("drone", "right")
            #print(moved)

        if action > 4:
            i = (action-5)//82
            action2 = (action-5)%82
            if i == 0:
                orientation = "H"
            else:
                orientation = "V"
            if action2 != 81:
                walled = self.placeWall(orientation, action2%9, action2//9)

        if self.roll != 2:
            self.roll += 1
        else:
            self.roll = 0
        return


pygame.init()
mygrid = grid(10)
#PlayerImage = pygame.image.load("Yas.jpg")
#Yas = pygame.transform.scale(PlayerImage, (225, 150))
defaultFont = pygame.font.SysFont('ariel', 150)
scoreFont = pygame.font.SysFont('ariel', 125)
titleFont = pygame.font.SysFont('ariel', 210)
smallFont = pygame.font.SysFont('ariel', 40)

titleImage = pygame.image.load("TitlePage.jpg")
titlePic = pygame.transform.scale(titleImage, (1600, 1000))
mazeImage = pygame.image.load("Maze.jpg")
mazePic = pygame.transform.scale(mazeImage, (600, 1000))

BLACK = (0,0,0)
WHITE = (255,255,255)
GREY = (177,177,177)
RED = (255,0,0)
BLUE = (0,0,255)
LIGHTBLUE = (0,255,255)
GREEN = (0,255,0)
LIGHTGREEN = (177, 255, 177)
YELLOW = (255,255,0)
MAGENTA = (255,0,255)
ORANGE = (255, 140, 0)

gameStage = 0
gameMode = 0
colorMode = [(0,255,255), (177, 255, 177)]
wallPlacementDirection = "V"

#0 empty
#1 rigged
#2 was there
#3 is there
    
def drawButton(color, xy, text = ""):
    pygame.draw.rect (win, color, xy)
    pygame.draw.rect (win, BLACK, xy, 2)
    theText = defaultFont.render (text, True, BLACK)
    win.blit (theText, (xy[0] + (xy[2] - theText.get_width())//2, (xy[1]*2+xy[3])//2 - 12))

def readInstructions(): 
    fi = open('Instructions.txt', 'r')
    lines = ""
    for line in range(27):
        iLine = (fi.readline().strip())
        lines = lines+"|"+iLine
    return lines

def showInstructions():
    text = readInstructions().split("|")
    for i in range(len(text)):
        instructionLine = smallFont.render(text[i], True, BLACK)
        win.blit(instructionLine, (20,(80 + (30*i))))
        
def redrawWin(playerPos):
    global gameStage
    global finalScore
    global gameMode
    global wallPlacementDirection
    global MovesTotal
    if gameStage == 0: #Menu
        win.blit (titlePic, (0,0))
        #win.blit(scoreFont.render("The Living Labyrinth", True, RED), (250,0))
        drawButton (LIGHTGREEN, (200, 350, 600, 180), "Start")
        drawButton (YELLOW, (200, 600, 600, 180), "How to Play")
        
        pygame.draw.rect (win, colorMode[gameMode], (200, 850, 600, 100))
        pygame.draw.rect (win, BLACK, (200, 850, 600, 100), 2)
        theText = smallFont.render ("Toggle Game Mode", True, BLACK)
        win.blit (theText, (355, 880))
        
        title1 = titleFont.render ("The Living Labyrinth", True, ORANGE)
        win.blit (title1, (50,100))
        pygame.display.update()
    elif gameStage == 1: #How to Play
        win.blit (titlePic, (0,0))
        pygame.draw.rect (win, GREY, (10, 100, 1200, 825))
        pygame.draw.rect (win, RED, (30, 30, 100, 50))
        pygame.draw.rect (win, BLACK, (30, 30, 100, 50), 2)
        theText = smallFont.render("Back", True, BLACK)
        win.blit (theText, (40, 45))
        showInstructions()
        pygame.display.update()
    elif gameStage == 2: #the Game
        if gameMode == 0:
            pygame.draw.rect(win, BLACK, (0, 0, 1000, 1000))
            
            for i in range(len(mygrid.visionGrid)):
                for j in range(len(mygrid.visionGrid[i])):
                    
                    if mygrid.playerGrid[i][j] == 2:
                        
                        pygame.draw.rect (win, WHITE, (mygrid.playerPos[0]*100 - 100, mygrid.playerPos[1]*100 - 100, 300, 300))
                        pygame.draw.rect (win, YELLOW, (mygrid.dronePos[0]*100, mygrid.dronePos[1]*100, 100, 100))
                        pygame.draw.rect (win, MAGENTA, (mygrid.playerPos[0]*100, mygrid.playerPos[1]*100, 100, 100))
                        
            for i in range(len(mygrid.wallGrid)):
                for j in range(len(mygrid.wallGrid[i])):
                    if mygrid.wallGrid[i][j] == "V":
                        pygame.draw.rect(win, BLACK, ((j * 100) + 90, i * 100, 20, 200))
                    elif mygrid.wallGrid[i][j] == "H":
                        pygame.draw.rect(win, BLACK, (j * 100, (i * 100) + 90, 200, 20))
                        
            win.blit (mazePic, (1000,0))
            
            pygame.draw.rect (win, RED, (1450, 900, 100, 50))
            pygame.draw.rect (win, BLACK, (1450, 900, 100, 50), 2)
            theText = smallFont.render("Back", True, BLACK)
            win.blit (theText, (1460, 915))
            
            win.blit(defaultFont.render(str(mygrid.level), True, RED), (1000, 0))
            win.blit(scoreFont.render("Score: " + str(mygrid.score), True, RED), (1000, 100))
        else:
            win.fill(WHITE)
            for i in range(len(mygrid.visionGrid)):
                for j in range(len(mygrid.visionGrid[i])):

                    if mygrid.playerGrid[i][j] == 1:
                        pygame.draw.rect(win, MAGENTA, (mygrid.playerPos[0] * 100, mygrid.playerPos[1] * 100, 100, 100))

                    elif mygrid.playerGrid[i][j] == 2:
                        pygame.draw.rect(win, YELLOW, (mygrid.dronePos[0] * 100, mygrid.dronePos[1] * 100, 100, 100))

            for i in range(len(mygrid.wallGrid)):
                for j in range(len(mygrid.wallGrid[i])):
                    if mygrid.wallGrid[i][j] == "V":
                        pygame.draw.rect(win, BLACK, ((j * 100) + 90, i * 100, 20, 200))
                    elif mygrid.wallGrid[i][j] == "H":
                        pygame.draw.rect(win, BLACK, (j * 100, (i * 100) + 90, 200, 20))
            if wallPlacementDirection == "V":
                pygame.draw.rect(win, RED, ((mouseQuadrant[0] * 100) + 90, mouseQuadrant[1] * 100, 20, 200))
            else:
                pygame.draw.rect(win, RED, (mouseQuadrant[0] * 100, (mouseQuadrant[1] * 100) + 90, 200, 20))

            win.blit (mazePic, (1000,0))
            
            pygame.draw.rect (win, RED, (1450, 900, 100, 50))
            pygame.draw.rect (win, BLACK, (1450, 900, 100, 50), 2)
            theText = smallFont.render("Back", True, BLACK)
            win.blit (theText, (1460, 915))

            pygame.draw.rect (win, RED, (1200, 500, 200, 100))
            pygame.draw.rect (win, BLACK, (1200, 500, 200, 100), 2)
            theText = smallFont.render("End Turn", True, BLACK)
            win.blit (theText, (1250, 520))

        pygame.display.update()
                
    elif gameStage == 3: #Victory counting
        win.blit (titlePic, (0,0))
        theText = titleFont.render("You Escaped", True, MAGENTA)
        win.blit (theText, (100,100))
        theText = titleFont.render("the Labyrinth!", True, MAGENTA)
        win.blit (theText, (100,350))
        for i in range(0, abs(finalScore), 25):
            if finalScore < 0:
                win.blit (titlePic, (0,0))
                theText = titleFont.render("You Escaped", True, MAGENTA)
                win.blit (theText, (100,100))
                theText = titleFont.render("the Labyrinth!", True, MAGENTA)
                win.blit (theText, (100,350))

                theScore = scoreFont.render ("Your final score: " + str(-i), True, MAGENTA)
                win.blit (theScore, (100, 800))
        
                pygame.time.delay (1)
                pygame.display.update()
            elif finalScore > 0:
                win.blit (titlePic, (0,0))
                theText = titleFont.render("You Escaped", True, MAGENTA)
                win.blit (theText, (100,100))
                theText = titleFont.render("the Labyrinth!", True, MAGENTA)
                win.blit (theText, (100,350))

                theScore = scoreFont.render ("Your final score: " + str(i), True, MAGENTA)
                win.blit (theScore, (100, 800))
                
                pygame.time.delay (1)
                pygame.display.update()

        theScore = scoreFont.render ("Your final score: " + str(finalScore), True, MAGENTA)
        win.blit (theScore, (100, 800))
        gameStage = 4
        pygame.display.update()
        
    elif gameStage == 4: #1 Victory royale
        win.blit (titlePic, (0,0))
        theText = titleFont.render("You Escaped", True, MAGENTA)
        win.blit (theText, (100,100))
        theText = titleFont.render("the Labyrinth!", True, MAGENTA)
        win.blit (theText, (100,350))
        
        theScore = scoreFont.render ("Your final score: " + str(finalScore), True, MAGENTA)
        win.blit (theScore, (100, 800))

        pygame.draw.rect (win, RED, (1450, 900, 120, 50))
        pygame.draw.rect (win, BLACK, (1450, 900, 120, 50), 2)
        theText = smallFont.render("Restart", True, BLACK)
        win.blit (theText, (1460, 915))

        pygame.display.update()
        
    elif gameStage == 5: #1 Victory royale
        win.blit (titlePic, (0,0))
        theText = titleFont.render("They got you!", True, MAGENTA)
        win.blit (theText, (100,100))

        theScore = scoreFont.render ("You survived: " + str(MovesTotal) + " moves" , True, MAGENTA)
        win.blit (theScore, (100, 800))

        pygame.draw.rect (win, RED, (1450, 900, 120, 50))
        pygame.draw.rect (win, BLACK, (1450, 900, 120, 50), 2)
        theText = smallFont.render("Restart", True, BLACK)
        win.blit (theText, (1460, 915))

        pygame.display.update()

        

inPlay = True
win = pygame.display.set_mode((1600,1000))
playerPos = [0,0]
font1 = pygame.font.SysFont("arial", 20)
Turn = 1
Moves = 0
MovesTotal = 0
pygame.mixer.music.load('Sounds.mp3')
pygame.mixer.music.play(-1, 0.0)

finalScore = 0
wallPlaces = 0


while inPlay:
    mouseQuadrant = doMouseStuff()
    redrawWin(mygrid.playerPos)
    pygame.time.delay(20)

    if Turn == 2 and gameMode == 1: #Bot
        if Moves < 2:
            mygrid.moveEntity("drone", mygrid.droneBestMove())
            Moves += 1
        else:
            Turn = 1
            Moves = 0
        if mygrid.dronePos == mygrid.playerPos:
            gameStage = 5
    
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            inPlay = False

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                inPlay = False
                mygrid.Lose()
            if gameStage == 2:
                if gameMode == 0:
                    if Turn == 1:
                        Turn = 2
                        #print(mygrid.CheckWin(mygrid.playerPos, mygrid.dronePos))
                        mygrid.step(model.predict(mygrid.getVision()))
                        #print(mygrid.CheckWin(mygrid.playerPos, mygrid.dronePos))
                        mygrid.step(model.predict(mygrid.getVision()))
                        #print(mygrid.CheckWin(mygrid.playerPos, mygrid.dronePos))
                        mygrid.step(model.predict(mygrid.getVision()))
                        #wprint(mygrid.CheckWin(mygrid.playerPos, mygrid.dronePos))

                    else:
                        if Moves == 2:
                            Moves = 0
                            Turn = 1
                    if event.key == pygame.K_a:
                        mygrid.moveEntity("player", "left")

                        Moves += 1
                        MovesTotal += 1
                        mygrid.score = 1000 - MovesTotal * 25

                    elif event.key == pygame.K_s:
                        mygrid.moveEntity("player", "down")

                        Moves += 1
                        MovesTotal += 1
                        mygrid.score = 1000 - MovesTotal * 25

                    elif event.key == pygame.K_d:
                        mygrid.moveEntity("player", "right")

                        Moves += 1
                        MovesTotal += 1
                        mygrid.score = 1000 - MovesTotal * 25

                    elif event.key == pygame.K_w:       
                        mygrid.moveEntity("player", "up")

                        Moves += 1
                        MovesTotal += 1
                        mygrid.score = 1000 - MovesTotal * 25
           
                    winner = mygrid.CheckWin(mygrid.playerPos, mygrid.dronePos)
                    if winner == True:
                        
                        if not mygrid.Win():
                            gameStage = 3
                        Moves = 0
                        Turn = 1
                        MovesTotal = 0
                else: #gameMode == 1

                    if Turn == 1: #Player
                        if Moves == 0:
                            if event.key == pygame.K_a:
                                mygrid.moveEntity("player", "left")

                                Moves += 1
                                MovesTotal += 1
                                mygrid.score = 1000 - MovesTotal * 25

                            elif event.key == pygame.K_s:
                                mygrid.moveEntity("player", "down")

                                Moves += 1
                                MovesTotal += 1
                                mygrid.score = 1000 - MovesTotal * 25

                            elif event.key == pygame.K_d:
                                mygrid.moveEntity("player", "right")

                                Moves += 1
                                MovesTotal += 1
                                mygrid.score = 1000 - MovesTotal * 25

                            elif event.key == pygame.K_w:       
                                mygrid.moveEntity("player", "up")

                                Moves += 1
                                MovesTotal += 1
                                mygrid.score = 1000 - MovesTotal * 25
           
                    winner = mygrid.CheckWin(mygrid.playerPos, mygrid.dronePos)
                    if winner == True:
                        
                        if not mygrid.Win():
                            gameStage = 3
                        Moves = 0
                        Turn = 2
                        MovesTotal = 0

                
        if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEWHEEL:
            clickPos = pygame.mouse.get_pos()
            
            if gameStage == 0: #Menu
                if 200 <= clickPos[0] <= 800 and 350 <= clickPos[1] <= 530:
                    gameStage = 2
                elif 200 <= clickPos[0] <= 800 and 600 <= clickPos[1] <= 780:
                    gameStage = 1
                elif 200 <= clickPos[0] <= 800 and 850 <= clickPos[1] <= 950:
                    if gameMode == 0:
                        gameMode = 1
                    elif gameMode == 1:
                        gameMode = 0

            elif gameStage == 1: #How to Play
                if 30 <= clickPos[0] <= 130 and 30 <= clickPos[1] <= 80:
                    gameStage = 0

            elif gameStage == 2:
                if gameMode == 0:
                    if 1450 <= clickPos[0] <= 1550 and 900 <= clickPos[1] <= 950:
                        gameStage = 0
                        finalScore = 0
                        mygrid.level = 1
                        mygrid.wipeAll()
                        mygrid.rigCells()
                        mygrid.playerPos = [random.randrange(1, 9), 9]
                        mygrid.dronePos = [random.randrange(1, 9), 0]
                        mygrid.scoreList = [0, 0, 0, 0, 0]
                        mygrid.score = 1000
                        mygrid.playerGrid[mygrid.playerPos[0]][mygrid.playerPos[1]] = 1
                        mygrid.playerGrid[mygrid.dronePos[0]][mygrid.dronePos[1]] = 2
                
                elif gameMode == 1:
                    if 1450 <= clickPos[0] <= 1550 and 900 <= clickPos[1] <= 950:
                        gameStage = 0
                        finalScore = 0
                        mygrid.level = 1
                        mygrid.wipeAll()
                        mygrid.rigCells()
                        mygrid.playerPos = [random.randrange(1, 9), 9]
                        mygrid.dronePos = [random.randrange(1, 9), 0]
                        mygrid.scoreList = [0, 0, 0, 0, 0]
                        mygrid.score = 1000
                        mygrid.playerGrid[mygrid.playerPos[0]][mygrid.playerPos[1]] = 1
                        mygrid.playerGrid[mygrid.dronePos[0]][mygrid.dronePos[1]] = 2
                        wallPlaces = 0
                    elif wallPlaces < 2:
                        if event.type == pygame.MOUSEWHEEL:
                            if wallPlacementDirection == "V":
                                wallPlacementDirection = "H"
                            else:
                                wallPlacementDirection = "V"
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            if event.button == 1:
                                mygrid.placeWall(wallPlacementDirection, mouseQuadrant[0], mouseQuadrant[1])
                                wallPlaces += 1
                    elif 1200 <= clickPos[0] <= 1400 and 500 <= clickPos[1] <= 600:
                        Turn = 2
                        wallPlaces = 0
                        Moves = 0
                        
            elif gameStage == 4:
                if 1450 <= clickPos[0] <= 1570 and 900 <= clickPos[1] <= 970:
                    gameStage = 0
                    finalScore = 0
                    mygrid.level = 1
                    mygrid.wipeAll()
                    mygrid.rigCells()
                    mygrid.playerPos = [random.randrange(1, 9), 9]
                    mygrid.dronePos = [random.randrange(1, 9), 0]
                    mygrid.scoreList = [0, 0, 0, 0, 0]
                    mygrid.score = 1000
                    mygrid.playerGrid[mygrid.playerPos[0]][mygrid.playerPos[1]] = 1
                    mygrid.playerGrid[mygrid.dronePos[0]][mygrid.dronePos[1]] = 2
                    
            elif gameStage == 5:
                if 1450 <= clickPos[0] <= 1570 and 900 <= clickPos[1] <= 970:
                    gameStage = 0
                    finalScore = 0
                    mygrid.level = 1
                    mygrid.wipeAll()
                    mygrid.rigCells()
                    mygrid.playerPos = [random.randrange(1, 9), 9]
                    mygrid.dronePos = [random.randrange(1, 9), 0]
                    mygrid.scoreList = [0, 0, 0, 0, 0]
                    mygrid.score = 1000
                    mygrid.playerGrid[mygrid.playerPos[0]][mygrid.playerPos[1]] = 1
                    mygrid.playerGrid[mygrid.dronePos[0]][mygrid.dronePos[1]] = 2

pygame.mixer.music.stop()
pygame.quit()




