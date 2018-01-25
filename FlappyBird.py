# Imports
import pygame
from random import randint,shuffle
import numpy as np
from math import pi, asin, sqrt
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv

# Enums
class Direction:
    left, right, up, down = range(4)
class NodeType:
    empty, snake, food, wall = range(4)

# Screen constants
block_size = 10
screen_size = (50,50)
screen_color = (0, 0, 0)
wall_color = (128, 128, 128)
snake_color = (0, 255, 255)
food_color = (0, 255, 0)

# Grid constants
columns, rows = screen_size[0], screen_size[1];

# Snake constants
snake_initial_size = 1
snake_position = (10,screen_size[1]/2)

# Wall constants
wallWidth = 3

class SnakeNode:
    def __init__(self,x,y):
        self.x = x
        self.y = y

def getGrid():
    grid = [[0 for x in range(columns)] for y in range(rows)]
    return grid

def getFlappyBirdNodes(x,y,grid,total_gravitational_force):
    # Create initial snake
    snake_nodes = []
    for i in range(snake_initial_size):
        y += total_gravitational_force
        segment = SnakeNode(x+i, y)
        snake_nodes.append(segment)
        grid[x+i][y] = NodeType.snake

    return snake_nodes

def drawNode(x,y,grid,screen):
    if grid[x][y] == NodeType.snake:  color = snake_color
    elif grid[x][y] == NodeType.food: color = food_color
    elif grid[x][y] == NodeType.wall: color = wall_color
    else:                             color = screen_color

    pygame.draw.rect(screen,color,pygame.Rect(x*block_size,y*block_size,block_size,block_size))

def isGameOver(snake_nodes,grid):
    head = snake_nodes[0]
    return grid[head.x][head.y] == NodeType.wall\
        or head.y == 0\
        or head.y == rows-1

def drawNodes(grid,screen):
    for x in range(columns):
        for y in range(rows):
            drawNode(x,y, grid,screen)

def neuralInputs(snake_nodes,wall_boundary_x,top_wall_boundary_y,bottom_wall_boundary_y):
    head = snake_nodes[0]

    wall_boundary_x = abs(head.x - wall_boundary_x)
    top_wall_boundary_y = abs(head.y - top_wall_boundary_y)
    bottom_wall_boundary_y = abs(head.y - bottom_wall_boundary_y)

    return wall_boundary_x,top_wall_boundary_y,bottom_wall_boundary_y

def getTrainedModel(data, labels):
    network = input_data(shape=[None, 4], name='input')
    network = fully_connected(network, 25, activation='relu')
    network = fully_connected(network, 25, activation='relu')
    network = fully_connected(network, 2, activation='linear')
    network = regression(network, optimizer='adam', learning_rate=1e-2, loss='mean_square', name='target')
    model = tflearn.DNN(network)

    model.fit(data, labels, n_epoch = 1, shuffle = True)
    return model

def getPredictedDirection(snake_nodes,model,inputs):
    head = snake_nodes[0]

    directions = [0,1]

    # shuffle(directions)

    for direction in directions:
        prediction = model.predict([[inputs[0],inputs[1],inputs[2],direction]])

        print(prediction)

        if np.argmax(prediction) == 1:
            break

    return direction

def getOutputForTraining(target_output,inputs,direction):

    return "\n{},{},{},{},{}".format(target_output,
                                        inputs[0],
                                        inputs[1],
                                        inputs[2],
                                        direction)

def getWalledGrid(grid,index):
    for i in range(index - wallWidth,index):
        for j in range(rows):
            if j < rows/2-5 or j > rows/2+5:
                grid[i][j] = NodeType.wall

    return grid,index,rows/2-5,rows/2+5

def runGame(death_count,font):

    # Game objects
    score_count = 0
    total_gravitational_force = 0
    screen = pygame.display.set_mode((screen_size[0]*block_size,
                                      screen_size[1]*block_size))
    wallIndex = columns
    grid = getGrid()
    snake_nodes = getFlappyBirdNodes(snake_position[0],
                                snake_position[1],
                                grid,
                                total_gravitational_force)

    # Game loop
    while not isGameOver(snake_nodes,grid):

        total_gravitational_force += 1
        grid = getGrid()
        snake_nodes = getFlappyBirdNodes(snake_position[0],
                                    snake_position[1],
                                    grid,
                                    total_gravitational_force)
        grid,wall_boundary_x,top_wall_boundary_y,bottom_wall_boundary_y = getWalledGrid(grid,wallIndex)
        inputs = neuralInputs(snake_nodes,wall_boundary_x,top_wall_boundary_y,bottom_wall_boundary_y)

        direction = getPredictedDirection(snake_nodes,model,inputs)
        # print(direction)
        total_gravitational_force -= (direction * 2)

        # Move the wall
        wallIndex -= 1
        if wallIndex <= 0:
            wallIndex = columns

        # If game is over, target output is -1
        # If snake has moved away from the goal, target output is 0
        # If snake has moved closer to the goal, target output is 1
        if isGameOver(snake_nodes,grid): target_output = 0
        # elif current_distance_between_snake_and_food >= previous_distance_between_snake_and_food:  target_output = 0
        else:                            target_output = 1

        output = getOutputForTraining(target_output,inputs,direction)
        # file = open("Data.csv","a")
        # file.write(output)
        # file.close()

        # Update score
        game_stats_label = font.render("Deaths: {}                    Score: {}".format(death_count,score_count), 1, (255,255,0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        # Drawing
        screen.fill(screen_color)
        drawNodes(grid,screen)
        screen.blit(game_stats_label, (0, 0))
        pygame.display.flip()

        # Clock ticking
        pygame.time.Clock().tick(999999999999)

        # Manual controls
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP] and direction!=Direction.down: direction = Direction.up
        elif pressed[pygame.K_DOWN] and direction!=Direction.up: direction = Direction.down

    death_count += 1
    runGame(death_count,font)

data,labels = load_csv("Data.csv",target_column=0,categorical_labels=True,n_classes=2)
model = getTrainedModel(data,labels)
death_count = 0
pygame.init()
font = pygame.font.SysFont("monospace", 50)
runGame(death_count,font)



























