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

# Wall constants
wallWidth = 3

class SnakeNode:
    def __init__(self,x,y):
        self.x = x
        self.y = y

def getGrid():
    grid = [[0 for x in range(columns)] for y in range(rows)]
    return grid

def getSnakeNodes(x,y,grid):
    # Create initial snake
    snake_nodes = []
    for i in range(snake_initial_size):
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

def isGameOver(snake_nodes):
    head = snake_nodes[0]
    return head.x == 0\
        or head.y == 0\
        or head.x == columns-1\
        or head.y == rows-1

def advanceSnake(snake_nodes,direction,grid):
    head = snake_nodes[0]
    tail = snake_nodes.pop()
    grid[tail.x][tail.y] = NodeType.empty
    
    if direction == Direction.up:
        tail.x = head.x
        tail.y = head.y - 1
    elif direction == Direction.down:
        tail.x = head.x
        tail.y = head.y + 1
    elif direction == Direction.left:
        tail.x = head.x - 1
        tail.y = head.y
    elif direction == Direction.right:
        tail.x = head.x + 1
        tail.y = head.y

    snake_nodes.insert(0,tail)

    if grid[tail.x][tail.y] != NodeType.food:
        grid[tail.x][tail.y] = NodeType.snake

    return snake_nodes

def drawNodes(grid,screen):
    for x in range(columns):
        for y in range(rows):
            drawNode(x,y, grid,screen)

def getNeighboringNodes(snake_nodes,direction,grid): # Left, forward, right nodes of snake
    head = snake_nodes[0]

    if direction == Direction.right:
        return (grid[head.x][head.y-1],grid[head.x+1][head.y],grid[head.x][head.y+1])
    elif direction == Direction.left:
        return (grid[head.x][head.y+1],grid[head.x-1][head.y],grid[head.x][head.y-1])
    elif direction == Direction.up:
        return (grid[head.x-1][head.y],grid[head.x][head.y-1],grid[head.x+1][head.y])
    else:
        return (grid[head.x+1][head.y],grid[head.x][head.y+1],grid[head.x-1][head.y])

def areNeighboringNodesBlocked(left,forward,right):
    return (int(left == NodeType.wall),int(forward == NodeType.wall),int(right == NodeType.wall))

def isAnyNeighboringNodesBlocked(left,forward,right):
    return left == NodeType.wall or forward == NodeType.wall or right == NodeType.wall

def distanceBetweenSnakeAndFood(snake_nodes,food_position):
    head = snake_nodes[0]

    food_x,food_y = food_position

    base = abs(food_x - head.x)
    perpendicular = abs(food_y - head.y)

    return base+perpendicular

def getOrthogonalAngle(snake_nodes,food_position):
    head = snake_nodes[0]

    food_x,food_y = food_position

    base = food_x - head.x
    perpendicular = food_y - head.y

    hypotenuse = sqrt(base**2 + perpendicular**2)+0.00001

    return asin(perpendicular/hypotenuse)/(pi/2)

def neuralInputs(snake_nodes,grid,absolute_direction,food_position):
    return (areNeighboringNodesBlocked(*getNeighboringNodes(snake_nodes,absolute_direction,grid)),
    round(getOrthogonalAngle(snake_nodes,food_position),3))

def getTrainedModel(data, labels):
    network = input_data(shape=[None, 5], name='input')
    network = fully_connected(network, 25, activation='relu')
    network = fully_connected(network, 25, activation='relu')
    network = fully_connected(network, 3, activation='linear')
    network = regression(network, optimizer='adam', learning_rate=1e-2, loss='mean_square', name='target')
    model = tflearn.DNN(network)

    model.fit(data, labels, n_epoch = 10, shuffle = True)
    return model

def getRelativeDirection(current_direction,next_direction):

    if current_direction == Direction.right:
        if next_direction == Direction.up: return -1
        elif next_direction == Direction.right: return 0
        else:                         return 1
    elif current_direction == Direction.left:
        if next_direction == Direction.down: return -1
        elif next_direction == Direction.left: return 0
        else:                         return 1
    elif current_direction == Direction.up:
        if next_direction == Direction.left: return -1
        elif next_direction == Direction.up: return 0
        else:                         return 1
    else:
        if next_direction == Direction.right: return -1
        elif next_direction == Direction.down: return 0
        else:                         return 1

def getPredictedDirection(snake_nodes,absolute_direction,model,inputs,grid):
    head = snake_nodes[0]

    relative_directions = [-1,0,1]

    shuffle(relative_directions)

    for relative_direction in relative_directions:
        prediction = model.predict([[inputs[0][0],inputs[0][1],inputs[0][2],inputs[1],relative_direction]])

        if np.argmax(prediction) == 1:
            break

    if absolute_direction == Direction.right:
        if relative_direction == -1:  return Direction.up,relative_direction
        elif relative_direction == 0: return Direction.right,relative_direction
        else:                         return Direction.down,relative_direction
    elif absolute_direction == Direction.left:
        if relative_direction == -1:  return Direction.down,relative_direction
        elif relative_direction == 0: return Direction.left,relative_direction
        else:                         return Direction.up,relative_direction
    elif absolute_direction == Direction.up:
        if relative_direction == -1:  return Direction.left,relative_direction
        elif relative_direction == 0: return Direction.up,relative_direction
        else:                         return Direction.right,relative_direction
    else:
        if relative_direction == -1:  return Direction.right,relative_direction
        elif relative_direction == 0: return Direction.down,relative_direction
        else:                         return Direction.left,relative_direction

def getOutputForTraining(target_output,inputs,snake_nodes,relative_direction):

    return "\n{},{},{},{},{},{}".format(target_output,
                                     inputs[0][0],
                                     inputs[0][1],
                                     inputs[0][2],
                                     inputs[1],
                                     relative_direction)

def generateFood(grid):
    food_position = (randint(1, columns-snake_initial_size-1),randint(1, rows-snake_initial_size-1))
    grid[food_position[0]][food_position[1]] = NodeType.food
    return food_position

def checkForFoodCollision(snake_nodes,grid):
    head = snake_nodes[0]
    return grid[head.x][head.y] == NodeType.food

def getWalledGrid(grid,index):
    for i in range(index - wallWidth,index):
        for j in range(rows):
            grid[i][j] = NodeType.wall

    return grid

def runGame(death_count,font):

    # Game objects
    directions = [Direction.right,Direction.left,Direction.up,Direction.down]
    direction = directions[randint(0,len(directions)-1)]
    snake_position = (randint(1, columns-snake_initial_size-1),randint(1, rows-snake_initial_size-1))
    screen = pygame.display.set_mode((screen_size[0]*block_size,
                                      screen_size[1]*block_size))

    wallIndex = columns

    # Game loop
    while not False:

        grid = getGrid()
        grid = getWalledGrid(grid,wallIndex)
        wallIndex -= 1
        if wallIndex <= 0:
            wallIndex = columns
        snake_nodes = getSnakeNodes(snake_position[0],
                                    snake_position[1],
                                    grid)
        food_position = generateFood(grid)

        # Update score
        death_count_label = font.render("Death count: {}".format(death_count), 1, (255,255,0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        # Drawing
        screen.fill(screen_color)
        drawNodes(grid,screen)
        screen.blit(death_count_label, (0, 0))
        pygame.display.flip()

        # Clock ticking
        pygame.time.Clock().tick(60)

        # Manual controls
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP] and direction!=Direction.down: direction = Direction.up
        elif pressed[pygame.K_DOWN] and direction!=Direction.up: direction = Direction.down

    death_count += 1
    runGame(death_count,font,model)

# data,labels = load_csv("Data.csv",target_column=0,categorical_labels=True,n_classes=3)
# model = getTrainedModel(data,labels)
death_count = 0
pygame.init()
font = pygame.font.SysFont("monospace", 50)
runGame(death_count,font)


























