# -*- coding: utf-8 -*-
"""
Recriação do jogo Sonda Lunar (Lunar Lander)

@author: Prof. Daniel Cavalcanti Jeronymo
"""

import numpy as np
import pygame
    
import sys
import traceback
import math
import random
import bisect
import pickle
import time
import copy
from neuralnetwork import layerDenseCreate, networkCalculate, networkDeserialize, networkSerialize, relu, sigmoid


class ImprovedNeuralController:
    def __init__(self):
        self.layers = [
            layerDenseCreate(8, 16, relu),  # Camada maior
            layerDenseCreate(16, 8, relu),   # Camada intermediária
            layerDenseCreate(8, 3, sigmoid)  # Camada de saída
        ]

    def get_controls(self, game_state, landscape):
        # Inputs mais informativos:
        inputs = np.array([
            game_state.landerx / GameConstants.screenWidth,
            game_state.landery / GameConstants.screenHeight,
            game_state.landerdx / 20,
            game_state.landerdy / 20,
            game_state.landerRotation / 180,
            # Distância até o local de pouso mais próximo
            self._distance_to_landing_spot(game_state, landscape),
            game_state.fuel / 500,
            # Altitude normalizada invertida (1 = no chão)
            1 - (game_state.landery / GameConstants.screenHeight)
        ])
        
        output = networkCalculate(inputs.reshape(1, -1), self.layers)[0]
        return output[0], output[1], output[2]  # left, thrust, right
    
    def _distance_to_landing_spot(self, game_state, landscape):
        landing_spots = [l for l in landscape if l.landingSpot]
        if not landing_spots:
            return 0.5  # Valor padrão se não houver locais de pouso
        
        min_dist = float('inf')
        for spot in landing_spots:
            spot_center = (spot.p0.x + spot.p1.x) / 2
            dist = abs(game_state.landerx - spot_center) / GameConstants.screenWidth
            if dist < min_dist:
                min_dist = dist
                
        return min_dist
class GameConstants:
    #                  R    G    B
    ColorWhite     = (255, 255, 255)
    ColorBlack     = (  0,   0,   0)
    ColorRed       = (255,   0,   0)
    ColorGreen     = (  0, 255,   0)
    ColorDarkGreen = (  0, 155,   0)
    ColorDarkGray  = ( 40,  40,  40)
    BackgroundColor = ColorBlack
    
    screenScale = 1
    screenWidth = screenScale*800
    screenHeight = screenScale*600
    
    randomSeed = 0
    terrainR = 1.0
    
    landerWidth = 20
    landerHeight = 20
    
    FPS = 30
    
    fontSize = 20
    
    gravity = 1.622*5 # moon gravity is 1.622m/s2, 5 is just a scale factor for the game
    thrustAcceleration = 2.124*gravity # engine thrust-to-weight ratio
    rotationFilter = 0.75 # higher is more difficult, must be in range [0,1]
    topSpeed = 0.35*1000 # not actually implemented but it's a good idea
    drag = 0.0003 # this is a wild guess - just a tiiiny bit of aerodynamics

class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        
    def as_tuple(self):
        return (self.x, self.y)

class Line:            
    def __init__(self, x0, y0, x1, y1, landingSpot):
        self.p0 = Point(x0, y0)
        self.p1 = Point(x1, y1)
        self.landingSpot = landingSpot
        
    def as_tuple(self):
        return (self.p0.as_tuple(), self.p1.as_tuple())
        
# Midpoint displacement algorithm for random terrain - better alternative to Perlin noise
# Source: https://bitesofcode.wordpress.com/2016/12/23/landscape-generation-using-midpoint-displacement/
# PS: this generates 2^num_of_iterations points
def midpoint_displacement(start, end, roughness, vertical_displacement=None,
                          num_of_iterations=16):
    """
    Given a straight line segment specified by a starting point and an endpoint
    in the form of [starting_point_x, starting_point_y] and [endpoint_x, endpoint_y],
    a roughness value > 0, an initial vertical displacement and a number of
    iterations > 0 applies the  midpoint algorithm to the specified segment and
    returns the obtained list of points in the form
    points = [[x_0, y_0],[x_1, y_1],...,[x_n, y_n]]
    """
    # Final number of points = (2^iterations)+1
    if vertical_displacement is None:
        # if no initial displacement is specified set displacement to:
        #  (y_start+y_end)/2
        vertical_displacement = (start[1]+end[1])/2
    # Data structure that stores the points is a list of lists where
    # each sublist represents a point and holds its x and y coordinates:
    # points=[[x_0, y_0],[x_1, y_1],...,[x_n, y_n]]
    #              |          |              |
    #           point 0    point 1        point n
    # The points list is always kept sorted from smallest to biggest x-value
    points = [start, end]
    iteration = 1
    while iteration <= num_of_iterations:
        # Since the list of points will be dynamically updated with the new computed
        # points after each midpoint displacement it is necessary to create a copy
        # of the state at the beginning of the iteration so we can iterate over
        # the original sequence.
        # Tuple type is used for security reasons since they are immutable in Python.
        points_tup = tuple(points)
        for i in range(len(points_tup)-1):
            # Calculate x and y midpoint coordinates:
            # [(x_i+x_(i+1))/2, (y_i+y_(i+1))/2]
            midpoint = list(map(lambda x: (points_tup[i][x]+points_tup[i+1][x])/2, [0, 1]))
            # Displace midpoint y-coordinate
            midpoint[1] += random.choice([-vertical_displacement, vertical_displacement])
            # Insert the displaced midpoint in the current list of points         
            bisect.insort(points, midpoint)
            # bisect allows to insert an element in a list so that its order
            # is preserved.
            # By default the maintained order is from smallest to biggest list first
            # element which is what we want.
        # Reduce displacement range
        vertical_displacement *= 2 ** (-roughness)
        # update number of iterations
        iteration += 1
    return points

class Game:
    class GameState:
        # Lander properties
        landerx = 0
        landery = 0
        landerdx = 0
        landerdy = 0
        landerddx = 0
        landerddy = 0
        landerRotation = 0
        landerLanded = False
        landerExploded = False

        # Lander implicit inputs (control inputs)
        landerThrust = 0             # saturated [0,1]
        landerTargetRotation = 0     # saturated [-90, 90]
        
        # Lander explicit inputs (player controlled inputs)
        rotateLanderLeft = False
        rotateLanderRight = False
        increaseLanderThrust = False

        # Metrics
        score = 0
        time = 0
        fuel = 0
    
    def __init__(self, width, height, ts, expectUserInputs=True):
        self.width = width
        self.height = height
        self.ts = ts # time step
        self.expectUserInputs = expectUserInputs
        
        # Game state list - stores a state for each time step
        gs = Game.GameState()
        gs.landerx = width - width//2
        gs.landery = height - height//6
        self.states = [gs]
        
        # Landscape, creates self.landscape and self.lines
        self.landscapeGenerator(width, height)
        
        # Determines if simulation is active or not
        self.alive = True

    def landscapeGenerator(self, width, height):
        #lines = []
        #line = Line(0, height//3, width, height//3, True)
        #lines.append(line)
        #return lines
        
        
        # Initial points
        points = [[0, height//3], [width, height//3]]

        # Create points for landscape with midpoint displacement
        points = midpoint_displacement(points[0], points[1], GameConstants.terrainR, height//3, 5)
        
        # Map points to the base of our window (up to 98%)
        #points = list(map(lambda p: [p[0], min(height*0.98, height-p[1])], points))
        points = list(map(lambda p: [p[0], max(height*0.02, p[1])], points))
        
        # Sort at least 3 landing spots (there may be more if there are points below 98% window)
        landingSpotsCount = 1
        landingSpots = random.sample(points[:-1], landingSpotsCount)
            
        # Create lines for our landscape
        self.landscape = []
        self.lines = []
        
        last_point = points[0]
        for point in points[1:]:
            landingSpot = False
            
            # If it's a designated landing spot make it flat
            if last_point in landingSpots:
                point[1] = last_point[1]

            # If it's flat then it's a landing spot
            if point[1] == last_point[1]:
                landingSpot = True

            line = Line(last_point[0], last_point[1], point[0], point[1], landingSpot)            
            self.landscape.append(line)
            
            line = Line(last_point[0], self.height - last_point[1], point[0], self.height - point[1], landingSpot)            
            self.lines.append(line)
            
            last_point = point
    
    def checkCollision(self, gs):
        w = GameConstants.landerWidth
        h = GameConstants.landerHeight
        
        # Lander corners (vertices)
        landerLeft = gs.landerx - w/2
        landerRight = gs.landerx + w/2
        landerBottom = gs.landery - h/2
        landerTop = gs.landery + h/2
        
        v0 = (landerRight, landerTop)
        v1 = (landerRight, landerBottom)
        v2 = (landerLeft, landerTop)
        v3 = (landerLeft, landerBottom)
        v = [v0, v1, v2, v3]
        
        # Check all lines in landscape
        for l in self.landscape:
            # If the left-most of right-most vertices are in this line's bounds
            if l.p0.x <= landerLeft <= l.p1.x or l.p0.x <= landerRight <= l.p1.x:
                m = (l.p1.y - l.p0.y)/(l.p1.x - l.p0.x)
                y = lambda x: m*(x - l.p0.x) + l.p0.y

                # consider only vertices in domain (x) of this line
                inDomain = list(map(lambda vi: l.p0.x <= vi[0] <= l.p1.x, v))
                vInDomain =  [vi for (vi, b) in zip(v, inDomain) if b]
                
                # check if any vertices are under (above in pixels) the line
                if any(map(lambda vi: vi[1] <= y(vi[0]), vInDomain)):
                    '''
                    print('BOOM')
                    print('p0 {} p1 {}'.format(l.p0.as_tuple(),l.p1.as_tuple()))
                    print('v0 {} v1 {} v2 {} v3 {}'.format(v0,v1,v2,v3))
                    print(list(map(lambda vi: vi[1] <= y(vi[0]), v)))
                    print(list(map(lambda vi: vi[0], v)))
                    print(list(map(lambda vi: vi[1], v)))
                    print(list(map(lambda vi: y(vi[0]), v)))
                    '''
                    if l.landingSpot and abs(gs.landerRotation) <= 15 and abs(gs.landerdy) <= 20:
                        gs.landerLanded = True
                    else:
                        gs.landerExploded = True
    
    # Implements a game tick
    # Each call simulates a world step
    def update(self):
        # If the game is done, do nothing
        #if self.landerLanded or self.landerExploded:
        if not self.alive:
            return

        # Get current game state
        gs = self.states[-1]
        
        # Update time tick
        gs.time += self.ts
        
        # Process user inputs - in the absence of user inputs, implicit inputs are expected (gs.landerThrust and gs.landerTargetRotation)
        if self.expectUserInputs:
            if gs.rotateLanderLeft:
                gs.landerTargetRotation += 0.5
            elif gs.rotateLanderRight:
                gs.landerTargetRotation -= 0.5

            if gs.increaseLanderThrust:
                gs.landerThrust += 0.1
            else:
                gs.landerThrust -= 0.1
            
        # Saturate rotation and thrust
        if gs.landerTargetRotation > 90:
                gs.landerTargetRotation = 90
        elif gs.landerTargetRotation < -90:
                gs.landerTargetRotation = -90

        if gs.landerThrust > 1:
            gs.landerThrust = 1 #0.7 + 3*random.random()/10# gives a nice turbulence
        elif gs.landerThrust < 0:
            gs.landerThrust = 0

        # TODO: it might be interesting to limit implicit inputs to a few digits
        #gs.landerTargetRotation = round(gs.landerTargetRotation, 2)
        #gs.landerThrust = round(gs.landerThrust, 2)

        # lowpass filter on rotation to simulate rotation dynamics
        rf = GameConstants.rotationFilter
        gs.landerRotation = gs.landerTargetRotation*(1 - rf) + gs.landerRotation*rf            
            
        # Fuel consumption is proportional to integral of thrust
        gs.fuel += gs.landerThrust
        
        ## First order integration (Newton method) - moves the lander
        # PS: *IMPORTANT* The order of x/y dx/dy ddx/ddy changes the final result!
        # We want dx/dt = F(t, ...). As such, each ODE is dependent only on PAST terms. Past terms is badly implemented for landerddx/y, should be fixed (TODO).
        # x/y
        gs.landerx += gs.landerdx*self.ts
        gs.landery += gs.landerdy*self.ts
        
        # dx/dy considering drag
        gs.landerdx += gs.landerddx*self.ts
        gs.landerdy += gs.landerddy*self.ts
                                                  
        # Update acceleration coefficients based on thrust
        gs.landerddx = -gs.landerThrust*GameConstants.thrustAcceleration*math.sin(math.radians(gs.landerRotation)) - GameConstants.drag*gs.landerdx
        gs.landerddy = gs.landerThrust*GameConstants.thrustAcceleration*math.cos(math.radians(gs.landerRotation)) - GameConstants.gravity - GameConstants.drag*gs.landerdy

        # Go around the moon when near an edge        
        gs.landerx %= self.width
                                                  
        # Check for collisions                                                  
        self.checkCollision(gs)
        
        # Signal if the game ended
        if gs.landerLanded or gs.landerExploded:
                self.alive = False
       

def landscapeDraw(screen, game):
    rects = []

    rects += [pygame.draw.lines(screen, GameConstants.ColorWhite, False, [p for l in game.lines for p in l.as_tuple()], 2)]

    for l in game.lines:
        if l.landingSpot:
            rects += [pygame.draw.line(screen, GameConstants.ColorGreen, l.p0.as_tuple(), l.p1.as_tuple(), 3)]
    
    
    return rects

def infoDraw(screen, font, game):
    rects = []
    
    aa = False
    fontSize = GameConstants.fontSize
    xInfo1, yInfo1 = 10, 15
    
    gs = game.states[-1]
    
    fontSurface = font.render("SCORE: {:04d}".format(gs.score), aa, GameConstants.ColorWhite)
    rects.append(screen.blit(fontSurface, [xInfo1, yInfo1]))
    
    fontSurface = font.render("TIME:  {:01d}:{:02d}".format(int(gs.time)//60, int(gs.time) % 60), aa, GameConstants.ColorWhite)
    rects.append(screen.blit(fontSurface, [xInfo1, yInfo1+fontSize]))
    
    fontSurface = font.render("FUEL:  {:04d}".format(int(gs.fuel)), aa, GameConstants.ColorWhite)
    rects.append(screen.blit(fontSurface, [xInfo1, yInfo1+2*fontSize]))
    
    
    fontSurface = font.render("ALTITUDE:          {:04d}".format(int(gs.landery)), aa, GameConstants.ColorWhite)
    fontRect = fontSurface.get_rect()
    fontRect.y = yInfo1
    fontRect.right = game.width - 10
    rects.append(screen.blit(fontSurface, fontRect))
    
    fontSurface = font.render("HORIZONTAL SPEED:  {:04d}".format(int(gs.landerdx)), aa, GameConstants.ColorWhite)
    fontRect = fontSurface.get_rect()
    fontRect.y = yInfo1 + fontSize
    fontRect.right = game.width - 10
    rects.append(screen.blit(fontSurface, fontRect))
    
    fontSurface = font.render("VERTICAL SPEED:    {:04d}".format(int(gs.landerdy)), aa, GameConstants.ColorWhite)
    fontRect = fontSurface.get_rect()
    fontRect.y = yInfo1 + 2*fontSize
    fontRect.right = game.width - 10
    rects.append(screen.blit(fontSurface, fontRect))
    
    fontSurface = font.render("ROTATION:          {:04d}".format(int(abs(gs.landerRotation))), aa, GameConstants.ColorWhite)
    fontRect = fontSurface.get_rect()
    fontRect.y = yInfo1 + 3*fontSize
    fontRect.right = game.width - 10
    rects.append(screen.blit(fontSurface, fontRect))
    
    return rects

def lunarLanderDraw(screen, game):
    rects = []
    
    w = GameConstants.landerWidth
    h = GameConstants.landerHeight
    gs = game.states[-1]
    
    surf = pygame.Surface((w+1, h+1))
    surf.set_colorkey((0,0,0)) # transparency
    
    #x = int(game.width - game.landerx)
    #y = int(game.height - game.landery)
    x = 0
    y = 0
    
    if gs.landerLanded:
        color = GameConstants.ColorGreen
    elif gs.landerExploded:
        color = GameConstants.ColorRed
    else:
        color = GameConstants.ColorWhite
    
    # Draw encompassing rectangle
    #pygame.draw.rect(surf, color, (x, y, w, h), 1)
    
    # Draw circle
    pygame.draw.circle(surf, color, (x+w//2, y+h//5), h//5, 1)
    
    # Draw small rectangle
    pygame.draw.rect(surf, color, (x+w//5, y+2*h//5, w-2*w//5, h//6), 1)
    
    # Draw legs
    pygame.draw.line(surf, color, (x+1.5*w//5, y+2*h//5+h//6), (x+0.5*w//5, y+h), 1)
    pygame.draw.line(surf, color, (x + w - 1.5*w//5, y+2*h//5+h//6), (x + w - 0.5*w//5, y+h), 1)
    
    # Draw feet
    pygame.draw.line(surf, color, (x, y+h), (x+1.0*w//5, y+h), 2)
    pygame.draw.line(surf, color, (x + w, y+h), (x + w - 1.0*w//5, y+h), 2)
    
    # Draw thrust
    tby = y+2*h//5+h//6 # thrust base y
    tm = y + h - tby    # thrust multiplier for height
    pygame.draw.line(surf, color, (x+3*w//5, tby), (x+w//2, tby+tm*gs.landerThrust), 1)
    pygame.draw.line(surf, color, (x + w - 3*w//5, tby), (x + w//2, tby+tm*gs.landerThrust), 1)
    
    # Apply rotation and center
    rotSurf = pygame.transform.rotate(surf, gs.landerRotation)
    r = pygame.Rect(rotSurf.get_rect())
    r.center = (gs.landerx, game.height - gs.landery)
    rects += [screen.blit(rotSurf, r)]
    
    return rects

def draw(screen, font, game):
    rects = []
            
    rects += [screen.fill(GameConstants.BackgroundColor)]
    rects += landscapeDraw(screen, game)
    rects += infoDraw(screen, font, game)
    rects += lunarLanderDraw(screen, game)
    
    return rects

def initialize():
    random.seed(GameConstants.randomSeed)
    pygame.init()
    game = Game(int(GameConstants.screenWidth), int(GameConstants.screenHeight), 1/GameConstants.FPS)
    font = pygame.font.SysFont('Courier', GameConstants.fontSize)
    fpsClock = pygame.time.Clock()

    # Create display surface
    screen = pygame.display.set_mode((game.width, game.height), pygame.DOUBLEBUF)
    screen.fill((0, 0, 0))
        
    return screen, font, game, fpsClock

def handleEvents(game):
    gs = game.states[-1]
    
    
    if not hasattr(game, 'neural_controller'):
        game.neural_controller = ImprovedNeuralController()
    
    left, thrust, right = game.neural_controller.get_controls(gs, game.landscape)
    
    gs.rotateLanderLeft = left
    gs.increaseLanderThrust = thrust
    gs.rotateLanderRight = right
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            sys.exit()
            
def avaliar_pouso(game):
    estado_final = game.states[-1]

    if estado_final.landerLanded:
        return 1000
    elif estado_final.landerExploded:
        vel = abs(estado_final.landerdx) + abs(estado_final.landerdy)
        return -500 - 10 * vel
    else:
        distancia = abs(estado_final.landery)
        vel = abs(estado_final.landerdx) + abs(estado_final.landerdy)
        return -distancia - vel
    
def evaluate_landing(game):
    final_state = game.states[-1]
    
    if final_state.landerLanded:
        # Recompensa por pouso bem-sucedido
        rotation_penalty = abs(final_state.landerRotation) * 2
        speed_penalty = (abs(final_state.landerdx) + abs(final_state.landerdy)) * 5
        return 1000 - rotation_penalty - speed_penalty
    elif final_state.landerExploded:
        # Penalidade por explosão baseada na velocidade
        return -1000 - (abs(final_state.landerdx) + abs(final_state.landerdy)) * 10
    else:
        # Recompensa intermediária baseada em:
        # - Proximidade ao solo
        # - Velocidade controlada
        # - Orientação adequada
        altitude_reward = (GameConstants.screenHeight - final_state.landery) * 0.1
        speed_penalty = (abs(final_state.landerdx) + abs(final_state.landerdy)) * 2
        rotation_penalty = abs(final_state.landerRotation) * 0.5
        return altitude_reward - speed_penalty - rotation_penalty
    
def genetic_algorithm_training(population_size=50, generations=20, mutation_rate=0.1):
    population = [ImprovedNeuralController() for _ in range(population_size)]
    
    for generation in range(generations):
        # Avaliar cada indivíduo
        scores = []
        for individual in population:
            score = 0
            for _ in range(3):  # Média de 3 tentativas
                result = simulate_game(individual)
                score += result
            scores.append(score / 3)
        
        # Selecionar os melhores
        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        best = ranked[0]
        print(f"Geração {generation}: Melhor score = {best[0]:.2f}")
        
        # Manter os melhores 25%
        top_quarter = [x[1] for x in ranked[:population_size//4]]
        
        # Reproduzir e mutar
        new_population = top_quarter.copy()
        while len(new_population) < population_size:
            parent = random.choice(top_quarter)
            child = copy.deepcopy(parent)
            
            # Aplicar mutação
            weights = networkSerialize(child.layers)
            for i in range(len(weights)):
                if random.random() < mutation_rate:
                    weights[i] += random.uniform(-0.5, 0.5)
            networkDeserialize(weights, child.layers)
            
            new_population.append(child)
        
        population = new_population
    
    return population[0]  # Retorna o melhor indivíduo
def train_controller():
    # Usar algoritmo genético para treinamento
    best_controller = genetic_algorithm_training()
    
    # Testar o melhor controlador
    print("\nTestando o melhor controlador:")
    simulate_game(best_controller, render=True)
    
    return best_controller


def simulate_game(controller, render=False, max_steps=1000):
    if render:
        # Inicializar pygame se estiver renderizando
        pygame.init()
        screen = pygame.display.set_mode((GameConstants.screenWidth, GameConstants.screenHeight))
        font = pygame.font.SysFont('Courier', GameConstants.fontSize)
        fpsClock = pygame.time.Clock()
    
    game = Game(GameConstants.screenWidth, GameConstants.screenHeight, 1/GameConstants.FPS, render)
    total_reward = 0
    
    for step in range(max_steps):
        if not game.alive:
            break

        gs = game.states[-1]
        left, thrust, right = controller.get_controls(gs, game.landscape)

        new_gs = copy.copy(gs)
        new_gs.rotateLanderLeft = left > 0.5
        new_gs.increaseLanderThrust = thrust > 0.5
        new_gs.rotateLanderRight = right > 0.5
        game.states.append(new_gs)

        game.update()
        
        # Cálculo da recompensa (mesmo código anterior)
        reward = 0
        if new_gs.landerdy < 0:
            reward += abs(new_gs.landerdy) * 0.1
        else:
            reward -= new_gs.landerdy * 0.5
        reward -= abs(new_gs.landerdx) * 0.3
        reward -= abs(new_gs.landerRotation) * 0.1
        reward += (500 - new_gs.fuel) * 0.01
        total_reward += reward

        if render:
            # Desenhar o jogo
            rects = []
            rects += [screen.fill(GameConstants.BackgroundColor)]
            rects += landscapeDraw(screen, game)
            rects += infoDraw(screen, font, game)
            rects += lunarLanderDraw(screen, game)
            pygame.display.update(rects)
            fpsClock.tick(GameConstants.FPS)

            # Processar eventos para não travar
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return total_reward

    # Recompensa/penalidade final (mesmo código anterior)
    final_state = game.states[-1]
    if final_state.landerLanded:
        rotation_penalty = abs(final_state.landerRotation) * 2
        speed_penalty = (abs(final_state.landerdx) + abs(final_state.landerdy)) * 5
        total_reward += 1000 - rotation_penalty - speed_penalty
    elif final_state.landerExploded:
        total_reward -= 1000 + (abs(final_state.landerdx) + abs(final_state.landerdy)) * 10

    if render:
        time.sleep(2)
        pygame.quit()
        
    return total_reward


def saveGame(game):
    with open("moonlander-{}.pickle".format(time.strftime("%Y%m%d-%H%M%S")), "wb") as f:
        pickle.dump([GameConstants, game], f)

        
def loadGame(fileName):
    with open(fileName, "rb") as f:
        GameConstants, game = pickle.load(f)
        
    return game
    
            
def mainGamePlayer():
    try:
        # Initialize pygame and etc.
        screen, font, game, fpsClock = initialize()
              
        # Main game loop
        while game.alive:
            # Copy current game state and add the new state for modifications
            gs = copy.copy(game.states[-1])
            game.states += [gs]

            # Handle events
            handleEvents(game)
                    
            # Update world
            game.update()
            
            # Draw this world frame
            rects = draw(screen, font, game)     
            pygame.display.update(rects)
            
            # Delay for required FPS
            fpsClock.tick(GameConstants.FPS)

        # save this playthrough    
        saveGame(game)
    except SystemExit:
        pass
    except Exception as e:
        #print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
        #raise Exception from e
    finally:
        # close up shop
        pygame.quit() 
    
def mainGameAutonomous(thrust, rotation):
    try:
        # Verify if both thrust and rotation are the same size
        if(len(rotation) != len(thrust)):
            raise Exception('Thrust and rotation vectors must be the same size')
        
        # Initialize pygame and etc.
        screen, font, game, fpsClock = initialize()

        # Disable user inputs and supply implicit inputs (thrust and target rotation)
        game.expectUserInputs = False
              
        # Main game loop
        for t, r in zip(thrust, rotation):
            # Copy current game state and add the new state for modifications
            gs = copy.copy(game.states[-1])
            game.states += [gs]

            # Handle events from data
            game.states[-1].landerThrust = t
            game.states[-1].landerTargetRotation = r
                    
            # Update world
            game.update() 

            # Draw this world frame
            rects = draw(screen, font, game)     
            pygame.display.update(rects)
            
            # Delay for required FPS
            fpsClock.tick(GameConstants.FPS)           

    except SystemExit:
        pass
    except:
        #print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
    finally:
        # close up shop
        pygame.quit() 

def mainGameAutonomousUserInputs(gamestates):
    try:
        # Initialize pygame and etc.
        screen, font, game, fpsClock = initialize()
              
        # Main game loop
        for state in gamestates:
            # Copy current game state and add the new state for modifications
            gs = copy.copy(game.states[-1])
            game.states += [gs]

            # Handle events from data
            game.states[-1].rotateLanderLeft = state.rotateLanderLeft
            game.states[-1].rotateLanderRight = state.rotateLanderRight
            game.states[-1].increaseLanderThrust = state.increaseLanderThrust
                    
            # Update world
            game.update() 

            # Draw this world frame
            rects = draw(screen, font, game)     
            pygame.display.update(rects)
            
            # Delay for required FPS
            fpsClock.tick(GameConstants.FPS)           
              
    except SystemExit:
        pass
    except:
        #print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
    finally:
        # close up shop
        pygame.quit() 


if __name__ == "__main__":
    # Treinar ou carregar um controlador treinado
    try:
        trained_controller = train_controller()
    except KeyboardInterrupt:
        print("Treinamento interrompido pelo usuário")
        sys.exit()
    
    # Configuração do pygame para visualização
    screen, font, game, fpsClock = initialize()
    
    # Loop de demonstração
    while True:
        gs = copy.copy(game.states[-1])
        game.states += [gs]
        
        left, thrust, right = trained_controller.get_controls(gs, game.landscape)
        gs.rotateLanderLeft = left > 0.5
        gs.increaseLanderThrust = thrust > 0.5
        gs.rotateLanderRight = right > 0.5
        
        game.update()
        rects = draw(screen, font, game)
        pygame.display.update(rects)
        fpsClock.tick(GameConstants.FPS)
        
        if not game.alive:
            time.sleep(2)
            screen, font, game, fpsClock = initialize()

    # Load a save game
    #game = loadGame('moonlander-20180516-135318.pickle')
    
    # game.states[1:] is needed to skip initial state that already exists in Game
    # assuming initial conditions are the same in data and in simulation

    '''
    # Implicit inputs
    thrusts = [gs.landerThrust for gs in game.states[1:]]
    rotations = [gs.landerTargetRotation for gs in game.states[1:]]
    mainGameAutonomous(thrusts, rotations)
    
    # Explicit inputs from user
    mainGameAutonomousUserInputs(game.states[1:])
    '''
	
    
    
    
    
    
    
    