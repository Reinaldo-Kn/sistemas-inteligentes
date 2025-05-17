# -*- coding: utf-8 -*-
"""
Recriação do Jogo da Velha com Jogador Automático

@author: Prof. Daniel Cavalcanti Jeronymo
"""

import pygame
import sys
import os
import traceback
import random
import numpy as np
import copy
import time

class GameConstants:
    #                  R    G    B
    ColorWhite     = (255, 255, 255)
    ColorBlack     = (  0,   0,   0)
    ColorRed       = (255,   0,   0)
    ColorGreen     = (  0, 255,   0)
    ColorBlue     = (  0, 0,   255)
    ColorDarkGreen = (  0, 155,   0)
    ColorDarkGray  = ( 40,  40,  40)
    BackgroundColor = ColorBlack
    
    screenScale = 1
    screenWidth = screenScale*600
    screenHeight = screenScale*600
    
    # grid size in units
    gridWidth = 3
    gridHeight = 3
    
    # grid size in pixels
    gridMarginSize = 5
    gridCellWidth = screenWidth//gridWidth - 2*gridMarginSize
    gridCellHeight = screenHeight//gridHeight - 2*gridMarginSize
    
    randomSeed = 0
    
    FPS = 30
    
    fontSize = 20

class Game:
    class GameState:
        
        grid = np.zeros((GameConstants.gridHeight, GameConstants.gridWidth))
        currentPlayer = 0
    
    def __init__(self, expectUserInputs=True):
        self.expectUserInputs = expectUserInputs
        
        # Game state list - stores a state for each time step (initial state)
        gs = Game.GameState()
        self.states = [gs]
        
        # Determines if simulation is active or not
        self.alive = True
        
        self.currentPlayer = 2
        
        # Journal of inputs by users (stack)
        self.eventJournal = []
        
        # Timer for computer moves
        self.last_computer_move_time = 0
        self.computer_move_delay = 500  # milliseconds
        
    def checkObjectiveState(self, gs):
        # Complete line?
        for i in range(3):
            s = set(gs.grid[i, :])
            if len(s) == 1 and min(s) != 0:
                return s.pop()
            
        # Complete column?
        for i in range(3):
            s = set(gs.grid[:, i])
            if len(s) == 1 and min(s) != 0:
                return s.pop()
            
        # Complete diagonal (main)?
        s = set([gs.grid[i, i] for i in range(3)])
        if len(s) == 1 and min(s) != 0:
            return s.pop()
        
        # Complete diagonal (opposite)?
        s = set([gs.grid[-i-1, i] for i in range(3)])
        if len(s) == 1 and min(s) != 0:
            return s.pop()
            
        # nope, not an objective state
        return 0
    
    def get_available_moves(self, grid):
        return [(i, j) for i in range(3) for j in range(3) if grid[i, j] == 0]

    def minimax(self, grid, depth, is_maximizing):
        winner = self.checkWinner(grid)
        if winner == 2:  # Jogador 2 (IA) venceu
            return 10 - depth
        elif winner == 1:  # Jogador 1 (humano) venceu
            return depth - 10
        elif len(self.get_available_moves(grid)) == 0:  # Empate
            return 0

        if is_maximizing:  # Vez da IA (jogador 2)
            best_score = -float('inf')
            for move in self.get_available_moves(grid):
                new_grid = grid.copy()
                new_grid[move] = 2
                score = self.minimax(new_grid, depth + 1, False)
                best_score = max(score, best_score)
            return best_score
        else:  # Vez do jogador humano (jogador 1)
            best_score = float('inf')
            for move in self.get_available_moves(grid):
                new_grid = grid.copy()
                new_grid[move] = 1
                score = self.minimax(new_grid, depth + 1, True)
                best_score = min(score, best_score)
            return best_score

    def checkWinner(self, grid):
        # Verifica linhas
        for i in range(3):
            if grid[i, 0] == grid[i, 1] == grid[i, 2] != 0:
                return grid[i, 0]
        
        # Verifica colunas
        for j in range(3):
            if grid[0, j] == grid[1, j] == grid[2, j] != 0:
                return grid[0, j]
        
        # Verifica diagonais
        if grid[0, 0] == grid[1, 1] == grid[2, 2] != 0:
            return grid[0, 0]
        if grid[0, 2] == grid[1, 1] == grid[2, 0] != 0:
            return grid[0, 2]
        
        return 0  # Nenhum vencedor

    def findBestMove(self, player):
        grid = self.states[-1].grid
        
        # Primeiro verifica se pode vencer na próxima jogada
        for move in self.get_available_moves(grid):
            new_grid = grid.copy()
            new_grid[move] = player
            if self.checkWinner(new_grid) == player:
                return move
        
        # Se não houver vitória imediata, verifica se precisa bloquear o oponente
        opponent = 1 if player == 2 else 2
        for move in self.get_available_moves(grid):
            new_grid = grid.copy()
            new_grid[move] = opponent
            if self.checkWinner(new_grid) == opponent:
                return move
        
        # Se não houver jogadas críticas, usa o minimax
        best_score = -float('inf') if player == 2 else float('inf')
        best_move = None
        
        for move in self.get_available_moves(grid):
            new_grid = grid.copy()
            new_grid[move] = player
            score = self.minimax(new_grid, 0, player != 2)
            
            if player == 2:  # Maximizando para a IA
                if score > best_score or best_move is None:
                    best_score = score
                    best_move = move
            else:  # Minimizando para o jogador humano
                if score < best_score or best_move is None:
                    best_score = score
                    best_move = move
        
        return best_move

    # Implements a game tick
    # Each call simulates a world step
    def update(self):  
        # If the game is done, do nothing
        if not self.alive:
            return
            
        # Get the current (last) game state
        gs = copy.copy(self.states[-1])
        
        
        if gs.currentPlayer == 2:
            current_time = pygame.time.get_ticks()
            if current_time - self.last_computer_move_time > self.computer_move_delay:
                best_move = self.findBestMove(2)
                if best_move is not None:
                    self.eventJournal.append(best_move)
                    self.last_computer_move_time = current_time

        
        # If there is no event, do nothing
        if not self.eventJournal:
            return
            
        # Switch player turn
        if gs.currentPlayer == 0:
            gs.currentPlayer = 1
        elif gs.currentPlayer == 1:
            gs.currentPlayer = 2
        elif gs.currentPlayer == 2:
            gs.currentPlayer = 1
            
        # Mark the cell clicked by this player if it's an empty cell
        x,y = self.eventJournal.pop()

        # Check if in bounds
        if x < 0 or y < 0 or x >= GameConstants.gridHeight or y >= GameConstants.gridWidth:
            return

        # Check if cell is empty
        if gs.grid[x][y] == 0:
            gs.grid[x][y] = gs.currentPlayer
        else: # invalid move
            return
        
        # Check if end of game
        if self.checkObjectiveState(gs):
            self.alive = False
                
        # Add the new modified state
        self.states += [gs]

def drawGrid(screen, game):
    screen.fill(GameConstants.BackgroundColor)
    rects = []
    
    gs = game.states[-1]
    grid = gs.grid
    current_player = gs.currentPlayer if gs.currentPlayer != 0 else 1

    for row in range(GameConstants.gridHeight):
        for col in range(GameConstants.gridWidth):
            m = GameConstants.gridMarginSize
            w = GameConstants.gridCellWidth
            h = GameConstants.gridCellHeight

            color = GameConstants.ColorWhite
            if grid[row][col] == 1:
                color = GameConstants.ColorRed
            elif grid[row][col] == 2:
                color = GameConstants.ColorBlue
            else:
                color = GameConstants.ColorWhite

            rect = pygame.Rect((2*m+w)*col + m, (2*m+h)*row + m, w, h)
            rects.append(pygame.draw.rect(screen, color, rect))

    return rects

def draw(screen, font, game):
    rects = []
            
    rects += drawGrid(screen, game)

    return rects

def initialize():
    random.seed(GameConstants.randomSeed)
    pygame.init()
    game = Game()
    font = pygame.font.SysFont('Courier', GameConstants.fontSize)
    fpsClock = pygame.time.Clock()

    # Create display surface
    screen = pygame.display.set_mode((GameConstants.screenWidth, GameConstants.screenHeight), pygame.DOUBLEBUF)
    screen.fill(GameConstants.BackgroundColor)
        
    return screen, font, game, fpsClock

def handleEvents(game):
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONUP:
            pos = pygame.mouse.get_pos()
            
            col = pos[0] // (GameConstants.screenWidth // GameConstants.gridWidth)
            row = pos[1] // (GameConstants.screenHeight // GameConstants.gridHeight)
            
            if game.states[-1].currentPlayer == 1 or game.states[-1].currentPlayer == 0:
                game.eventJournal.append((row, col))
            
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            sys.exit()

def mainGamePlayer():
    try:
        # Initialize pygame and etc.
        screen, font, game, fpsClock = initialize()
              
        # Main game loop
        while game.alive:
            # Handle events
            handleEvents(game)
                    
            # Update world
            game.update()
            
            # Draw this world frame
            rects = draw(screen, font, game)     
            pygame.display.update(rects)
            
            # Delay for required FPS
            fpsClock.tick(GameConstants.FPS)
            
        # debug prints
        winner = game.checkObjectiveState(game.states[-1])
        if winner == 1:
            print("Player 1 (Red) wins!")
        elif winner == 2:
            print("Player 2 (Blue) wins!")
        else:
            print("It's a tie!")
            
        pygame.time.wait(3000)
        
        pygame.quit()
    except SystemExit:
        pass
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        pygame.quit()
    
if __name__ == "__main__":
    # Set the working directory (where we expect to find files) to the same
    # directory this .py file is in. You can leave this out of your own
    # code, but it is needed to easily run the examples using "python -m"
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_path)

    mainGamePlayer()