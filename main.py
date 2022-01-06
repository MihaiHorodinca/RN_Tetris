import pygame
import random
import copy
import numpy as np
from qlearn_agent import Agent

S = [['.00',
      '00.'],
     ['0.',
      '00',
      '.0']]

Z = [['00.',
      '.00'],
     ['.0',
      '00',
      '0.']]

I = [['0',
      '0',
      '0',
      '0', ],
     ['0000']]

O = [['00',
      '00']]

J = [['0..',
      '000'],
     ['00',
      '0.',
      '0.'],
     ['000',
      '..0'],
     ['.0',
      '.0',
      '00']]

L = [['..0',
      '000'],
     ['0.',
      '0.',
      '00'],
     ['000',
      '0..'],
     ['00',
      '.0',
      '.0']]

T = [['.0.',
      '000'],
     ['0.',
      '00',
      '0.'],
     ['000',
      '.0.'],
     ['.0',
      '00',
      '.0']]

C = [['0.0',
      '000'],
     ['00',
      '0.',
      '00'],
     ['000',
      '0.0'],
     ['00',
      '.0',
      '00']]

shapes = [S, Z, I, O, J, L, T, C]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0),
                (255, 165, 0), (0, 0, 255), (128, 0, 128), (255, 0, 255)]

clear_scores = [40, 100, 300, 1200]


class Piece(object):
    def __init__(self, row, column, shape):
        self.x = row
        self.y = column
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0  # number from 0-3


class Game(object):
    def __init__(self):
        self.s_width = 800
        self.s_height = 700
        self.play_width = 300  # meaning 300 // 10 = 30 width per block
        self.play_height = 600  # meaning 600 // 20 = 20 height per block
        self.block_size = 30
        self.top_left_y = (self.s_width - self.play_width) // 2
        self.top_left_x = self.s_height - self.play_height
        self.surface = pygame.display.set_mode((self.s_width, self.s_height))

        self.locked_positions = {}
        self.grid = self.create_grid(self.locked_positions)
        self.current_piece = self.get_shape()
        self.next_piece = self.get_shape()

    def reset(self):
        self.locked_positions = {}
        self.grid = self.create_grid(self.locked_positions)
        self.current_piece = self.get_shape()
        self.next_piece = self.get_shape()

    def create_grid(self, locked_positions={}):
        grid = [[(0, 0, 0) for x in range(10)] for x in range(20)]

        for i, row in enumerate(grid):
            for j, block in enumerate(row):
                if (i, j) in locked_positions:
                    c = locked_positions[(i, j)]
                    grid[i][j] = c
        return grid

    def get_shape(self):
        return Piece(0, self.play_width / self.block_size / 2 - 1, random.choice(shapes))

    def draw_grid(self, surface, grid):
        for i, row in enumerate(grid):
            for j, block in enumerate(row):
                pygame.draw.rect(surface, block,
                                 (self.top_left_y + j * self.block_size, self.top_left_x + i * self.block_size,
                                  self.block_size, self.block_size), 0)

    def draw_window(self, surface, grid):
        surface.fill((0, 0, 0))

        self.draw_grid(surface, grid)

        current_piece_pos = self.piece_to_positions(self.current_piece)
        for pos in current_piece_pos:
            i, j = pos
            pygame.draw.rect(surface, self.current_piece.color,
                             (self.top_left_y + j * self.block_size, self.top_left_x + i * self.block_size,
                              self.block_size, self.block_size), 0)

        pygame.draw.rect(surface, (128, 128, 128),
                         (self.top_left_y, self.top_left_x, self.play_width, self.play_height), 5)

        pygame.display.update()

    def piece_to_positions(self, piece):
        piece_positions = []

        for i, row in enumerate(piece.shape[piece.rotation]):
            for j, char in enumerate(row):
                if char == '0':
                    piece_positions.append((int(piece.x + i), int(piece.y + j)))

        return piece_positions

    def valid_space(self, piece, grid):
        accepted_positions = [[(i, j) for j in range(int(self.play_width / self.block_size)) if grid[i][j] == (0, 0, 0)]
                              for i in range(int(self.play_height / self.block_size))]
        accepted_positions = [pos for sub in accepted_positions for pos in sub]

        piece_positions = self.piece_to_positions(piece)

        for pos in piece_positions:
            if pos not in accepted_positions:
                return False

        return True

    def clear_rows(self, locked_positions):
        grid = self.create_grid(locked_positions)
        n_rows_cleared = 0

        for x, row in enumerate(grid):
            must_clear = True

            for block in row:
                if block == (0, 0, 0):
                    must_clear = False
                    break

            if must_clear:
                n_rows_cleared += 1
                row_to_move = x

                while row_to_move > 0:
                    grid[row_to_move] = grid[row_to_move - 1]
                    row_to_move -= 1

        locked_positions = {}

        for x, row in enumerate(grid):
            for y, block in enumerate(row):
                if grid[x][y] != (0, 0, 0):
                    locked_positions[(x, y)] = grid[x][y]

        return locked_positions, n_rows_cleared

    def compute_score(self, action):
        score = 0
        heights = [0 for _ in range(10)]
        holes = 0
        bumpiness = 0
        mock_piece = copy.deepcopy(self.current_piece)
        mock_locked_positions = copy.deepcopy(self.locked_positions)
        mock_grid = copy.deepcopy(self.create_grid(mock_locked_positions))

        if action == 2:
            lines_droped = 0
            while self.valid_space(mock_piece, mock_grid):
                mock_piece.x += 1
                lines_droped += 1
            mock_piece.x -= 1
            lines_droped -= 1

            score = lines_droped * 2
            current_piece_positions = self.piece_to_positions(mock_piece)

            for pos in current_piece_positions:
                mock_locked_positions[(pos[0], pos[1])] = mock_piece.color
            mock_grid = self.create_grid(mock_locked_positions)

        for x, row in enumerate(mock_grid):
            for y, block in enumerate(row):
                if block != (0, 0, 0) and heights[y] < 20 - x:
                    heights[y] = 20 - x
                if block == (0, 0, 0) and heights[y] > 20 - x:
                    holes += 1

        for i in range(9):
            bumpiness += abs(heights[i] - heights[i + 1])

        score -= sum(heights) * 3
        score -= holes * 2
        score -= bumpiness * 1

        if action == 0 or action == 1:
            return score - 5
        if action == 3:
            return score - 10
        if action == 2:
            return score

    def compile_state(self):
        state = np.zeros(21)

        for x, row in enumerate(self.grid):
            for y, block in enumerate(row):
                if block != (0, 0, 0) and state[y] < 20 - x:
                    state[y] = 20 - x

        state[10] = self.current_piece.x
        state[11] = self.current_piece.y

        state[12 + shapes.index(self.current_piece.shape)] = 1

        state[20] = self.current_piece.rotation

        return state

    def step(self, action):
        change_piece = False
        score = self.compute_score(action)
        done = False
        self.grid = self.create_grid(self.locked_positions)

        if action == 0:  # move left
            self.current_piece.y -= 1
            if not self.valid_space(self.current_piece, self.grid):
                self.current_piece.y += 1
                score -= 200
        if action == 1:  # move right
            self.current_piece.y += 1
            if not self.valid_space(self.current_piece, self.grid):
                self.current_piece.y -= 1
                score -= 200
        if action == 2:  # hard drop
            while self.valid_space(self.current_piece, self.grid):
                self.current_piece.x += 1
            self.current_piece.x -= 1
            change_piece = True
        if action == 3:  # rotate
            self.current_piece.rotation = (self.current_piece.rotation + 1) % len(self.current_piece.shape)
            if not self.valid_space(self.current_piece, self.grid):
                self.current_piece.rotation = (self.current_piece.rotation - 1) % len(self.current_piece.shape)

        self.current_piece.x += 1
        if not self.valid_space(self.current_piece, self.grid):
            self.current_piece.x -= 1
            change_piece = True

        if change_piece:
            current_piece_positions = self.piece_to_positions(self.current_piece)

            for pos in current_piece_positions:
                self.locked_positions[(pos[0], pos[1])] = self.current_piece.color
            self.current_piece = self.next_piece
            self.next_piece = self.get_shape()
            self.locked_positions, no_cleared = self.clear_rows(self.locked_positions)
            self.grid = self.create_grid(self.locked_positions)
            score += clear_scores[no_cleared]

            if not self.valid_space(self.current_piece, self.grid):
                done = True

        state = self.compile_state()
        if done:
            score -= 10000

        self.draw_window(self.surface, self.grid)

        return state, score, done


if __name__ == '__main__':

    game = Game()
    no_games = 50
    agent = Agent(alpha=0.0005, gamma=0.99, eps=1., eps_dot=0.996, eps_min=0.01,
                  no_actions=4, batch_size=16, in_d=21)

    scores = []

    for i in range(no_games):
        done = False
        score = 0
        game.reset()
        state = game.compile_state()

        while not done:
            action = agent.choose_action(state)
            state_, reward, done = game.step(action)
            score += reward
            agent.save_all(state, action, reward, state_, done)
            state = state_

            agent.learn()

        scores.append(score)
        avg_score = np.mean(scores[max(0, i - 10):])
        print("Game : ", i)
        print("Score : ", score)
        print("Average : ", avg_score)
