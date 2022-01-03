import pygame
import random

s_width = 800
s_height = 700
play_width = 300  # meaning 300 // 10 = 30 width per block
play_height = 600  # meaning 600 // 20 = 20 height per block
block_size = 30

top_left_y = (s_width - play_width) // 2
top_left_x = s_height - play_height

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


class Piece(object):
    def __init__(self, row, column, shape):
        self.x = row
        self.y = column
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0  # number from 0-3


def create_grid(locked_positions={}):
    grid = [[(0, 0, 0) for x in range(10)] for x in range(20)]

    for i, row in enumerate(grid):
        for j, block in enumerate(row):
            if (i, j) in locked_positions:
                c = locked_positions[(i, j)]
                grid[i][j] = c
    return grid


def get_shape():
    return Piece(0, play_width / block_size / 2 - 1, random.choice(shapes))


def draw_grid(surface, grid):
    for i, row in enumerate(grid):
        for j, block in enumerate(row):
            pygame.draw.rect(surface, block, (top_left_y + j * block_size, top_left_x + i * block_size,
                                              block_size, block_size), 0)


def draw_window(surface, grid):
    surface.fill((0, 0, 0))

    draw_grid(surface, grid)

    pygame.draw.rect(surface, (128, 128, 128), (top_left_y, top_left_x, play_width, play_height), 5)

    pygame.display.update()


def piece_to_positions(piece):
    piece_positions = []

    for i, row in enumerate(piece.shape[piece.rotation]):
        for j, char in enumerate(row):
            if char == '0':
                piece_positions.append((int(piece.x + i), int(piece.y + j)))

    return piece_positions


def valid_space(piece, grid):
    accepted_positions = [[(i, j) for j in range(int(play_width / block_size)) if grid[i][j] == (0, 0, 0)]
                          for i in range(int(play_height / block_size))]
    accepted_positions = [pos for sub in accepted_positions for pos in sub]

    piece_positions = piece_to_positions(piece)

    for pos in piece_positions:
        if pos not in accepted_positions:
            return False

    return True


def clear_rows(locked_positions):
    grid = create_grid(locked_positions)
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

    return locked_positions


def draw_next_piece(surface, piece):
    pygame.draw.rect(surface, (200, 200, 200), (0, 0, 200, 200), 0)


if __name__ == '__main__':
    locked_positions = {}  # (x,y):(255,0,0)
    grid = create_grid(locked_positions)

    surface = pygame.display.set_mode((s_width, s_height))

    change_piece = False
    run = True
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 500

    while run:
        grid = create_grid(locked_positions)
        clock.tick()
        fall_time += clock.get_rawtime()

        if fall_time > fall_speed:
            fall_time = 0
            current_piece.x += 1
            if not valid_space(current_piece, grid):
                current_piece.x -= 1
                change_piece = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.display.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece.y -= 1
                    if not valid_space(current_piece, grid):
                        current_piece.y += 1

                elif event.key == pygame.K_RIGHT:
                    current_piece.y += 1
                    if not valid_space(current_piece, grid):
                        current_piece.y -= 1

                elif event.key == pygame.K_UP:
                    # rotate shape
                    current_piece.rotation = (current_piece.rotation + 1) % len(current_piece.shape)
                    if not valid_space(current_piece, grid):
                        current_piece.rotation = (current_piece.rotation - 1) % len(current_piece.shape)

                if event.key == pygame.K_DOWN:
                    # move shape down
                    current_piece.x += 1
                    if not valid_space(current_piece, grid):
                        current_piece.x -= 1

        current_piece_positions = piece_to_positions(current_piece)

        for pos in current_piece_positions:
            x, y = pos
            grid[x][y] = current_piece.color

        if change_piece:
            for pos in current_piece_positions:
                locked_positions[(pos[0], pos[1])] = current_piece.color
            current_piece = next_piece
            next_piece = get_shape()
            change_piece = False
            locked_positions = clear_rows(locked_positions)

            if not valid_space(current_piece, grid):
                quit()

        draw_window(surface, grid)
