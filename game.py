import random
from enum import Enum
from collections import namedtuple

import numpy as np

WHITE = (245, 247, 250)
RED = (255, 91, 105)
GREEN = (52, 211, 153)
GREEN_DARK = (16, 185, 129)
BLACK = (11, 18, 32)
GRID = (29, 41, 59)

BLOCK_SIZE = 20
SPEED = 40


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")


class SnakeGameAI:
    """Headless Snake environment. Rendering is returned as an RGB array for Streamlit."""

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y),
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            point = Point(x, y)
            if point not in self.snake:
                self.food = point
                return

    def play_step(self, action):
        self.frame_iteration += 1
        self._move(action)
        self.snake.insert(0, self.head)

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            return -10, True, self.score

        reward = 0
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        return reward, False, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
            return True
        return pt in self.snake[1:]

    def _move(self, action):
        # Action encoding: [straight, right turn, left turn]
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = clockwise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_direction = clockwise[(idx + 1) % 4]
        else:
            new_direction = clockwise[(idx - 1) % 4]

        self.direction = new_direction
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def render(self):
        """Return a uint8 RGB image without opening a desktop window."""
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (self.w, self.h), BLACK)
        draw = ImageDraw.Draw(image)

        for x in range(0, self.w, BLOCK_SIZE):
            draw.line((x, 0, x, self.h), fill=GRID, width=1)
        for y in range(0, self.h, BLOCK_SIZE):
            draw.line((0, y, self.w, y), fill=GRID, width=1)

        fx, fy = self.food
        draw.rounded_rectangle(
            (fx + 2, fy + 2, fx + BLOCK_SIZE - 2, fy + BLOCK_SIZE - 2),
            radius=5,
            fill=RED,
        )

        for index, point in reversed(list(enumerate(self.snake))):
            x, y = point
            color = GREEN if index == 0 else GREEN_DARK
            draw.rounded_rectangle(
                (x + 1, y + 1, x + BLOCK_SIZE - 1, y + BLOCK_SIZE - 1),
                radius=5,
                fill=color,
            )

        draw.text((12, 10), f"Score: {self.score}", fill=WHITE)
        return np.asarray(image, dtype=np.uint8)
