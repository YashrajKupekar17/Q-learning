from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np

from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled

# Custom map for the extended Taxi environment (10x10 grid)
MAP = [
    "+--------------------+",
    "|R: | : :G| : | : : |",
    "| : | : : | : | : : |",
    "| : : : : : : | : : |",
    "| | : | : | | | | : |",
    "|Y| : |B: | | : | : |",
    "| : : : : : : | : : |",
    "| : | : : : : | : : |",
    "| : | : : |G| : |R: |",
    "| : : : : | | | | : |",
    "|B| : |Y: : : | | : |",
    "+--------------------+",
]

WINDOW_SIZE = (550, 350)


class TaxiEnv(Env):
    """
    Custom Taxi Environment for a 10x10 grid with extended pickup/dropoff locations.
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        global MAP
        # Pad each row to the maximum length for consistency
        max_length = max(len(row) for row in MAP)
        MAP = [row.ljust(max_length) for row in MAP]
        self.desc = np.array([list(row) for row in MAP], dtype="c")

        # Define extended locations (8 in total)
        self.locs = [(0, 0), (0, 4), (4, 0), (4, 3), (7, 5), (7, 8), (9, 0), (9, 3)]
        self.locs_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (255, 255, 0),
            (0, 0, 255),
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 255),
            (0, 0, 0),
        ]

        # Define grid dimensions (10x10)
        self.num_rows = 10
        self.num_columns = 10
        max_row = self.num_rows - 1
        max_col = self.num_columns - 1

        # Determine total number of states
        # Calculation: taxi_row (10) * taxi_col (10) * (number of passenger locations = len(locs)+1) * (number of destinations = len(locs))
        num_states = self.num_rows * self.num_columns * (len(self.locs) + 1) * len(self.locs)
        num_actions = 6

        self.initial_state_distrib = np.zeros(num_states)
        self.P = {state: {action: [] for action in range(num_actions)} for state in range(num_states)}

        # Build transition dynamics for each state and action.
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                for pass_idx in range(len(self.locs) + 1):
                    for dest_idx in range(len(self.locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        # Only count initial states where passenger is at one of the first 4 locations and not already at the destination.
                        if pass_idx < 4 and pass_idx != dest_idx:
                            self.initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = -1  # default reward
                            terminated = False
                            taxi_loc = (row, col)

                            if action == 0:  # south
                                new_row = min(row + 1, max_row)
                            elif action == 1:  # north
                                new_row = max(row - 1, 0)
                            elif action == 2:  # east
                                # Check if passage exists (using the map description)
                                if self.desc[1 + row, 2 * col + 2] == b":":
                                    new_col = min(col + 1, max_col)
                            elif action == 3:  # west
                                if self.desc[1 + row, 2 * col] == b":":
                                    new_col = max(col - 1, 0)
                            elif action == 4:  # pickup
                                if pass_idx < 4 and taxi_loc == self.locs[pass_idx]:
                                    new_pass_idx = 4  # passenger is now in taxi
                                else:
                                    reward = -10
                            elif action == 5:  # dropoff
                                if (taxi_loc == self.locs[dest_idx]) and pass_idx == 4:
                                    new_pass_idx = dest_idx
                                    terminated = True
                                    reward = 40
                                elif (taxi_loc in self.locs) and pass_idx == 4:
                                    new_pass_idx = self.locs.index(taxi_loc)
                                else:
                                    reward = -10

                            new_state = self.encode(new_row, new_col, new_pass_idx, dest_idx)
                            self.P[state][action].append((1.0, new_state, reward, terminated))

        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)

        self.render_mode = render_mode
        # Initialize random number generator; allow seeding later
        self.np_random = np.random.RandomState()
        self.s = None
        self.lastaction = None

        # Pygame-related variables for rendering
        self.window = None
        self.clock = None
        self.cell_size = (
            WINDOW_SIZE[0] / self.desc.shape[1],
            WINDOW_SIZE[1] / self.desc.shape[0],
        )
        self.taxi_imgs = None
        self.taxi_orientation = 0
        self.passenger_img = None
        self.destination_img = None
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        # Encoding order: taxi_row, taxi_col, passenger_location, destination
        i = taxi_row
        i *= self.num_columns
        i += taxi_col
        i *= (len(self.locs) + 1)
        i += pass_loc
        i *= len(self.locs)
        i += dest_idx
        return i

    def decode(self, i):
        # Reverse the encoding process.
        dest_idx = i % len(self.locs)
        i //= len(self.locs)
        pass_loc = i % (len(self.locs) + 1)
        i //= (len(self.locs) + 1)
        taxi_col = i % self.num_columns
        taxi_row = i // self.num_columns
        return taxi_row, taxi_col, pass_loc, dest_idx

    def action_mask(self, state: int):
        """
        Compute an action mask for the state.
        For a 10x10 grid, allow movement if within grid bounds.
        """
        mask = np.zeros(6, dtype=np.int8)
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)
        if taxi_row < self.num_rows - 1:
            mask[0] = 1  # can move south
        if taxi_row > 0:
            mask[1] = 1  # can move north
        if taxi_col < self.num_columns - 1 and self.desc[taxi_row + 1, 2 * taxi_col + 2] == b":":
            mask[2] = 1  # can move east if passage exists
        if taxi_col > 0 and self.desc[taxi_row + 1, 2 * taxi_col] == b":":
            mask[3] = 1  # can move west if passage exists
        if pass_loc < 4 and (taxi_row, taxi_col) == self.locs[pass_loc]:
            mask[4] = 1  # pickup allowed
        if pass_loc == 4 and ((taxi_row, taxi_col) == self.locs[dest_idx] or (taxi_row, taxi_col) in self.locs):
            mask[5] = 1  # dropoff allowed
        return mask

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p, "action_mask": self.action_mask(s)})

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        self.taxi_orientation = 0
        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s)}

    def render(self):
        if self.render_mode is None:
            logger.warn("Render mode not specified. Use render_mode='human' or 'ansi'.")
        if self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled("pygame is not installed, run `pip install gym[toy_text]`")
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Taxi")
            if mode == "human":
                self.window = pygame.display.set_mode(WINDOW_SIZE)
            elif mode == "rgb_array":
                self.window = pygame.Surface(WINDOW_SIZE)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.taxi_imgs is None:
            file_names = [
                path.join(path.dirname(__file__), "img/cab_front.png"),
                path.join(path.dirname(__file__), "img/cab_rear.png"),
                path.join(path.dirname(__file__), "img/cab_right.png"),
                path.join(path.dirname(__file__), "img/cab_left.png"),
            ]
            self.taxi_imgs = [pygame.transform.scale(pygame.image.load(fn), self.cell_size) for fn in file_names]
        if self.passenger_img is None:
            file_name = path.join(path.dirname(__file__), "img/passenger.png")
            self.passenger_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
        if self.destination_img is None:
            file_name = path.join(path.dirname(__file__), "img/hotel.png")
            self.destination_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
            self.destination_img.set_alpha(170)
        if self.median_horiz is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_left.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_horiz.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_right.png"),
            ]
            self.median_horiz = [pygame.transform.scale(pygame.image.load(fn), self.cell_size) for fn in file_names]
        if self.median_vert is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_top.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_vert.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_bottom.png"),
            ]
            self.median_vert = [pygame.transform.scale(pygame.image.load(fn), self.cell_size) for fn in file_names]
        if self.background_img is None:
            file_name = path.join(path.dirname(__file__), "img/taxi_background.png")
            self.background_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)

        desc = self.desc
        for y in range(desc.shape[0]):
            for x in range(desc.shape[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1])
                self.window.blit(self.background_img, cell)
                if desc[y][x] == b"|" and (y == 0 or desc[y - 1][x] != b"|"):
                    self.window.blit(self.median_vert[0], cell)
                elif desc[y][x] == b"|" and (y == desc.shape[0] - 1 or desc[y + 1][x] != b"|"):
                    self.window.blit(self.median_vert[2], cell)
                elif desc[y][x] == b"|":
                    self.window.blit(self.median_vert[1], cell)
                elif desc[y][x] == b"-" and (x == 0 or desc[y][x - 1] != b"-"):
                    self.window.blit(self.median_horiz[0], cell)
                elif desc[y][x] == b"-" and (x == desc.shape[1] - 1 or desc[y][x + 1] != b"-"):
                    self.window.blit(self.median_horiz[2], cell)
                elif desc[y][x] == b"-":
                    self.window.blit(self.median_horiz[1], cell)

        for cell, color in zip(self.locs, self.locs_colors):
            color_cell = pygame.Surface(self.cell_size)
            color_cell.set_alpha(128)
            color_cell.fill(color)
            loc = self.get_surf_loc(cell)
            self.window.blit(color_cell, (loc[0], loc[1] + 10))

        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)
        if pass_idx < 4:
            self.window.blit(self.passenger_img, self.get_surf_loc(self.locs[pass_idx]))
        if self.lastaction in [0, 1, 2, 3]:
            self.taxi_orientation = self.lastaction
        dest_loc = self.get_surf_loc(self.locs[dest_idx])
        taxi_location = self.get_surf_loc((taxi_row, taxi_col))
        if dest_loc[1] <= taxi_location[1]:
            self.window.blit(self.destination_img, (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2))
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
        else:
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
            self.window.blit(self.destination_img, (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2))

        if mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    def get_surf_loc(self, map_loc):
        return ((map_loc[1] * 2 + 1) * self.cell_size[0], (map_loc[0] + 1) * self.cell_size[1])

    def _render_text(self):
        desc = self.desc.copy().tolist()
        outfile = StringIO()
        out = [[c.decode("utf-8") for c in line] for line in desc]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                f"  ({['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'][self.lastaction]})\n"
            )
        else:
            outfile.write("\n")
        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
