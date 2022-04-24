#!/usr/bin/python

import gym
import numpy as np

from gym import spaces
from general_bayes_adaptive_pomdps.domains.box_pushing.box_pushing_core import Agent, Box

DIRECTION = [(0, 1), (1, 0), (0, -1), (-1, 0)]
ACTIONS = ["Move_Forward", "Turn_L", "Turn_R", "Stay"]


class SmallBoxPushing(gym.Env):

    """
    Small Box Pushing Environment: agents aim to separately push small boxes to a goal area
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(
        self,
        grid_dim=(4, 4),
        n_agent=2,
        terminate_step=100,
        terminal_reward_only=False,
        small_box_reward=10,
        *args,
        **kwargs
    ):

        # env generic settings
        assert n_agent <= 6 and n_agent <= grid_dim[0], "Too many agents"
        self.n_agent = n_agent
        self.action_space = [spaces.Discrete(4)] * self.n_agent
        self.observation_space = [spaces.MultiBinary(5)] * self.n_agent
        self.xlen, self.ylen = grid_dim
        self.terminate_step = terminate_step
        self.terminal_reward_only = terminal_reward_only
        self.small_box_reward = small_box_reward
        self.viewer = None
        # create agents and boxes
        self.createAgents()
        self.createBoxes()

    @property
    def obs_size(self):
        return [self.observation_space[0].n] * self.n_agent

    @property
    def n_action(self):
        return [a.n for a in self.action_spaces]

    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)

    @property
    def action_spaces(self):
        return self.action_space

    def createAgents(self, agents_pos=None):

        if self.ylen >= 8.0:
            A0 = Agent(0, 1.5, 1.5, 1, (self.xlen, self.ylen))
            A1 = Agent(1, self.xlen - 1.5, 1.5, 3, (self.xlen, self.ylen))
        elif self.ylen == 6.0:
            A0 = Agent(0, 0.5, 1.5, 1, (self.xlen, self.ylen))
            A1 = Agent(1, 5.5, 1.5, 3, (self.xlen, self.ylen))
        else:
            A0 = Agent(0, 0.5, 0.5, 1, (self.xlen, self.ylen))
            A1 = Agent(1, 3.5, 0.5, 3, (self.xlen, self.ylen))

        if agents_pos is not None:
            # orientation
            assert 0 <= agents_pos[2] < 4
            assert 0 <= agents_pos[5] < 4

            # XY positions of 2 agents
            assert 0 <= agents_pos[0] < self.xlen
            assert 0 <= agents_pos[1] < self.ylen
            assert 0 <= agents_pos[3] < self.xlen
            assert 0 <= agents_pos[4] < self.ylen

            A0 = Agent(0, agents_pos[0] + 0.5, agents_pos[1] + 0.5, agents_pos[2], (self.xlen, self.ylen))
            A1 = Agent(1, agents_pos[3] + 0.5, agents_pos[4] + 0.5, agents_pos[5], (self.xlen, self.ylen))

        self.agents = [A0, A1]

        for i in range(self.n_agent - 2):
            if self.ylen >= 6.0:
                if i % 2 == 0:
                    A = Agent(
                        i + 2,
                        self.xlen / 2.0 - (i // 2 * 1.0 + 0.5),
                        1.5,
                        1,
                        (self.xlen, self.ylen),
                    )
                else:
                    A = Agent(
                        i + 2,
                        self.xlen / 2.0 + (i // 2 * 1.0 + 0.5),
                        1.5,
                        3,
                        (self.xlen, self.ylen),
                    )
            else:
                if i % 2 == 0:
                    A = Agent(
                        i + 2,
                        self.xlen / 2.0 - (i // 2 * 1.0 + 0.5),
                        0.5,
                        1,
                        (self.xlen, self.ylen),
                    )
                else:
                    A = Agent(
                        i + 2,
                        self.xlen / 2.0 + (i // 2 * 1.0 + 0.5),
                        0.5,
                        3,
                        (self.xlen, self.ylen),
                    )
            self.agents.append(A)

    def createBoxes(self, boxes_pos=None):

        if self.ylen >= 8.0:
            SB_0 = Box(0, 1.5, self.ylen / 2 + 0.5, 1.0, 1.0)
            SB_1 = Box(1, self.ylen - 1.5, self.ylen / 2 + 0.5, 1.0, 1.0)
        elif self.ylen == 6.0:
            SB_0 = Box(0, 0.5, self.ylen / 2 + 0.5, 1.0, 1.0)
            SB_1 = Box(1, 5.5, self.ylen / 2 + 0.5, 1.0, 1.0)
        else:
            SB_0 = Box(0, 0.5, self.ylen / 2 + 0.5, 1.0, 1.0)
            SB_1 = Box(1, 3.5, self.ylen / 2 + 0.5, 1.0, 1.0)

        if boxes_pos is not None:
            SB_0 = Box(0, boxes_pos[0] + 0.5, boxes_pos[1] + 0.5, 1.0, 1.0)
            SB_1 = Box(1, boxes_pos[2] + 0.5, boxes_pos[3] + 0.5, 1.0, 1.0)

        self.boxes = [SB_0, SB_1]

        for i in range(self.n_agent - 2):
            if i % 2 == 0:
                SB = Box(
                    1, self.xlen / 2.0 - (i // 2 * 1.0 + 0.5), self.ylen / 2 + 0.5, 1.0, 1.0
                )
            else:
                SB = Box(
                    1, self.xlen / 2.0 + (i // 2 * 1.0 + 0.5), self.ylen / 2 + 0.5, 1.0, 1.0
                )
            self.boxes.append(SB)

    def reset(self, debug=False, agents_pos=None, boxes_pos=None):
        """
        Parameter
        ---------
        debug: bool
            if debug, will render the environment for visualization
        entity_pos:
            if not None, will specify the positions of agents and boxes

        Return
        ------
        List[numpy.array]
            a list of agents' observations
        """
        self.createAgents(agents_pos=agents_pos)
        self.createBoxes(boxes_pos=boxes_pos)

        if debug:
            self.render()

        return self._getobs()

    def step(self, actions, debug=True):

        # assume current step does not terminate
        terminate = 0

        if self.terminal_reward_only:
            rewards = 0.0
        else:
            rewards = -0.1
        for idx, agent in enumerate(self.agents):
            reward = agent.step(actions[idx], self.boxes)
            if not self.terminal_reward_only:
                rewards += reward

        # check whether any box is pushed to the goal area and update reward
        reward = 0.0
        for idx, box in enumerate(self.boxes):
            if box.ycoord == self.ylen - 0.5:
                terminate = 1
                reward = reward + self.small_box_reward
        rewards += reward

        if debug:
            self.render()
            print(" ")
            print("Actions list:")
            for ag in self.agents:
                print(
                    "Agent_" + str(ag.idx) + " \t action \t\t{}".format(ACTIONS[ag.cur_action])
                )
                print(" ")

        observations = self._getobs(debug)

        return observations, [rewards] * self.n_agent, [bool(terminate)] * self.n_agent, {}

    def _getobs(self, debug=False):
        """
        Return
        ------
        List[numpy.arry]
            a list of agents' observations
        """

        if debug:
            print("")
            print("Observations list:")

        observations = []
        for idx, agent in enumerate(self.agents):

            # assume empty front
            obs = 0

            # observe small box
            for box in self.boxes:
                if (
                    box.xcoord == agent.xcoord + DIRECTION[agent.ori][0]
                    and box.ycoord == agent.ycoord + DIRECTION[agent.ori][1]
                ):
                    obs = 1
                    break

            # observe wall
            if (
                agent.xcoord + DIRECTION[agent.ori][0] > self.xlen
                or agent.xcoord + DIRECTION[agent.ori][0] < 0.0
                or agent.ycoord + DIRECTION[agent.ori][1] > self.ylen
                or agent.ycoord + DIRECTION[agent.ori][1] < 0.0
            ):
                obs = 2
                break

            # observe agent
            for teamate_idx in range(self.n_agent):
                if teamate_idx != idx:
                    if (
                        agent.xcoord + DIRECTION[agent.ori][0]
                        == self.agents[teamate_idx].xcoord
                    ) and (
                        agent.ycoord + DIRECTION[agent.ori][1]
                        == self.agents[teamate_idx].ycoord
                    ):
                        obs = 3
                        break

            if debug:
                print("Agent_" + str(idx) + " \t obs  \t\t{}".format(obs))

            observations.append(obs)

        return observations

    def get_obs(self):
        """
        Request the current observation

        Return
        ------
        list
            the current type of the cell in-front (empty, box, wall, agent)
        """

        return np.array(self._getobs(), dtype=int)

    def get_state(self):
        """
        Request environmental state

        Return
        ------
        numpy.array
            the positions of all entities
        """

        positions = []
        for ag in self.agents:
            positions.append(ag.xcoord - 0.5)
            positions.append(ag.ycoord - 0.5)
            positions.append(ag.ori)
        for bx in self.boxes:
            positions.append(bx.xcoord - 0.5)
            positions.append(bx.ycoord - 0.5)

        return np.array(positions, dtype=int)

    def render(self, mode="human"):

        screen_width = 8 * 100
        screen_height = 8 * 100

        scale = 8 / self.ylen

        agent_size = 30.0
        agent_in_size = 25.0
        agent_clrs = [
            ((0.15, 0.65, 0.15), (0.0, 0.8, 0.4)),
            ((0.15, 0.15, 0.65), (0.0, 0.4, 0.8)),
            ((0.2, 0.0, 0.4), (0.5, 0.0, 1.0)),
            ((0.12, 0.12, 0.12), (0.5, 0.5, 0.5)),
            ((0, 0.4, 0.4), (0.0, 1.0, 1.0)),
            ((0.4, 0.2, 0.0), (1.0, 0.5, 0.0)),
        ]

        small_box_size = 85.0
        small_box_clrs = [(0.43, 0.28, 0.02), (0.67, 0.43, 0.02)]
        small_box_in_size = 75.0

        if self.viewer is None:
            from general_bayes_adaptive_pomdps.domains.box_pushing import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # -------------------draw line-----------------
            for l in range(1, self.ylen):
                line = rendering.Line((0.0, l * 100 * scale), (screen_width, l * 100 * scale))
                line.linewidth.stroke = 4
                line.set_color(0.50, 0.50, 0.50)
                self.viewer.add_geom(line)

            for l in range(1, self.ylen):
                line = rendering.Line((l * 100 * scale, 0.0), (l * 100 * scale, screen_width))
                line.linewidth.stroke = 4
                line.set_color(0.50, 0.50, 0.50)
                self.viewer.add_geom(line)

            line = rendering.Line((0.0, 0.0), (0.0, screen_width))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, 0.0), (screen_width, 0.0))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((screen_width, 0.0), (screen_width, screen_width))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, screen_width), (screen_width, screen_width))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            # -------------------draw goal
            goal = rendering.FilledPolygon(
                [
                    (-(screen_width - 8) / 2.0, (-50 + 2) * scale),
                    (-(screen_width - 8) / 2.0, (50 - 2) * scale),
                    ((screen_width - 8) / 2.0, (50 - 2) * scale),
                    ((screen_width - 8) / 2.0, -(50 - 2) * scale),
                ]
            )
            goal.set_color(1.0, 1.0, 0.0)
            goal_trans = rendering.Transform(
                translation=(screen_width / 2.0, (self.ylen - 0.5) * 100 * scale)
            )
            goal.add_attr(goal_trans)
            self.viewer.add_geom(goal)

            # -------------------draw small box
            self.small_box_trans = []
            for box in self.boxes:
                small_box = rendering.FilledPolygon(
                    [
                        (-small_box_size / 2.0 * scale, -small_box_size / 2.0 * scale),
                        (-small_box_size / 2.0 * scale, small_box_size / 2.0 * scale),
                        (small_box_size / 2.0 * scale, small_box_size / 2.0 * scale),
                        (small_box_size / 2.0 * scale, -small_box_size / 2.0 * scale),
                    ]
                )
                small_box.set_color(*small_box_clrs[0])
                self.small_box_trans.append(
                    rendering.Transform(
                        translation=(box.xcoord * 100 * scale, box.ycoord * 100 * scale)
                    )
                )
                small_box.add_attr(self.small_box_trans[-1])
                self.viewer.add_geom(small_box)

            self.small_box_in_trans = []
            for box in self.boxes:
                small_box_in = rendering.FilledPolygon(
                    [
                        (-small_box_in_size / 2.0 * scale, -small_box_in_size / 2.0 * scale),
                        (-small_box_in_size / 2.0 * scale, small_box_in_size / 2.0 * scale),
                        (small_box_in_size / 2.0 * scale, small_box_in_size / 2.0 * scale),
                        (small_box_in_size / 2.0 * scale, -small_box_in_size / 2.0 * scale),
                    ]
                )
                small_box_in.set_color(*small_box_clrs[1])
                self.small_box_in_trans.append(
                    rendering.Transform(
                        translation=(box.xcoord * 100 * scale, box.ycoord * 100 * scale)
                    )
                )
                small_box_in.add_attr(self.small_box_in_trans[-1])
                self.viewer.add_geom(small_box_in)

            # -------------------draw agent
            self.agent_trans = []
            for ag in self.agents:
                agent = rendering.make_circle(radius=agent_size * scale)
                agent.set_color(*agent_clrs[ag.idx][0])
                self.agent_trans.append(
                    rendering.Transform(
                        translation=(ag.xcoord * 100 * scale, ag.ycoord * 100 * scale)
                    )
                )
                agent.add_attr(self.agent_trans[-1])
                self.viewer.add_geom(agent)

            self.agent_in_trans = []
            for ag in self.agents:
                agent_in = rendering.make_circle(radius=agent_in_size * scale)
                agent_in.set_color(*agent_clrs[ag.idx][1])
                self.agent_in_trans.append(
                    rendering.Transform(
                        translation=(ag.xcoord * 100 * scale, ag.ycoord * 100 * scale)
                    )
                )
                agent_in.add_attr(self.agent_in_trans[-1])
                self.viewer.add_geom(agent_in)

            # -------------------draw agent sensor
            sensor_size = 20.0
            sensor_in_size = 14.0
            sensor_clrs = ((0.65, 0.15, 0.15), (1.0, 0.2, 0.2))

            self.sensor_trans = []
            for idx in range(self.n_agent):
                sensor = rendering.FilledPolygon(
                    [
                        (-sensor_size / 2.0 * scale, -sensor_size / 2.0 * scale),
                        (-sensor_size / 2.0 * scale, sensor_size / 2.0 * scale),
                        (sensor_size / 2.0 * scale, sensor_size / 2.0 * scale),
                        (sensor_size / 2.0 * scale, -sensor_size / 2.0 * scale),
                    ]
                )
                sensor.set_color(*sensor_clrs[0])
                self.sensor_trans.append(
                    rendering.Transform(
                        translation=(
                            self.agents[idx].xcoord * 100 * scale
                            + (agent_size) * DIRECTION[self.agents[idx].ori][0] * scale,
                            self.agents[idx].ycoord * 100 * scale
                            + (agent_size) * DIRECTION[self.agents[idx].ori][1] * scale,
                        )
                    )
                )
                sensor.add_attr(self.sensor_trans[-1])
                self.viewer.add_geom(sensor)

            self.sensor_in_trans = []
            for idx in range(self.n_agent):
                sensor_in = rendering.FilledPolygon(
                    [
                        (-sensor_in_size / 2.0 * scale, -sensor_in_size / 2.0 * scale),
                        (-sensor_in_size / 2.0 * scale, sensor_in_size / 2.0 * scale),
                        (sensor_in_size / 2.0 * scale, sensor_in_size / 2.0 * scale),
                        (sensor_in_size / 2.0 * scale, -sensor_in_size / 2.0 * scale),
                    ]
                )
                sensor_in.set_color(*sensor_clrs[1])
                self.sensor_in_trans.append(
                    rendering.Transform(
                        translation=(
                            self.agents[idx].xcoord * 100 * scale
                            + (agent_size) * DIRECTION[self.agents[idx].ori][0] * scale,
                            self.agents[idx].ycoord * 100 * scale
                            + (agent_size) * DIRECTION[self.agents[idx].ori][1] * scale,
                        )
                    )
                )
                sensor_in.add_attr(self.sensor_in_trans[-1])
                self.viewer.add_geom(sensor_in)

        for idx, trans in enumerate(self.small_box_trans):
            trans.set_translation(
                self.boxes[idx].xcoord * 100 * scale, self.boxes[idx].ycoord * 100 * scale
            )
        for idx, trans in enumerate(self.small_box_in_trans):
            trans.set_translation(
                self.boxes[idx].xcoord * 100 * scale, self.boxes[idx].ycoord * 100 * scale
            )

        for idx, trans in enumerate(self.agent_trans):
            trans.set_translation(
                self.agents[idx].xcoord * 100 * scale, self.agents[idx].ycoord * 100 * scale
            )
        for idx, trans in enumerate(self.agent_in_trans):
            trans.set_translation(
                self.agents[idx].xcoord * 100 * scale, self.agents[idx].ycoord * 100 * scale
            )

        for idx, trans in enumerate(self.sensor_trans):
            trans.set_translation(
                self.agents[idx].xcoord * 100 * scale
                + (agent_size) * DIRECTION[self.agents[idx].ori][0] * scale,
                self.agents[idx].ycoord * 100 * scale
                + (agent_size) * DIRECTION[self.agents[idx].ori][1] * scale,
            )
            trans.set_rotation(0.0)
        for idx, trans in enumerate(self.sensor_in_trans):
            trans.set_translation(
                self.agents[idx].xcoord * 100 * scale
                + (agent_size) * DIRECTION[self.agents[idx].ori][0] * scale,
                self.agents[idx].ycoord * 100 * scale
                + (agent_size) * DIRECTION[self.agents[idx].ori][1] * scale,
            )
            trans.set_rotation(0.0)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")