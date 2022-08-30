#!/usr/bin/python
import gym
import numpy as np
from typing import List
import math

from general_bayes_adaptive_pomdps.domains.speedy_tool_delivery.core import (
    AgentTurtlebot_v4,
    AgentFetch_v4,
    AgentHuman,
    BeliefWayPoint,
    MacroAction
)
from gym import spaces

class ObjSearchDelivery(gym.Env):

    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : 50
            }

    def __init__(self,
                 human_speeds,
                 n_objs=3,
                 n_each_obj=2,
                 TB_move_speed=0.6,
                 fetch_look_for_obj_tc=6,
                 render=False,
                 *args, **kwargs):

        self.n_agent = 1

        #-----------------basic settings for this domain
        # define the number of different objects needs for each human to finish the whole task
        self.n_objs = n_objs
        # total amount of each obj in the env
        self.n_each_obj = n_each_obj
        # define the number of steps for each human finishing the task 
        self.n_steps_human_task = self.n_objs + 1

        #-----------------def belief waypoints
        self.BWPs = []
        self.BWPs.append(BeliefWayPoint('ReceiveToolSpot', 0, 1.5, 3.5))
        self.BWPs.append(BeliefWayPoint('ToolDeliverySpot_0', 1, 6.0, 3.0))
        self.BWPs.append(BeliefWayPoint('ToolDeliverySpot_1', 2, 6.0, 2.0))

        self.BWPs_T0 = self.BWPs.copy()

        self.viewer = None
        self.max_timesteps = 500
        self.num_rooms = 3

        self.TB_move_speed = TB_move_speed
        self.fetch_look_for_obj_tc = fetch_look_for_obj_tc
        self.human_speed = [[human_speeds[0]]*4, [human_speeds[1]]*4]

        self.rendering = render

        self.coord_x_idx = 0
        self.coord_y_idx = 1
        self.timestep_idx = 2
        self.room_idx = 3

    def get_state(self):
        raise NotImplementedError

    def create_turtlebot_actions(self):
        raise NotImplementedError

    def create_fetch_actions(self):
        raise NotImplementedError

    def createAgents(self):
        raise NotImplementedError

    def createHumans(self):

        #-----------------initialize Three Humans
        Human0 = AgentHuman(0,
                            self.n_steps_human_task,
                            self.human_speed[0],
                            list(range(self.n_objs)))
        Human1 = AgentHuman(1,
                            self.n_steps_human_task,
                            self.human_speed[1],
                            list(range(self.n_objs)))

        self.humans = [Human0, Human1]
        self.n_human = len(self.humans)
        self.n_human_finished = []   # recording the number of human who has finished his own task

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        # reset the agents in this env
        self.createAgents()

        # reset the humans in this env
        for human in self.humans:
            human.reset()
        self.n_human_finished = []

        self.t = 0   # indicates the beginning of one episode, check _getobs()
        self.count_step = 0

        if self.rendering:
            self.render()

        return self._getobs()

    def _getobs(self):
        raise NotImplementedError

    def render(self, mode='human'):

        screen_width = 700
        screen_height = 500

        if self.viewer is None:
            import general_bayes_adaptive_pomdps.domains.speedy_tool_delivery.rendering as rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            line = rendering.Line((0.0, 0.0), (0.0, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, 0.0), (screen_width, 0.0))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((screen_width, 0.0), (screen_width, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, screen_height), (screen_width, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            #--------------------------------draw rooms' boundaries

            for i in range(0, 100, 2):
                line_tool_room = rendering.Line((350, i*5), (350, (i+1)*5))
                line_tool_room.set_color(0, 0, 0)
                line_tool_room.linewidth.stroke = 2
                self.viewer.add_geom(line_tool_room)

            for i in range(0, 80, 2):
                line_WA = rendering.Line((500+i*5, 300), (500+(i+1)*5, 300))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0, 0, 0)
                self.viewer.add_geom(line_WA)

            for i in range(0, 80, 2):
                line_WA = rendering.Line((500, 300+i*5), (500, 300+(i+1)*5))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0, 0, 0)
                self.viewer.add_geom(line_WA)

            for i in range(0, 80, 2):
                line_WA = rendering.Line((500+i*5, 200), (500+(i+1)*5, 200))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0, 0, 0)
                self.viewer.add_geom(line_WA)

            for i in range(0,40,2):
                line_WA = rendering.Line((500, 0+i*5), (500, 0+(i+1)*5))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0, 0, 0)
                self.viewer.add_geom(line_WA)

            #---------------------------draw BW0
            for i in range(len(self.BWPs)):
                BWP = rendering.make_circle(radius=6)
                BWP.set_color(178.0/255.0, 34.0/255.0, 34.0/255.0)
                x_coord, y_coord = self.BWPs[i].xcoord*100, self.BWPs[i].ycoord*100
                BWPtrans = rendering.Transform(translation=(x_coord, y_coord))
                BWP.add_attr(BWPtrans)
                self.viewer.add_geom(BWP)

            #-------------------------------draw table
            tablewidth = 60.0
            tableheight = 125.0
            l, r, t, b = -tablewidth/2.0, tablewidth/2.0, tableheight/2.0, -tableheight/2.0
            table = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            table.set_color(0.43, 0.28, 0.02)
            tabletrans = rendering.Transform(translation=(175, 250))
            table.add_attr(tabletrans)
            self.viewer.add_geom(table)

            tablewidth = 54.0
            tableheight = 119.0
            l, r, t, b = -tablewidth/2.0, tablewidth/2.0, tableheight/2.0, -tableheight/2.0
            table = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            table.set_color(0.67, 0.43, 0.02)
            tabletrans = rendering.Transform(translation=(175, 250))
            table.add_attr(tabletrans)
            self.viewer.add_geom(table)

            #-----------------------------draw Fetch
            fetch_x, fetch_y = self.agents[0].fetch.xcoord, self.agents[0].fetch.ycoord
            fetch = rendering.make_circle(radius=26)
            fetch.set_color(*(0.0, 0.0, 0.0))
            self.fetchtrans = rendering.Transform(translation=(fetch_x*100, fetch_y*100))
            fetch.add_attr(self.fetchtrans)
            self.viewer.add_geom(fetch)

            #-----------------------------draw Fetch
            fetch_c = rendering.make_circle(radius=23)
            fetch_c.set_color(*(0.5, 0.5, 0.5))
            self.fetchtrans_c = rendering.Transform(translation=(fetch_x*100, fetch_y*100))
            fetch_c.add_attr(self.fetchtrans_c)
            self.viewer.add_geom(fetch_c)

            #-----------------------------draw Fetch arms
            self.arm2 = rendering.FilledPolygon([(-5.0, -20.0,), (-5.0, 20.0),
                                                 (5.0, 20.0), (5.0, -20.0)])
            self.arm2.set_color(0.0, 0.0, 0.0)
            self.arm2trans = rendering.Transform(translation=(fetch_x*10000+49, fetch_y*100),
                                                 rotation=-90/180*np.pi)
            self.arm2.add_attr(self.arm2trans)
            self.viewer.add_geom(self.arm2)

            self.arm2_c = rendering.FilledPolygon([(-3.0, -18.0,), (-3.0, 18.0),
                                                   (3.0, 18.0), (3.0, -18.0)])
            self.arm2_c.set_color(0.5, 0.5, 0.5)
            self.arm2trans_c = rendering.Transform(translation=(fetch_x*10000+48, fetch_y*100),
                                                   rotation=-90/180*np.pi)
            self.arm2_c.add_attr(self.arm2trans_c)
            self.viewer.add_geom(self.arm2_c)

            self.arm1 = rendering.FilledPolygon([(-5.0, -38.0,), (-5.0, 38.0),
                                                 (5.0, 38.0), (5.0, -38.0)])
            self.arm1.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(108, 187), rotation=15/180*np.pi)
            self.arm1.add_attr(arm1trans)
            # self.viewer.add_geom(self.arm1)

            self.arm1_c = rendering.FilledPolygon([(-3.0, -36.0,), (-3.0, 36.0),
                                                   (3.0, 36.0), (3.0, -36.0)])
            self.arm1_c.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(108, 187), rotation=15/180*np.pi)
            self.arm1_c.add_attr(arm1trans)
            # self.viewer.add_geom(self.arm1_c)

            self.arm0 = rendering.FilledPolygon([(-5.0, -38.0,), (-5.0, 38.0),
                                                 (5.0, 38.0), (5.0, -38.0)])
            self.arm0.set_color(1.0, 1.0, 1.0)
            arm0trans = rendering.Transform(translation=(108, 313), rotation=-15/180*np.pi)
            self.arm0.add_attr(arm0trans)
            self.viewer.add_geom(self.arm0)

            self.arm0_c = rendering.FilledPolygon([(-3.0, -36.0,), (-3.0, 36.0),
                                                   (3.0, 36.0), (3.0, -36.0)])
            self.arm0_c.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(108, 313), rotation=-15/180*np.pi)
            self.arm0_c.add_attr(arm1trans)
            self.viewer.add_geom(self.arm0_c)

            #----------------------------draw Turtlebot_1
            turtlebot_1 = rendering.make_circle(radius=17.0)
            turtlebot_1.set_color(*(0.15, 0.65, 0.15))
            bot1_x, bot1_y = self.agents[0].xcoord, self.agents[0].ycoord
            self.turtlebot_1trans = rendering.Transform(translation=(bot1_x*100, bot1_y*100))
            turtlebot_1.add_attr(self.turtlebot_1trans)
            self.viewer.add_geom(turtlebot_1)

            turtlebot_1_c = rendering.make_circle(radius=14.0)
            turtlebot_1_c.set_color(*(0.0, 0.8, 0.4))
            self.turtlebot_1trans_c = rendering.Transform(translation=(bot1_x*100, bot1_y*100))
            turtlebot_1_c.add_attr(self.turtlebot_1trans_c)
            self.viewer.add_geom(turtlebot_1_c)

            #----------------------------draw human_0's status
            self.human0_progress_bar = []
            total_steps = self.humans[0].task_total_steps
            for i in range(total_steps):
                progress_bar = rendering.FilledPolygon([(-10, -10), (-10, 10),
                                                        (10, 10), (10, -10)])
                progress_bar.set_color(0.8, 0.8, 0.8)
                progress_bartrans = rendering.Transform(translation=(520+i*26, 480))
                progress_bar.add_attr(progress_bartrans)
                self.viewer.add_geom(progress_bar)
                self.human0_progress_bar.append(progress_bar)

            #----------------------------draw human_1's status
            self.human1_progress_bar = []
            total_steps = self.humans[1].task_total_steps
            for i in range(total_steps):
                progress_bar = rendering.FilledPolygon([(-10, -10), (-10, 10),
                                                        (10, 10), (10, -10)])
                progress_bar.set_color(0.8, 0.8, 0.8)
                progress_bartrans = rendering.Transform(translation=(520+i*26,20))
                progress_bar.add_attr(progress_bartrans)
                self.viewer.add_geom(progress_bar)
                self.human1_progress_bar.append(progress_bar)

        # draw each robot's status
        bot1_x, bot1_y = self.agents[0].xcoord, self.agents[0].ycoord
        self.turtlebot_1trans.set_translation(bot1_x*100, bot1_y*100)
        self.turtlebot_1trans_c.set_translation(bot1_x*100, bot1_y*100)

        fetch_x, fetch_y = self.agents[0].fetch.xcoord, self.agents[0].fetch.ycoord
        self.fetchtrans.set_translation(fetch_x*100, fetch_y*100)

        # reset human's progress bar
        for idx, bar in enumerate(self.human0_progress_bar):
            bar.set_color(0.8, 0.8, 0.8)
        for idx, bar in enumerate(self.human1_progress_bar):
            bar.set_color(0.8, 0.8, 0.8)

        # draw each human's status
        if self.humans[0].cur_step_time_left > 0:
            for idx, bar in enumerate(self.human0_progress_bar):
                if idx < self.humans[0].cur_step:
                    bar.set_color(0.0, 0.0, 0.0)
                if idx == self.humans[0].cur_step:
                    bar.set_color(0.0, 1.0, 0.0)
                    break
        else:
            for idx, bar in enumerate(self.human0_progress_bar):
                if idx <= self.humans[0].cur_step:
                    bar.set_color(0.0, 0.0, 0.0)

        # draw each human's status
        if self.humans[1].cur_step_time_left > 0:
            for idx, bar in enumerate(self.human1_progress_bar):
                if idx < self.humans[1].cur_step:
                    bar.set_color(0.0, 0.0, 0.0)
                if idx == self.humans[1].cur_step:
                    bar.set_color(0.0, 1.0, 0.0)
                    break
        else:
            for idx, bar in enumerate(self.human1_progress_bar):
                if idx <= self.humans[1].cur_step:
                    bar.set_color(0.0,0.0,0.0)

        # reset fetch arm
        self.arm0.set_color(1.0, 1.0, 1.0)
        self.arm0_c.set_color(1.0, 1.0, 1.0)
        self.arm1.set_color(1.0, 1.0, 1.0)
        self.arm1_c.set_color(1.0, 1.0, 1.0)

        self.arm2trans_c.set_translation(fetch_x*10000+48, fetch_y*100)
        self.arm2trans.set_translation(fetch_x*10000+49, fetch_y*100)

        self.pass_objs = 0

        if self.agents[0].fetch.cur_action is not None and \
                self.agents[0].fetch.cur_action_time_left <= 0.0 and \
                np.sum(self.agents[0].fetch.count_found_obj) <= self.n_objs*self.n_each_obj and \
                self.pass_objs < self.n_objs and \
                self.agents[0].fetch.tool_found:
            self.pass_objs += 1
            self.arm0.set_color(0.0, 0.0, 0.0)
            self.arm0_c.set_color(0.5, 0.5, 0.5)
            self.agents[0].fetch.tool_found = False

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class ObjSearchDelivery_v4(ObjSearchDelivery):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, *args, **kwargs):

        super(ObjSearchDelivery_v4, self).__init__(*args, **kwargs)

        self.create_turtlebot_actions()
        self.create_fetch_actions()
        self.createAgents()
        self.createHumans()

    def create_turtlebot_actions(self):

        # discrete room locations: [3]
        # which object in the basket: [3]*n_objs
        # which object are on the table: [2]* n_objs (only observable in the tool-room)
        # human 0 working step: [n_objs + 1] (only observable in the work-room)
        # human 1 working step: [n_objs + 1] (only observable in the work-room)
        self.observation_space = spaces.MultiDiscrete([3]
                                                      + [3]*(self.n_objs)
                                                      + [2]*(self.n_objs)
                                                      + [self.n_objs + 1]
                                                      + [self.n_objs + 1])

        #-----------------def macro-actions for Turtlebot
        self.T_MAs = []

        for i in range(self.n_objs):
            self.T_MAs.append(MacroAction(f'Get_Tool_{i}', i, expected_t_cost=None, ma_bwpterm=0))

        for i in range(2):
            self.T_MAs.append(MacroAction(f'Deliver_{i}', self.n_objs + i, expected_t_cost=None, ma_bwpterm=i+1))

    def known_dyn_coord_fcn(self, s, a: int, return_dist=False) -> List:
        """Implement a known part of the dynamics
        return_dist: if want to return categorical distributions for 
        features
        this version does not predict the next room location
        """

        x = s[self.coord_x_idx]
        y = s[self.coord_y_idx]
        room = s[self.room_idx]
        cur_timestep = s[self.timestep_idx]

        if a < self.n_objs:  # get tool actions
            goal_idx = 0  # tool-room
        elif a == self.n_objs:  # deliver 0
            goal_idx = 1
        elif a == self.n_objs + 1: # deliver 1
            goal_idx = 2
        else:
            raise NotImplementedError

        # self.BWPs.append(BeliefWayPoint('ReceiveToolSpot', 0, 1.5, 3.5))
        # self.BWPs.append(BeliefWayPoint('ToolDeliverySpot_0', 1, 6.0, 3.0))
        # self.BWPs.append(BeliefWayPoint('ToolDeliverySpot_1', 2, 6.0, 2.0))

        # 1. Travel time
        dest_x = self.BWPs[goal_idx].xcoord
        dest_y = self.BWPs[goal_idx].ycoord
        dist = round(np.sqrt((x - dest_x)**2 + (y - dest_y)**2), 2)

        travel_time = math.ceil(dist/self.TB_move_speed)
        next_timestep = cur_timestep + travel_time

        # 2: Wait Fetch time
        wait_time = 0
        if room == 0 and goal_idx == 0:
            if a < self.n_objs:
                if cur_timestep == 0:
                    wait_time = 5
                else:
                    wait_time = 6

        if room in [1, 2] and goal_idx == 0:
            wait_time = 5

        next_timestep += wait_time

        # 3: Normal step time
        step_time = 0
        
        if room == 1 and goal_idx == 1 and y == 3.0:
            if a == self.n_objs:
                step_time = 1

        if room == 2 and goal_idx == 2 and y == 2.0:
            if a == self.n_objs + 1:
                step_time = 1

        next_timestep += step_time

        next_timestep = int(next_timestep)

        if return_dist:
            next_timestep_onehot = np.zeros(self.max_timesteps)
            next_timestep_onehot[next_timestep] = 1.0
            return [dest_x, dest_y], [next_timestep_onehot]
        else:
            return [dest_x, dest_y], [next_timestep]

    def create_fetch_actions(self):
        self.action_space_F = spaces.Discrete(self.n_objs)

        #-----------------def macro-actions for Fetch Robot
        self.F_MAs = []
        for i in range(self.n_objs):
            F_MA = MacroAction(f'Find_Pass_Tool_{i}',
                               i,
                               expected_t_cost=6)
            self.F_MAs.append(F_MA)

    def createAgents(self):
        #-----------------initialize One Fetch Robot
        Fetch = AgentFetch_v4(2, 0.9, 1.8, self.F_MAs, self.n_objs, self.n_each_obj)

        #-----------------initialize Turtlebot
        Turtlebot = AgentTurtlebot_v4(0, 0.0, 0.0, self.BWPs_T0, self.T_MAs, self.n_objs,
                                      fetch=Fetch, speed=self.TB_move_speed)

        self.agents = [Turtlebot]

    def step(self, action):
        """
        Parameters
        ----------
        actions : int | List[..]
           The discrete macro-action index for one or more agents. 

        Returns
        -------
        observations : ndarry | List[..]
            A list of  each agent's macor-observation.
        done : bool
            Whether the current episode is over or not.
        info: dict{}
            "mac_done": binary(1/0) | List[..]
                whether the current macro_action is done or not.
            "cur_mac": int | list[..]
                "The current macro-action's indices"
        """

        cur_actions = []
        cur_actions_done = []

        # Turtlebot executes one step
        for idx, turtlebot in enumerate(self.agents[0:1]):
            # when the previous macro-action has not been finished, return the previous action id
            while not turtlebot.cur_action_done:
                turtlebot.step(action[idx], self.humans)
                cur_actions.append(action[idx])

                self.count_step += 1

                # each human executes one step
                for jdx, human in enumerate(self.humans):
                    if jdx in self.n_human_finished:
                        continue
                    human.cur_step_time_left -= 1.0
                    if human.cur_step_time_left <= 0.0 and human.next_requested_obj_obtained:
                        human.step()
                    if human.whole_task_finished:
                        self.n_human_finished.append(jdx)

                if self.rendering:
                    self.render()
                    time.sleep(0.25)

                    print(" ")
                    print("Actions list:")
                    print("Turtlebot  \t action \t\t{}".format(self.agents[0].cur_action.name))
                    print("           \t action_t_left \t\t{}".format(self.agents[0].cur_action_time_left))
                    print("           \t action_done \t\t{}".format(self.agents[0].cur_action_done))
                    print("           \t action_t_left \t\t{}".format(self.agents[0].fetch.cur_action_time_left))
                    print("           \t action_done \t\t{}".format(self.agents[0].fetch.cur_action_done))

                if self.rendering:
                    print("")
                    print("Humans status:")
                    for kdx, human in enumerate(self.humans):
                        print("Human" + str(kdx) + " \t\t cur_step  \t\t\t{}".format(human.cur_step))
                        print("      " + " \t\t cur_step_t_left  \t\t{}".format(human.cur_step_time_left))
                        print("      " + " \t\t next_request_obj  \t\t{}".format(human.next_request_obj_idx))
                        print("      " + " \t\t requested_obj_obtain  \t\t{}".format(human.next_requested_obj_obtained))
                        print("      " + " \t\t whole_task_finished  \t\t{}".format(human.whole_task_finished))
                        print(" ")

            # Reset to accept another macro-action
            turtlebot.cur_action_done = False

        # collect the info about the cur_actions and if they are finished
        for idx, agent in enumerate(self.agents):
            cur_actions_done.append(1 if agent.cur_action_done else 0)

        observations = self._getobs()

        return np.array(observations, dtype=int), 0, False, {}

    def _getobs(self):
        # OBSERVATION
        # discrete room locations: [3]
        # which object in the basket: [3]*n_objs
        # which object are on the table: [2]*n_objs (only observable in the tool-room)
        # human 0 working step: [n_objs + 1] (only observable in the work-room)
        # human 1 working step: [n_objs + 1] (only observable in the work-room)

        #--------------------get observations at the beginning of each episode
        if self.t == 0:
            observations = np.zeros(len(self.observation_space))
            self.t = 1

            return observations

        #---------------------get observations for the two turtlebots
        if self.rendering:
            print("")
            print("observations list:")

        observations = []
        agent = self.agents[0]
        # get observation about room location
        # tool-room
        if agent.xcoord < 3.5:
            room = 0
        # work-room
        else:
            if agent.ycoord > 2.5:
                room = 1
            else:
                room = 2
        obs_0 = [room]

        if self.rendering:
            print(f"Turtlebot" + " \t loc  \t\t\t {room}")

        # get observation about which tools are in the basket
        obs_1 = agent.objs_in_basket

        if self.rendering:
            print(f"          \t Basket_objs \t\t{obs_1}")

        # get observation about which tools are on the table (only available in the tool-room)
        obs_2 = np.zeros(self.n_objs)
        if len(agent.fetch.passed_tools) > 0 and room == 0:
            for tool_idx in range(self.n_objs):
                occurences = agent.fetch.passed_tools.count(tool_idx)
                if occurences == 2:
                    obs_2[tool_idx] = 1.0

        if self.rendering:
            print(f"          \t Table_objs \t\t{obs_2}")

        # get observation about the human's current step (only available in the work-room)
        if room in [0, 2]:
            obs_3 = [0]
        else:
            obs_3 = [self.humans[0].cur_step]

        if room in [0, 1]:
            obs_4 = [0]
        else:
            obs_4 = [self.humans[1].cur_step]

        if self.rendering:
            print(f"          \t Hm_0_cur_step \t\t{obs_3[0]}")
            print(f"          \t Hm_1_cur_step \t\t{obs_4[0]}")

        # combine all observations
        obs = np.hstack((obs_0, obs_1, obs_2, obs_3, obs_4))

        assert len(obs) == len(self.observation_space)

        observations.append(obs)

        return observations

    def get_state(self):
        # x_coord, y_coord
        # current timestep
        # discrete room locations: [2]
        # which object in the basket: [3]*(n_objs)
        # which object are on the table: [2]*(n_objs)
        # human 0 working step: [n_objs + 1]
        # human 1 working step: [n_objs + 1]

        state = []

        state.append(self.agents[0].xcoord)
        state.append(self.agents[0].ycoord)

       # current timestep
        state += [self.count_step]

        # turtlebot room [tool-room, work-room]
        assert len(self.agents) == 1
        # tool-room
        if self.agents[0].xcoord < 3.5:
            room = 0
        # work-room
        else:
            if self.agents[0].ycoord > 2.5:
                room = 1
            else:
                room = 2
        state.append(room)

        # which objects in basket
        # 0 means not in the basket
        state += self.agents[0].objs_in_basket.tolist()

        # which objects are at the staging area
        # 0 means is at
        objs = np.zeros(self.n_objs)
        if len(self.agents[0].fetch.passed_tools) > 0:
            for tool_idx in self.agents[0].fetch.passed_tools:
                occurences = self.agents[0].fetch.passed_tools.count(tool_idx)

                if occurences == 2:
                    objs[tool_idx] = 1.0

                if self.rendering:
                    print("Take Tool:", tool_idx)
                    print("In table: ", objs, self.agents[0].fetch.passed_tools)
        state += objs.tolist()

        if self.rendering:
            print(f"Len of state: {len(state)}")
            if state[0] == 0:
                print(f"Turtlebot pos: Tool-Room {state[0]}")
            else:
                print(f"Turtlebot pos: Work-Room {state[0]}")
            print(f"Objs in Turtlebot basket {state[1:4]}")
            print(f"Objs in staging area {state[4:7]}")
            print(f"Tool order {state[7:]}")

        # the human working step
        state += [self.humans[0].cur_step]
        state += [self.humans[1].cur_step]

        return np.array(state)

if __name__ == "__main__":
    import time

    env = ObjSearchDelivery_v4(human_speeds=[10, 15], render=False)

    step_delay = 1

    env.reset()

    optimal = True

    # Optimal
    if optimal:
        # for tool in [0, 1, 2]:
        #     # Get tool
        #     env.step([tool])
        #     print(env.get_state())

        #     # Deliver-1
        #     env.step([4])
        #     print(env.get_state())

        env.step([4])
        print(env.get_state())

        env.step([0])
        print(env.get_state())

        # env.step([1])
        # print(env.get_state())

        # env.step([2])
        # print(env.get_state())

        # env.step([2])
        # print(env.get_state())

        # env.step([2])
        # print(env.get_state())

        # # Deliver
        # for tool in [0, 1, 2]:
        #     # Get tool
        #     env.step([tool])
        #     print(env.get_state())

        #     # Deliver-0
        #     env.step([3])
        #     print(env.get_state())

    # Sub-optimal
    else:
        # Get all tools
        env.step([0])

        env.step([1])

        env.step([2])

        env.step([0])

        env.step([1])

        env.step([2])

        for _ in range(6):
            env.step([3])