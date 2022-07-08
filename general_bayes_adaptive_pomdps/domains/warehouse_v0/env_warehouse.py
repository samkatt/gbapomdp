from ast import Del
import numpy as np
import random
import cv2
import time
import one_to_one

Go_to_WR = 0
Go_to_TR = 1
Get_tool_1 = 2
Get_tool_2 = 3
Get_tool_3 = 4
Get_tool_4 = 5
Deliver_tool = 6
No_Op = 7

class Human(object):
    WORKING = 0
    WAITING_TOOL = 1
    def __init__(self, time_cost=3):
        self.time_step = 0
        self.time_cost = time_cost
        self.current_status = self.WORKING

    def update(self):
        if self.current_status == self.WORKING:
            self.time_step += 1

            if self.time_step == self.time_cost:
                self.current_status = self.WAITING_TOOL
                self.time_step = 0

    def start_working(self):
        '''this function is called when a desired tool is delivered'''
        assert self.current_status == self.WAITING_TOOL
        self.current_status = self.WORKING

    def is_waiting(self):
        return self.current_status == self.WAITING_TOOL

    def set(self, current_status, time_step):
        self.current_status = current_status
        self.time_step = time_step

class TurtleBot(object):
    AT_TOOL_ROOM = 0
    AT_WORK_ROOM = 1
    ROOM_BOUNDARY = 4

    def __init__(self):

        # initialize at work room
        self.pos = [2, 6]

    def set_room(self, room_id):
        # tool room
        if room_id == 0:
            self.pos = [2, 0]
        # work room
        else:
            self.pos = [2, 6]

    def move(self, action):
        if action == Go_to_WR:
            self.go_to_wr()

        if action == Go_to_TR:
            self.go_to_tr()

    def go_to_tr(self):
        self.pos = [2, 0]

    def go_to_wr(self):
        self.pos = [2, 6]

    def get_location(self):
        if self.pos[1] > self.ROOM_BOUNDARY:
            return self.AT_WORK_ROOM
        else:
            return self.AT_TOOL_ROOM

class EnvWareHouse(object):
    def __init__(self):
        self.reset()

    def step(self, action):
        if action[0] < Get_tool_1:
            saved_class = self._clear_bot_box()
            self.bot.move(action[0])
            self.raw_occupancy[self.bot.pos[0], self.bot.pos[1]] = saved_class

        elif Get_tool_1 <= action[0] <= Get_tool_4:
            self._get_tool(action[0])

        elif action[0] == Deliver_tool:
            self._drop_tool()

        elif action[0] == No_Op:
            pass

        else:
            raise NotImplementedError

        self.human.update()

        if self.human.is_waiting():
            self.raw_occupancy[3, 4] = 8
        else:
            self.raw_occupancy[3, 4] = 7


    def _drop_tool(self):
        bot = self.bot

        # check if at work room
        assert bot.pos[0] == 2
        at_work_room = (bot.get_location() == bot.AT_WORK_ROOM)

        # check if carrying a tool
        current_tool = self.raw_occupancy[bot.pos[0], bot.pos[1]] - 2
        carrying_tool = current_tool > 0

        # check if the tool is desired
        good_tool = (current_tool == self.desired_tool + 1)

        # check if human is not working
        human_waiting = self.human.is_waiting()

        if at_work_room and carrying_tool and good_tool and human_waiting:
            # mark the bot to be free
            self.raw_occupancy[bot.pos[0], bot.pos[1]] = 2

            # mark the allocated tool
            self.raw_occupancy[0, int(5 + current_tool)] = 1

            # increase the desired tool
            self.desired_tool += 1

            assert self.desired_tool <= 4

            # tell human to restart working
            self.human.start_working()


    def _get_tool(self, action):
        assert Get_tool_1 <= action <= Get_tool_4
        tool_idx = action - 2
        bot = self.bot

        # check if tool is available
        tool_avail = (self.raw_occupancy[0, tool_idx] == 1)

        # check if at tool room
        assert bot.pos[0] == 2
        at_tool_room = bot.get_location() == bot.AT_TOOL_ROOM

        # check if the agent is free
        bot_free = (self.raw_occupancy[bot.pos[0], bot.pos[1]] == 2)

        if tool_avail and bot_free and at_tool_room:
            # clear the chosen tool
            self.raw_occupancy[0, tool_idx] = 0

            # indicate the tool carried by the bot
            self.raw_occupancy[bot.pos[0], bot.pos[1]] = 3 + tool_idx


    def reset(self, state=None):
        # state: (location, current-tool) + (human-status, desired-tool, time-step)
        # location: tool-room or work-room (2)
        # current-tool: 0, 1, 2, 3, 4 (5)
        # human-status: working or waiting (2)
        # desired-tool: 1, 2, 3, 4, 5 (5)
        # human time step: 0, 1, 2 (3)
        self.raw_occupancy = np.zeros((4, 10))

        if state is not None:
            assert len(state) == 5
        if state is not None:
            (bot_loc_id, bot_tool) = state[:2]
            (human_status, desired_tool) = state[2:4]
            human_step_cnt = state[4]
        else:
            (bot_loc_id, bot_tool) = (1, 0) # (at workroom, free)
            (human_status, desired_tool) = (0, 0) # (working, 0)
            human_step_cnt = 0

        # first blacken all tool positions
        for i in range(4):
            self.raw_occupancy[0, i] = 1

        # a tool is carrying by turtlebot, whiten the corresponding tool positions
        assert bot_tool <= 4
        if bot_tool > 0:
            self.raw_occupancy[0, bot_tool - 1] = 0

        # whiten tool positions depending on the current desired tool
        # i.e., if the current desired tool is 2 then the tool positions 0, 1 should be clear
        for i in range(desired_tool):
            self.raw_occupancy[0, i] = 0

        # color tool delivered position
        assert 0 <= desired_tool <= 4
        for i in range(6, 6 + desired_tool):
            self.raw_occupancy[0, i] = 1

        # human position
        assert human_status in [Human.WAITING_TOOL, Human.WORKING]
        self.raw_occupancy[3, 4] = 7 + human_status

        self.bot = TurtleBot()

        self.bot.set_room(bot_loc_id)

        self.human = Human()
        self.human.set(human_status, human_step_cnt)

        self.raw_occupancy[self.bot.pos[0], self.bot.pos[1]] = 2 + bot_tool

        self.desired_tool = desired_tool

    def get_state(self):
        """
        Request environmental state

        Return
        ------
        numpy.array
        """

        bot_state = [self.bot.get_location(), self.raw_occupancy[self.bot.pos[0], self.bot.pos[1]] - 2]

        return np.array(bot_state + [self.human.current_status, self.desired_tool, self.human.time_step], dtype=int)

    def get_obs(self):
        """
        Request environmental state

        Return
        ------
        numpy.array
        """

        bot_state = [self.bot.get_location(), self.raw_occupancy[self.bot.pos[0], self.bot.pos[1]] - 2]

        return np.array(bot_state + [self.human.current_status, self.desired_tool, self.human.time_step], dtype=int)

    def is_done(self):
        # Human must finish using the final tool
        return self.desired_tool == 4 and self.human.time_step == 2

    def is_terminal_state(self, state):
        # Human must finish using the final tool
        assert len(state) == 5
        desired_tool = state[3]
        human_timestep = state[4]
        return desired_tool == 4 and human_timestep == 2

    def render(self):
        obs = np.ones((4 * 60, 10 * 60, 3))

        for i in range(4):
            for j in range(10):
                if self.raw_occupancy[i, j] == 1:
                    cv2.rectangle(obs, (j*60, i*60), (j*60+60, i*60+60), (0, 0, 0), -1)

                # Carrying no tool
                if self.raw_occupancy[i, j] == 2:
                    cv2.rectangle(obs, (j*60, i*60), (j*60+60, i*60+60), (255, 0, 0), 2)
                    cv2.putText(obs, 'F', (j*60 + 25, i*60 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

                # Carrying tool #1
                if self.raw_occupancy[i, j] == 3:
                    cv2.rectangle(obs, (j*60, i*60), (j*60+60, i*60+60), (255, 0, 0), 2)
                    cv2.putText(obs, '0', (j*60 + 25, i*60 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

                # Carrying tool #2
                if self.raw_occupancy[i, j] == 4:
                    cv2.rectangle(obs, (j*60, i*60), (j*60+60, i*60+60), (255, 0, 0), 2)
                    cv2.putText(obs, '1', (j*60 + 25, i*60 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

                # Carrying tool #3
                if self.raw_occupancy[i, j] == 5:
                    cv2.rectangle(obs, (j*60, i*60), (j*60+60, i*60+60), (255, 0, 0), 2)
                    cv2.putText(obs, '2', (j*60 + 25, i*60 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

                # Carrying tool #4
                if self.raw_occupancy[i, j] == 6:
                    cv2.rectangle(obs, (j*60, i*60), (j*60+60, i*60+60), (255, 0, 0), 2)
                    cv2.putText(obs, '3', (j*60 + 25, i*60 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

                # Human is working
                if self.raw_occupancy[i, j] == 7:
                    cv2.rectangle(obs, (j*60, i*60), (j*60+60, i*60+60), (255, 255, 0), 2)
                    cv2.putText(obs, f'W{self.human.time_step}', (j*60 + 15, i*60 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

                # Human is waiting for a tool
                if self.raw_occupancy[i, j] == 8:
                    cv2.rectangle(obs, (j*60, i*60), (j*60+60, i*60+60), (255, 255, 0), 2)
                    cv2.putText(obs, f'S{self.human.time_step}', (j*60 + 15, i*60 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

        cv2.imshow('image', obs)
        cv2.waitKey(10)

    def _clear_bot_box(self):
        bot = self.bot
        saved_class = self.raw_occupancy[bot.pos[0], bot.pos[1]]
        self.raw_occupancy[bot.pos[0], bot.pos[1]] = 0
        return saved_class

if __name__ == "__main__":

    sem_action_space = one_to_one.JointNamedSpace(
        a1=one_to_one.RangeSpace(8),  # 0: move forward, 1: turn left, 2: turn right, 3: stay
    )
    action_lst = list(sem_action_space.elems)
    index2str = ['Go_to_WR', 'Go_to_TR', 'Get_Tool_0', 'Get_Tool_1',
                'Get_Tool_2', 'Get_Tool_3', 'Deliver', 'No_Op']

    def action_to_str(action_idx):
        actions = action_lst[action_idx]
        return f"agent 0: {index2str[actions.a1.value]},\
        agent 1: {index2str[actions.a2.value]}"

    env = EnvWareHouse()
    # env.reset()
    # env.render()
    # time.sleep(2)

    # action_list = [Go_to_TR, Get_tool_1, Go_to_WR, Deliver_tool, Go_to_TR, Get_tool_2, Go_to_WR, Deliver_tool]

    # action_list += [Go_to_TR, Get_tool_3, Go_to_WR, Deliver_tool, Go_to_TR, Get_tool_4, Go_to_WR, Deliver_tool, No_Op]

    # for action in action_list:
    #     env.step([action])
    #     env.render()
    #     time.sleep(1)

    # assert env.is_done() is True

    env.reset(state=[1, 0, 0, 0, 0])
    env.step([0])
    print(env.get_state())
    env.step([0])
    print(env.get_state())
    env.step([4])
    print(env.get_state())
    env.step([3])
    print(env.get_state())

