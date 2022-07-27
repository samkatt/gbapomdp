from ast import Del
import numpy as np
import cv2
import time
import one_to_one

from general_bayes_adaptive_pomdps.core import (
    TerminalState,
    InvalidState
)

Go_to_WR = 0
Go_to_TR = 1
Get_tool_1 = 2
Get_tool_2 = 3
Get_tool_3 = 4
Get_tool_4 = 5
Deliver_tool = 6
No_Op = 7

class Human(object):
    WAITING_TOOL = 0
    WORKING = 1
    def __init__(self, time_cost=3):
        self.time_step = 0
        self.time_cost = time_cost
        self.current_status = self.WAITING_TOOL

    def update(self):
        if self.current_status == self.WORKING:
            self.time_step += 1

            # finish using the current tool, therefore, waiting for another tool
            # and reset the time step
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
    AT_WORK_ROOM = 0
    AT_TOOL_ROOM = 1
    ROOM_BOUNDARY = 4

    POS_WORK_ROOM = [2, 6]
    POS_TOOL_ROOM = [2, 0]

    def __init__(self):

        # initialize at work room
        self.pos = self.POS_WORK_ROOM

    def set_room(self, room_id):
        # tool room
        if room_id == 0:
            self.pos = self.POS_WORK_ROOM
        # work room
        else:
            self.pos = self.POS_TOOL_ROOM

    def move(self, act):
        if act == Go_to_WR:
            self.pos = self.POS_WORK_ROOM

        elif act == Go_to_TR:
            self.pos = self.POS_TOOL_ROOM

        else:
            raise NotImplementedError

    def get_location(self):
        assert self.pos[0] == 2
        if self.pos[1] > self.ROOM_BOUNDARY:
            return self.AT_WORK_ROOM
        else:
            return self.AT_TOOL_ROOM

class EnvWareHouse(object):
    def __init__(self, correct_tool_order):
        assert len(correct_tool_order) == 4
        self.correct_tool_order = correct_tool_order
        self.reset()

    def step(self, action):
        if action[0] < Get_tool_1:
            saved_class = self._clear_bot_box()
            self.bot.move(action[0])
            self.raw_occupancy[self.bot.pos[0], self.bot.pos[1]] = saved_class

        elif Get_tool_1 <= action[0] <= Get_tool_4:
            self._get_tool(action[0])

        elif action[0] == Deliver_tool:
            self._deliver_tool()

        elif action[0] == No_Op:
            pass

        else:
            raise NotImplementedError

        self.human.update()

        if self.human.is_waiting():
            self.raw_occupancy[3, 4] = 7
        else:
            self.raw_occupancy[3, 4] = 8

    def _deliver_tool(self):
        bot = self.bot

        # check if at work room
        assert bot.pos[0] == 2
        at_work_room = (bot.get_location() == bot.AT_WORK_ROOM)

        # check if carrying a tool
        current_tool = self.raw_occupancy[bot.pos[0], bot.pos[1]] - 2
        carrying_tool = current_tool > 0

        # check if the tool is desired
        if self.task_stage < 4:
            good_tool = (current_tool == self.correct_tool_order[self.task_stage])
        else:
            # if task_stage is at least 4, then any next tool is not a good tool
            # because there are no need to deliver another tool
            good_tool = False

        # check if human is not working
        human_waiting = self.human.is_waiting()

        if at_work_room and carrying_tool and good_tool and human_waiting:
            # mark the bot to be free
            self.raw_occupancy[bot.pos[0], bot.pos[1]] = 2

            # mark the allocated tool
            assert 1 <= current_tool <= 4
            self.raw_occupancy[0, int(5 + current_tool)] = 1

            # increase the task stage
            self.task_stage += 1

            assert self.task_stage <= 4

            # tell human to restart working
            self.human.start_working()


    def _get_tool(self, act):
        assert Get_tool_1 <= act <= Get_tool_4
        tool_idx = act - 2
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
        # location: 0 (tool-room) or 1 (work-room) (2)
        # current-tool carried by the turtlebot: 0, 1, 2, 3, 4 (5) 0 means carrying nothing
        # task-stage: 0, 1, 2, 3, 4 (5)
        # human step count: 0, 1, 2 (3)
        # human status: 0, 1 (2)
        # human-desired-tool: 0, 1, 2, 3, 4 (5) 0: means tool#1

        self.raw_occupancy = np.zeros((4, 10))

        if state is not None:
            assert len(state) == 6, print(state)
        if state is not None:
            (bot_loc_id, bot_tool) = state[:2]
            (task_stage, human_step_cnt, human_status, human_desired_tool) = state[2:6]
            assert 0 <= task_stage <= 4
            assert 0 <= human_step_cnt <= 2
            assert 0 <= human_status <= 1
            assert 0 <= human_desired_tool <= 3

            if task_stage == 4:
                if human_desired_tool != self.correct_tool_order[-1] - 1:
                    raise InvalidState
            else:
                if human_desired_tool != self.correct_tool_order[task_stage] - 1:
                    raise InvalidState

            if human_status == Human.WAITING_TOOL:
                # if human is waiting, step count should be zero
                if human_step_cnt != 0:
                    raise InvalidState
            else:
                # if not, step count should be non-zero
                if human_step_cnt == 0:
                    raise InvalidState

        else:
            (bot_loc_id, bot_tool) = (0, 0) # (at workroom, free)
            (task_stage, human_step_cnt, human_status, human_desired_tool) = (0, 0, 0, 0)

        # first blacken all tool positions on the left (all tools are available)
        for i in range(4):
            self.raw_occupancy[0, i] = 1

        # if a tool is carrying by turtlebot, clear the corresponding tool positions
        assert bot_tool <= 4
        if bot_tool > 0:
            self.raw_occupancy[0, bot_tool - 1] = 0

        # clear tool positions depending on the current task stage
        # i.e., if the current task stage is [3] then the tool index [1] [2] were delivered
        if task_stage >= 1:
            for i in range(task_stage):
                self.raw_occupancy[0, self.correct_tool_order[i] - 1] = 0

        # color tool delivered position
        assert 0 <= task_stage <= 4
        if task_stage >= 1:
            for i in range(6, 6 + task_stage):
                self.raw_occupancy[0, i] = 1

        # human position
        assert human_status in [Human.WAITING_TOOL, Human.WORKING]
        self.raw_occupancy[3, 4] = 7 + human_status  # 7: waiting, 8: working - for rendering

        self.bot = TurtleBot()
        self.bot.set_room(bot_loc_id)
        self.raw_occupancy[self.bot.pos[0], self.bot.pos[1]] = 2 + bot_tool

        self.human = Human()
        self.human.set(human_status, human_step_cnt)

        self.task_stage = task_stage

    def get_state(self):
        """
        Request environmental state

        Return
        ------
        numpy.array
        """

        # location: 0 (tool-room) or 1 (work-room) (2)
        # current-tool carried by the turtlebot: 0, 1, 2, 3, 4 (5)
        # task-stage: 0, 1, 2, 3, 4 (5)
        # human step count: 0, 1, 2 (3)
        # human status: 0, 1 (2)
        # human-desired-tool: 0, 1, 2, 3 (4) 0: means tool#1

        # the desired tool does not change when the final task stage is reached
        if self.task_stage == 4:
            desired_tool = self.correct_tool_order[-1] - 1
        else:
            desired_tool = self.correct_tool_order[self.task_stage] - 1

        bot_state = [self.bot.get_location(), self.raw_occupancy[self.bot.pos[0], self.bot.pos[1]] - 2]
        other_state = [self.task_stage, self.human.time_step,
                       self.human.current_status, desired_tool]

        return np.array(bot_state + other_state, dtype=int)

    def is_terminal_state(self, state):
        # location: 0 (tool-room) or 1 (work-room) (2)
        # current-tool carried by the turtlebot: 0, 1, 2, 3, 4 (5)
        # task-stage: 0, 1, 2, 3, 4 (5)
        # human step count: 0, 1, 2 (3)
        # human status: 0, 1 (2)
        # human-desired-tool: 0, 1, 2, 3, 4 (5) 0: means tool#1

        assert len(state) == 6

        task_stage = state[2]
        human_status = state[4]
        current_desired_tool = state[5]

        terminal = False

        if current_desired_tool + 1 == self.correct_tool_order[-1] \
            and human_status == Human.WORKING \
            and task_stage == len(self.correct_tool_order):
            terminal = True

        return terminal

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

                # Human is waiting for a tool
                if self.raw_occupancy[i, j] == 7:
                    cv2.rectangle(obs, (j*60, i*60), (j*60+60, i*60+60), (255, 255, 0), 2)
                    cv2.putText(obs, f'S{self.human.time_step}', (j*60 + 15, i*60 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

                # Human is working
                if self.raw_occupancy[i, j] == 8:
                    cv2.rectangle(obs, (j*60, i*60), (j*60+60, i*60+60), (255, 255, 0), 2)
                    cv2.putText(obs, f'W{self.human.time_step}', (j*60 + 15, i*60 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

        cv2.imshow('image', obs)
        cv2.waitKey(10)

    def _clear_bot_box(self):
        bot = self.bot
        saved_class = self.raw_occupancy[bot.pos[0], bot.pos[1]]
        self.raw_occupancy[bot.pos[0], bot.pos[1]] = 0
        return saved_class

if __name__ == "__main__":

    env = EnvWareHouse(correct_tool_order=[4, 3, 2, 1])
    env.reset()
    env.render()
    time.sleep(2)

    action_list = [Go_to_TR, Get_tool_4, Go_to_WR, Deliver_tool, Go_to_TR, Get_tool_3, Go_to_WR, Deliver_tool]
    action_list += [Go_to_TR, Get_tool_2, Go_to_WR, Deliver_tool, Go_to_TR, Get_tool_1, Go_to_WR, Deliver_tool, No_Op]

    for action in action_list:
        env.step([action])
        env.render()
        time.sleep(1)

    # assert env.is_done() is True
