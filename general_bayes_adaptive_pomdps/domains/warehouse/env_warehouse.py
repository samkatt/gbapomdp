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
    def __init__(self, id, time_cost=3):
        self.id = id

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

    def __init__(self, id):
        self.id = id

        # initialize at work room
        if self.id == 1:
            self.pos = [2, 6]
        elif self.id == 2:
            self.pos = [2, 9]

    def set_room(self, id, room_id):
        # tool room
        if room_id == 0:
            if self.id == 1:
                self.pos = [2, 0]
            elif self.id == 2:
                self.pos = [2, 3]
            else:
                raise NotImplementedError
        # work room
        else:
            if self.id == 1:
                self.pos = [2, 6]
            elif self.id == 2:
                self.pos = [2, 9]
            else:
                raise NotImplementedError

    def move(self, action):
        if action == Go_to_WR:
            self.go_to_wr()

        if action == Go_to_TR:
            self.go_to_tr()

    def go_to_tr(self):
        if self.id == 1:
            self.pos = [2, 0]
        elif self.id == 2:
            self.pos = [2, 3]
        else:
            raise NotImplementedError

    def go_to_wr(self):
        if self.id == 1:
            self.pos = [2, 6]
        elif self.id == 2:
            self.pos = [2, 9]
        else:
            raise NotImplementedError

    def get_location(self):
        if self.pos[1] > self.ROOM_BOUNDARY:
            return self.AT_WORK_ROOM
        else:
            return self.AT_TOOL_ROOM

class EnvWareHouse(object):
    BOT1_ID = 1
    BOT2_ID = 2

    def __init__(self):
        self.reset()

    def step(self, action):
        if action[0] <= 1:
            saved_class = self._clear_bot_box(1)
            self.bot1.move(action[0])
            self.raw_occupancy[self.bot1.pos[0], self.bot1.pos[1]] = saved_class
        
        if action[1] <= 1:
            saved_class = self._clear_bot_box(2)
            self.bot2.move(action[1])
            self.raw_occupancy[self.bot2.pos[0], self.bot2.pos[1]] = saved_class

        if Get_tool_1 <= action[0] <= Get_tool_4 and Get_tool_1 <= action[1] <= Get_tool_4:
            # If both turtlebot want to get the same tool, only one will get the tool
            if action[0] == action[1]:
                if random.choice([True, False]):
                    self._get_tool(action[0], self.BOT1_ID)
                else:
                    self._get_tool(action[1], self.BOT2_ID)
            else:
                self._get_tool(action[0], self.BOT1_ID)
                self._get_tool(action[1], self.BOT2_ID)
        else:
            if Get_tool_1 <= action[0] <= Get_tool_4:
                self._get_tool(action[0], self.BOT1_ID)

            if Get_tool_1 <= action[1] <= Get_tool_4:
                self._get_tool(action[1], self.BOT2_ID)

        if action[0] == Deliver_tool and action[1] == Deliver_tool:
            # If both want to drop a tool, only one can do that
            if random.choice([True, False]):
                self._drop_tool(self.BOT1_ID)
            else:
                self._drop_tool(self.BOT2_ID)
        else:
            if action[0] == Deliver_tool:
                self._drop_tool(self.BOT1_ID)

            if action[1] == Deliver_tool:
                self._drop_tool(self.BOT2_ID)

        self.human.update()

        if self.human.is_waiting():
            self.raw_occupancy[3, 4] = 8
        else:
            self.raw_occupancy[3, 4] = 7


    def _drop_tool(self, bot_id):
        assert bot_id in [self.BOT1_ID, self.BOT2_ID]

        if bot_id == self.BOT1_ID:
            bot = self.bot1
        else:
            bot = self.bot2

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


    def _get_tool(self, action, bot_id):
        assert bot_id in [self.BOT1_ID, self.BOT2_ID]
        assert Get_tool_1 <= action <= Get_tool_4
        tool_idx = action - 2

        if bot_id == self.BOT1_ID:
            bot = self.bot1
        else:
            bot = self.bot2

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
        # state: (location, current-tool) per turtlebot + (human-status, desired-tool)
        # location: tool-room or work-room (2)
        # current-tool: 0, 1, 2, 3, 4 (5)
        # human-status: working or waiting (2)
        # desired-tool: 1, 2, 3, 4, 5 (5)
        # human time step: 0, 1, 2 (3)
        self.raw_occupancy = np.zeros((4, 10))

        if state is not None:
            assert len(state) == 5
        if state is not None:
            (bot1_loc_id, bot1_tool) = state[:2]
            (bot2_loc_id, bot2_tool) = state[2:4]
            (human_status, desired_tool) = state[4:6]
            human_step_cnt = state[6]
        else:
            (bot1_loc_id, bot1_tool) = (0, 0)  # (at toolroom, free)
            (bot2_loc_id, bot2_tool) = (0, 0)  # (at toolroom, free)
            (human_status, desired_tool) = (0, 0)  # (working, 0)
            human_step_cnt = 0

        # first blacken all tool positions
        for i in range(4):
            self.raw_occupancy[0, i] = 1

        # a tool is carrying by turtlebot 1, whiten the corresponding tool positions
        assert bot1_tool <= 4
        if bot1_tool > 0:
            self.raw_occupancy[0, bot1_tool - 1] = 0

        # a tool is carrying by turtlebot 2, whiten the corresponding tool positions
        assert bot2_tool <= 4
        if bot2_tool > 0:
            self.raw_occupancy[0, bot2_tool - 1] = 0

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

        self.bot1 = TurtleBot(id=self.BOT1_ID)
        self.bot2 = TurtleBot(id=self.BOT2_ID)

        self.bot1.set_room(self.BOT1_ID, bot1_loc_id)
        self.bot2.set_room(self.BOT2_ID, bot2_loc_id)

        self.human = Human(id=1)
        self.human.set(human_status, human_step_cnt)

        self.raw_occupancy[self.bot1.pos[0], self.bot1.pos[1]] = 2 + bot1_tool
        self.raw_occupancy[self.bot2.pos[0], self.bot2.pos[1]] = 2 + bot2_tool

        self.desired_tool = desired_tool

        return self.get_obs()

    def get_state(self):
        """
        Request environmental state

        Return
        ------
        numpy.array
        """

        bot1_state = [self.bot1.get_location(), self.raw_occupancy[self.bot1.pos[0], self.bot1.pos[1]] - 2]
        bot2_state = [self.bot2.get_location(), self.raw_occupancy[self.bot2.pos[0], self.bot2.pos[1]] - 2]

        return np.array(bot1_state + bot2_state + [self.human.current_status, self.desired_tool, self.human.time_step], dtype=int)

    def get_obs(self):
        """
        Request environmental state

        Return
        ------
        numpy.array
        """

        bot1_state = [self.bot1.get_location(), self.raw_occupancy[self.bot1.pos[0], self.bot1.pos[1]]]
        bot2_state = [self.bot2.get_location(), self.raw_occupancy[self.bot2.pos[0], self.bot2.pos[1]]]

        return np.array(bot1_state + bot2_state + [self.human.current_status, self.desired_tool], dtype=int)

    def is_done(self):
        # Human must finish using the final tool
        return self.desired_tool == 4 and self.human.is_waiting()

    def is_terminal_state(self, state):
        # Human must finish using the final tool
        assert len(state) == 7
        desired_tool = state[5]
        human_status = state[4]
        return desired_tool == 4 and human_status == self.human.WAITING_TOOL

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
                    cv2.putText(obs, 'W', (j*60 + 25, i*60 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

                # Human is waiting for a tool
                if self.raw_occupancy[i, j] == 8:
                    cv2.rectangle(obs, (j*60, i*60), (j*60+60, i*60+60), (255, 255, 0), 2)
                    cv2.putText(obs, 'S', (j*60 + 25, i*60 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

        cv2.imshow('image', obs)
        cv2.waitKey(10)

    def _clear_bot_box(self, bot_id):
        if bot_id == 1:
            bot = self.bot1
        else:
            bot = self.bot2
        saved_class = self.raw_occupancy[bot.pos[0], bot.pos[1]]
        self.raw_occupancy[bot.pos[0], bot.pos[1]] = 0
        return saved_class

if __name__ == "__main__":

    sem_action_space = one_to_one.JointNamedSpace(
        a1=one_to_one.RangeSpace(8),  # 0: move forward, 1: turn left, 2: turn right, 3: stay
        a2=one_to_one.RangeSpace(8),
    )
    action_lst = list(sem_action_space.elems)
    index2str = ['Go_to_WR', 'Go_to_TR', 'Get_Tool_0', 'Get_Tool_1',
                'Get_Tool_2', 'Get_Tool_3', 'Deliver', 'No_Op']


    def action_to_str(action_idx):
        actions = action_lst[action_idx]
        print(actions.a1.value, actions.a2.value)
        return f"agent 0: {index2str[actions.a1.value]},\
        agent 1: {index2str[actions.a2.value]}"

    print(action_to_str(35))

    env = EnvWareHouse()
    env.reset()
    env.render()

    env.step([1, 1])
    env.render()
    time.sleep(1)

    env.step([2, 3])
    env.render()
    time.sleep(1)

    env.step([0, 0])
    env.render()
    time.sleep(1)

    env.step([6, 0])
    env.render()
    time.sleep(1)

    env.step([7, 7])
    env.render()
    time.sleep(1)

    env.step([7, 7])
    env.render()
    time.sleep(1)

    env.step([7, 6])
    env.render()
    time.sleep(1)

    env.step([1, 1])
    env.render()
    time.sleep(1)

    env.step([4, 5])
    env.render()
    time.sleep(1)

    env.step([0, 0])
    env.render()
    time.sleep(1)

    env.step([6, 0])
    env.render()
    time.sleep(1)

    env.step([7, 7])
    env.render()
    time.sleep(1)

    env.step([7, 7])
    env.render()
    time.sleep(1)

    env.step([7, 6])
    env.render()
    time.sleep(1)

    env.step([7, 7])
    env.render()
    time.sleep(1)

    env.step([7, 7])
    env.render()
    time.sleep(1)

    assert env.is_done() is True

    # Reset at a certain state
        # state: (location, current-tool) per turtlebot + (human-status, desired-tool)
        # location: tool-room or work-room (2)
        # current-tool: 0, 1, 2, 3, 4 (5)
        # human-status: working or waiting (2)
        # desired-tool: 1, 2, 3, 4, 5 (5)
        # human time step: 0, 1, 2 (3)
    # env.reset(state=[0, 0, 0, 0, 0, 0, 0])
    # env.render()
    # time.sleep(10)

    # env.reset(state=[1, 0, 1, 0, 0, 0, 0])
    # env.render()
    # time.sleep(10)

    # env.reset(state=[1, 0, 0, 2, 1, 0, 0])
    # env.render()
    # time.sleep(2)

    # env.reset(state=[1, 2, 1, 1, 1, 0, 0])
    # env.render()
    # time.sleep(2)

    # env.step([4, 3])
    # env.render()
    # time.sleep(2)