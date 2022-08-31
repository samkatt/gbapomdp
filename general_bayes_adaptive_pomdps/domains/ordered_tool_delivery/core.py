#!/usr/bin/python
import numpy as np

class AgentTurtlebot_v4(object):

    """Properties for a Turtlebot"""

    def __init__(self,
                 idx,
                 init_x,
                 init_y,
                 beliefwaypoints,
                 MAs,
                 n_objs,
                 fetch,
                 speed=0.6):

        # unique agent's id
        self.idx = idx
        # agent's name
        self.name = 'Turtlebot'+ str(self.idx)
        # agent's 2D position x
        self.xcoord = init_x
        # agent's 2D position y
        self.ycoord = init_y
        # applicable waypoints to move to
        self.BWPs = beliefwaypoints
        # record which belief waypoint the agent currently is
        self.cur_BWP = None
        # obtain applicable macro_actions
        self.macro_actions = MAs
        # agent's current macro_action
        self.cur_action = None
        # how much time left to finish current macro_action
        self.cur_action_time_left = 0.0
        self.cur_action_done = False
        # turtlebot base movement speed
        self.speed = speed

        # reached receipt spot
        self.reached_receipt_spot = False

        # communication info
        self.n_objs = n_objs
        # fetch robot
        self.fetch = fetch
        # keep tracking the objects in the basket
        self.objs_in_basket = np.zeros(n_objs)

    def step(self, action, humans):

        """Depends on the input macro-action to run low-level controller to achieve
           primitive action execution.
        """

        assert action < len(self.macro_actions), "The action received is out of the range"

        # update current action info
        self.cur_action = self.macro_actions[action]
        self.cur_action_done = False
        self.reached_receipt_spot = False

        # move to the corresponding waypoints
        bwpterm_idx = self.cur_action.ma_bwpterm
        if self.cur_action.expected_t_cost != 1:
            dist = round(self._get_dist(self.BWPs[bwpterm_idx]), 2)
            if dist <= self.speed:
                self.xcoord = self.BWPs[bwpterm_idx].xcoord
                self.ycoord = self.BWPs[bwpterm_idx].ycoord
                self.cur_BWP = self.BWPs[bwpterm_idx]
                if self.cur_action_time_left > 0.0:
                    self.cur_action_time_left = 0.0

                # Deliver_Tool action
                if action == self.n_objs:
                    self.cur_action_done = True
                # Get_Tool_i action
                else:
                    self.reached_receipt_spot = True
            else:
                delta_x = self.speed / dist * (self.BWPs[bwpterm_idx].xcoord - self.xcoord)
                delta_y = self.speed / dist * (self.BWPs[bwpterm_idx].ycoord - self.ycoord)
                self.xcoord += delta_x
                self.ycoord += delta_y
                self.cur_action_time_left = dist - self.speed
        else:
            self.xcoord = self.BWPs[bwpterm_idx].xcoord
            self.ycoord = self.BWPs[bwpterm_idx].ycoord
            self.cur_BWP = self.BWPs[bwpterm_idx]

            # Deliver_Tool action
            if action == self.n_objs:
                self.cur_action_done = True
            # Get_Tool_i action
            else:
                self.reached_receipt_spot = True

        # change the human's properties when turtlebot delivers correct objects
        if self.cur_BWP is not None and \
            self.cur_action_done and \
            (action == self.n_objs):

            assert len(humans) == 1
            human = humans[0]

            if not human.next_requested_obj_obtained and \
                human.cur_step_time_left <= 1 and \
                self.objs_in_basket[human.next_request_obj_idx] > 0.0:
                self.objs_in_basket[human.next_request_obj_idx] -= 1.0
                human.next_requested_obj_obtained = True

        # ask Fetch to hand over the desired tool when at the receipt spot
        if action < self.n_objs and self.reached_receipt_spot:
            # action will be used to decide which tool to find
            self.fetch.step(action)

            # Fetch done then the whole Get_Tool action is also done
            if self.fetch.cur_action_done:
                if self.fetch.tool_found:
                    obj_idx = action
                    assert self.objs_in_basket[obj_idx] == 0.0
                    self.objs_in_basket[obj_idx] += 1.0

                self.cur_action_done = True

        return

    def _get_dist(self, goal):
        """Compute the distance from the turtlebot to a goal waypoint"""
        return np.sqrt((goal.xcoord - self.xcoord)**2 + (goal.ycoord - self.ycoord)**2)

class AgentFetch_v4(object):
    """Properties for a Fetch robot
    """

    def __init__(self,
                 idx,
                 init_x,
                 init_y,
                 MAs,
                 n_objs,
                 n_each_obj):

        # unique agent's id
        self.idx = idx
        # agent's name
        self.name = 'Fetch'
        # agent's 2D position x
        self.xcoord = init_x
        # agent's 2D position y
        self.ycoord = init_y
        # obtain applicable macro_actions
        self.macro_actions = MAs
        # agent's current macro_action
        self.cur_action = None
        # how much time left to finish current macro_action
        self.cur_action_time_left = 0.0
        self.cur_action_done = False
        # the number of different objects in this env
        self.n_objs = n_objs
        # the number of each obj in the env
        self.n_each_obj = n_each_obj
        self.count_found_obj = np.zeros(n_objs)

        ################# communication info ######################
        self.tool_found = False

        # list of tools that are passed to the turtlebot
        self.passed_tools = []

    def step(self, action):

        """Depends on the input macro-action to run low-level controller to
           achieve primitive action execution.
        """

        reward = 0.0

        self.cur_action = self.macro_actions[action]

        if self.cur_action_time_left == 0.0:
            self.cur_action_time_left = self.cur_action.t_cost
            self.cur_action_done = False

        self.cur_action_time_left -= 1.0

        if self.cur_action_time_left > 0.0:
            return reward
        else:
            found_obj_idx = self.cur_action.idx
            self.tool_found = False
            if self.count_found_obj[found_obj_idx] < self.n_each_obj:
                self.count_found_obj[found_obj_idx] += 1.0
                self.tool_found = True
                self.passed_tools.append(found_obj_idx)

            self.cur_action_done = True

        return reward

class AgentHuman(object):

    """Properties for a Human in the env"""

    def __init__(self,
                 idx,
                 task_total_steps,
                 expected_timecost_per_task_step,
                 request_objs_per_task_step):

        # unique agent's id
        self.idx = idx
        # the total number of steps for finishing the task
        self.task_total_steps = task_total_steps
        # a vector to indicate the expected time cost for each human to finish each task step
        self.expected_timecost_per_task_step = expected_timecost_per_task_step
        # a vector to inidcate the tools needed for each task step
        self.request_objs_per_task_step = request_objs_per_task_step

        self.cur_step = 0
        self.cur_step_time_left = self.expected_timecost_per_task_step[self.cur_step]

        # indicates the tool needed for next task step
        self.next_request_obj_idx = self.request_objs_per_task_step[self.cur_step]
        # indicates if the tool needed for next step has been delivered
        self.next_requested_obj_obtained = False
        # indicates if the human has finished the whole task
        self.whole_task_finished = False
        self.accept_tool = False

    def step(self):

        # check if the human already finished whole task
        if self.cur_step + 1 == self.task_total_steps:
            assert self.whole_task_finished is False
            self.whole_task_finished = True
        else:
            self.cur_step += 1
            self.cur_step_time_left = self.expected_timecost_per_task_step[self.cur_step]
            # update the request obj for next step
            if self.cur_step + 1 < self.task_total_steps:
                self.next_request_obj_idx = self.request_objs_per_task_step[self.cur_step] 
                self.next_requested_obj_obtained = False

    def reset(self):
        self.cur_step = 0
        self.cur_step_time_left = self.expected_timecost_per_task_step[self.cur_step]

        # indicates the tool needed for next task step
        self.next_request_obj_idx = self.request_objs_per_task_step[self.cur_step]  
        # indicates if the tool needed for next step has been delivered
        self.next_requested_obj_obtained = False
        # indicates if the human has finished the whole task
        self.whole_task_finished = False
        self.accept_tool = False

class MacroAction(object):

    """Properties for a macro_action"""

    def __init__(self, 
                 name,
                 idx,
                 expected_t_cost=None,
                 std=None,
                 ma_bwpterm=None):

        # the name of this macro-action
        self.name = name
        # the index of this macro-action
        self.idx = idx    
        # None is for moving action. When it is done depends on the specify speed.
        self.expected_t_cost = expected_t_cost
        self.std = std
        if std is None:
            # the time cost of finishing this macro-action
            self.real_t_cost = expected_t_cost
        else:
            self.real_t_cost = np.random.normal(expected_t_cost, std)
        # used for moving action to indicate at which belief waypoint this macro-action
        # will be terminated,
        # None means the terminate belief waypoint is same as where the action is initialized.
        self.ma_bwpterm = ma_bwpterm

    @property
    def t_cost(self):
        if self.std is None:
            # the time cost of finishing this macro-action
            return self.expected_t_cost
        else:
            # resample a time cost for the macro-action
            return round(np.random.normal(self.expected_t_cost, self.std), 1)

class BeliefWayPoint(object):

    """Properties for a waypoint in the 2D sapce"""

    def __init__(self,
                 name,
                 idx,
                 xcoord,
                 ycoord):

        self.name = name
        self.idx = idx
        self.xcoord = xcoord
        self.ycoord = ycoord
