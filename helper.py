from env_agent import EnvAgent
import numpy as np
from transform_pose import dist_qpos

class Helper(EnvAgent):
    def __init__(self, food_name):
        super().__init__("Hroot")
        self.food_name = food_name
        self.helper_state = "helping"
        self.current_wait_steps = 0
        self.food_given = False

    def get_action(self, data, wait_pos, max_wait_steps, goal_dist, food_range):

        self.food_eaten(data)

        learner_qpos = self.get_learner_qpos(data)
        # Changed goal_qpos dist to 0.5 for food_eaten test, originally 1
        goal_qpos = dist_qpos(learner_qpos, goal_dist)
        print(f"Current State: {self.helper_state}")
        print(f"{self.current_wait_steps} / {max_wait_steps} Wait Steps")

        if self.helper_state == "helping":
            placed = self.place_food(data, goal_qpos, food_range)
            if placed:
                print("placed")
                self.helper_state = "waiting"
                self.food_given = True
                return np.array([0, 0])
            else:
                print("not placed")
                self.carry_food(data)
                return self.move_to_goal(data, goal_qpos, 2.0)
        elif self.helper_state == "waiting":
            if self.current_wait_steps >= max_wait_steps:
                self.helper_state = "helping"
                self.current_wait_steps = 0
                self.food_given = False
                return np.array([0, 0])
            else:
                self.current_wait_steps += 1
                return self.wait(data, wait_pos)

    def carry_food(self, data):
        helper_qpos = self.get_self_qpos(data)
        food_helper_qpos = dist_qpos(helper_qpos, 0.8)
        food_helper_qpos[2] = 0.75
        self.set_food_qpos(data, self.food_name, food_helper_qpos)

    def in_range(self, current_qpos, target_pos, range):
        lower_left = [target_pos[0] - range, target_pos[1] - range]
        upper_right = [target_pos[0] + range, target_pos[1] + range]
        in_range = lower_left[0] <= current_qpos[0] <= upper_right[0] and lower_left[1] <= current_qpos[1] <= upper_right[1]
        return in_range

    def place_food(self, data, goal_qpos, food_range):
        food_qpos = self.get_food_qpos(data)
        # place food changed for food_eaten test to 0, originally 0.25
        in_range = self.in_range(food_qpos, goal_qpos, food_range)
        return in_range

    def wait(self, data, wait_pos):
        helper_qpos = self.get_self_qpos(data)
        wait_qpos = np.copy(helper_qpos)
        wait_qpos[:2] = wait_pos
        action = self.move_to_goal(data, wait_qpos, 1)
        if self.in_range(helper_qpos, wait_pos, 0.25):
            return np.array([0, 0])
        else:
            return action

    def food_eaten(self, data):
        for contact in data.contact:
            geom1_name = data.geom(contact.geom1).name
            geom2_name = data.geom(contact.geom2).name
            if ('beak' in geom1_name + geom2_name) and ("food_geom_11" in geom1_name + geom2_name):
                self.food_given = True
                break
        if self.food_given:
            self.helper_state = "waiting"

    def get_food_qpos(self, data):
        return data.joint(self.food_name).qpos

    def set_food_qpos(self, data, food_name, qpos):
        data.joint(food_name).qpos = qpos

    def get_obstacles_qpos(self, data):
        obstacles_qpos = super().get_obstacles_qpos(data)
        if self.food_given:
            obstacles_qpos.append(data.joint("food_free_11").qpos)

        if self.helper_state == "waiting":
            learner_qpos = self.get_learner_qpos(data)
            obstacles_qpos.append(learner_qpos)

        return obstacles_qpos


    def reset(self):
        self.helper_state = "helping"
        self.current_wait_steps = 0
        self.food_given = False
