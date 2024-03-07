import numpy as np
import pytransform3d.rotations as pr
from movement_avoidance import attract, repulse, resultant, angle_to_wheel
from transform_pose import get_pose_qpos,new_pose

class Helper(object):
    def __init__(self):
        pass
    def get_action(self, data):
        obstacles_qpos = []
        for i in range(1 , 11):
            food = 'food_free_' + str(i)
            obstacles_qpos.append(data.joint(food).qpos)
        # print("OBSTACLESSSSSSSSSSSSS -----------------")
        # print(obstacles_qpos)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        helper_qpos = data.joint('Hroot').qpos
        learner_qpos = data.joint('root').qpos

        # goal_qpos = self.food_placement_qpos(learner_qpos, helper_qpos, 1)

        action = self.move_to_goal(helper_qpos, learner_qpos, obstacles_qpos, 1)

        return action

    def food_placement_qpos(self, learner_qpos, helper_qpos, dist):
        pose = get_pose_qpos(learner_qpos, helper_qpos)
        goal_pose = new_pose(pose, dist)
        goal_qpos = learner_qpos
        goal_qpos[:2] = goal_pose[:2]
        goal_quat = pr.quaternion_from_angle(2, goal_pose[2])
        goal_qpos[3:] = goal_quat

        return goal_qpos
    def move_to_goal(self, helper_qpos, goal_qpos, obstacles_qpos, max_dist, intensity=1.0):
        attract_angle = attract(helper_qpos, goal_qpos)

        comb_repulse_vector = np.array([0.0, 0.0])

        for obstacle_qpos in obstacles_qpos:
            repulse_angle, repulse_magnitude = repulse(helper_qpos, obstacle_qpos, max_dist, intensity)
            repulse_vector = np.array(
                [np.cos(repulse_angle) * repulse_magnitude, np.sin(repulse_angle) * repulse_magnitude])
            comb_repulse_vector += repulse_vector

        if np.linalg.norm(comb_repulse_vector) > 0:
            comb_repulse_angle = np.arctan2(comb_repulse_vector[1], comb_repulse_vector[0])
        else:
            comb_repulse_angle = attract_angle

        resultant_angle = resultant(attract_angle, comb_repulse_angle,
                                                  np.linalg.norm(comb_repulse_vector))

        wheel_actions = angle_to_wheel(resultant_angle)

        return wheel_actions


