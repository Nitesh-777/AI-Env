import numpy as np
from movement_avoidance import attract, repulse, resultant, angle_to_wheel
from transform_pose import dist_qpos

class EnvAgent(object):
    def __init__(self, agent_name):
        self.agent_name = agent_name

    def move_to_goal(self, data, goal_qpos, max_dist, intensity=1.0):
        obstacles_qpos = self.get_obstacles_qpos(data)
        self_qpos = self.get_self_qpos(data)
        attract_angle = attract(self_qpos, goal_qpos)
        comb_repulse_vector = np.array([0.0, 0.0])

        for obstacle_qpos in obstacles_qpos:
            repulse_angle, repulse_magnitude = repulse(self_qpos, obstacle_qpos, max_dist, intensity)
            repulse_vector = np.array(
                [np.cos(repulse_angle) * repulse_magnitude, np.sin(repulse_angle) * repulse_magnitude])
            comb_repulse_vector += repulse_vector

        if np.linalg.norm(comb_repulse_vector) > 0:
            comb_repulse_angle = np.arctan2(comb_repulse_vector[1], comb_repulse_vector[0])
        else:
            comb_repulse_angle = attract_angle

        resultant_angle = resultant(attract_angle, comb_repulse_angle, np.linalg.norm(comb_repulse_vector))
        wheel_actions = angle_to_wheel(resultant_angle, threshold=np.radians(30))

        return wheel_actions

    def get_self_qpos(self, data):
        return data.joint(self.agent_name).qpos

    def get_learner_qpos(self, data):
        return data.joint('root').qpos

    def get_obstacles_qpos(self, data):
        obstacles_qpos = []
        for i in range(1, 11):
            food = 'food_free_' + str(i)
            obstacles_qpos.append(data.joint(food).qpos)
        return obstacles_qpos


