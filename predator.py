import numpy as np

from env_agent import EnvAgent


class Predator(EnvAgent):
    def __init__(self):
        super().__init__("Proot")
        self.predator_state = "attacking"
        self.attacked = False


    def get_action(self, data):
        self.learner_attacked(data)
        if self.predator_state == "attacking":
            target_qpos = self.get_learner_qpos(data)
            return self.move_to_goal(data, target_qpos, 1)
        return np.array([0, 0])

    def learner_attacked(self, data):
        for contact in data.contact:
            geom1_name = data.geom(contact.geom1).name
            geom2_name = data.geom(contact.geom2).name
            if ('Pbeak' in geom1_name + geom2_name) and ("torso_geom" in geom1_name + geom2_name):
                self.attacked = True
                break

    def get_obstacles_qpos(self, data):
        obstacles_qpos = super().get_obstacles_qpos(data)
        helper_qpos = data.joint('Hroot').qpos
        obstacles_qpos.append(helper_qpos)
        return obstacles_qpos

    def reset(self):
        self.predator_state = "attacking"
        self.attacked = False
