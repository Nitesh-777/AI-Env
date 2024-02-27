import numpy as np
# from transform_pose import get_pose

class Helper(object):
    def __init__(self):
        pass
    def get_action(self, data):
        agent_quat = data.body("torso").xquat
        agent_matrix = data.body("torso").xmat
        helper_quat = data.body("Htorso").xquat
        helper_matrix = data.body("Htorso").xmat



        # agent_qpos = data.joint("Hroot").qpos[:]
        # helper_qpos = data.body("Htorso").qpos[:]

        # x3 = data.body("torso").xaxis

        # q = data.qpos
        # print("AGENT -------")
        # print(data.body("torso").xpos)
        # print(f"x quaternion of learner: {agent_quat}")
        # print(f"x transformation matrix of agent: {agent_matrix}")
        # print(f"QPOS:  {agent_qpos}")
        # print("HELPER ~~~~~~~")
        # print(data.body("Htorso").xpos)

        # print(f"x quaternion of helper: {helper_quat}")
        # print(f"x transformation matrix of agent: {helper_matrix}")
        # print(f"QPOS:  {helper_qpos}")

        # print(f"x axis vector of agent: {x3}")
        # print(f"q position of agent: {q}")
        print("body(torso) -------------------------")
        print(f"xpos: {data.body('torso').xpos}")
        print(f"xipos: {data.body('torso').xipos}")
        print(f"xmat: {data.body('torso').xmat}")
        print(f"ximat: {data.body('torso').ximat}")
        print(f"xquat: {data.body('torso').xquat}")
        print("joint(root) #########################")
        print(f"qpos: {data.joint('root').qpos}")
        print(f"qvel: {data.joint('root').qvel}")
        print(f"xanchor: {data.joint('root').xanchor}")
        print(f"xaxis: {data.joint('root').xaxis}")



        print(dir(data.body('torso')))
        print(dir(data.joint('root')))
        # print(dir(data.joint('root')))
        return np.array([1, 1])
