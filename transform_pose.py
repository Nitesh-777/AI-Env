import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import matplotlib.pyplot as plt


def get_pose(pose1, pose2):
    def transform(x, y, angle):
        rotation_matrix_2d = pr.active_matrix_from_angle(2, angle)
        transformation_matrix = np.eye(4)
        transformation_matrix[0:3, 0:3] = rotation_matrix_2d
        transformation_matrix[0, 3] = x
        transformation_matrix[1, 3] = y
        # translation_matrix = pt.translate_transform(np.eye(4), [x, y, 0])
        # print(translation_matrix)
        # # rotation_matrix = pt.rotate_transform(np.eye(4), angle, [0, 0, 1])
        # rotation_matrix = pr.matrix_from_axis_angle([0, 0, 1, angle])
        # print(rotation_matrix)
        # print(pr.check_matrix(rotation_matrix))
        # transformation_matrix = np.dot(translation_matrix, rotation_matrix)
        return transformation_matrix
    pose1_transform = transform(*agent_pose)
    print(f"pose1 transformation matrix \n{pose1_transform}")
    pose2_transform = transform(*helper_pose)
    print(f"pose2 transformation matrix \n{pose2_transform}")
    pose2_inverse = pt.invert_transform(pose2_transform)
    print(f"pose2 inverse matrix \n{pose2_inverse}")
    relative_matrix = np.dot(pose2_inverse, pose1_transform)
    print(f"relative pose matrix \n{relative_matrix}")
    final_pose = [relative_matrix[0, 3], relative_matrix[1, 3],
                  np.arctan2(relative_matrix[1, 0], relative_matrix[0, 0])]
    print(f"final pose \n{final_pose}")
    return final_pose


def visualise(ax, pose, color, label):
    x, y, angle = pose
    dx = np.cos(angle)
    dy = np.sin(angle)
    ax.plot(x, y, 'o', color=color)
    ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc=color, ec=color)
    if label:
        ax.text(x + 0.1, y + 0.1, label, color=color)

agent_pose = [1, 1, np.pi / 8]  # Pose X
helper_pose = [2, 1, np.pi / 4]  # Pose Y
agent_pose_helper = get_pose(agent_pose, helper_pose)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.set_title('Agent Perspective')
ax1.set_aspect('equal')
ax1.set_xlim(0, 5)
ax1.set_ylim(0, 5)
visualise(ax1, agent_pose, 'blue', 'Pose X')
visualise(ax1, helper_pose, 'green', 'Pose Y')
ax2.set_title("Helper Perspective")
ax2.set_aspect('equal')
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
visualise(ax2, [0, 0, 0], 'green', 'Pose Y (Origin)')
visualise(ax2, agent_pose_helper, 'blue', 'Transformed Pose X')
plt.show()
