import sys
import numpy as np
import matplotlib.pyplot as plt

import utils
from questions import q1_a, q1_c, q2, q3, q4_a, q4_c


def hw4(question):
    if question == 'q1_a':
        T = utils.make_tfm_matrix((-0.1, 0.5, 1), (0.3, 0.4, 0.5))
        P = utils.make_plane3d(10, tfm=T)
        P += 0.02 * np.random.randn(P.shape[0], 3)

        # plot the plane
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(*P.T)

        normal, center = q1_a(P)
        utils.plot_plane(ax, normal, center)
        plt.show()

    elif question == 'q1_b':
        T = utils.make_tfm_matrix((-0.1, 0.5, 1), (0.3, 0.4, 0.5))
        P = utils.make_plane3d(10, tfm=T)
        P = P + 0.02 * np.random.randn(P.shape[0], 3)

        # add outlier noise
        P[np.arange(8)] += np.random.randn(8, 3)

        # plot the plane
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(*P.T)

        normal, center = q1_a(P)
        utils.plot_plane(ax, normal, center)
        plt.show()

    elif question == 'q1_c':
        T = utils.make_tfm_matrix((-0.1, 0.5, 1), (0.3, 0.4, 0.5))
        P = utils.make_plane3d(10, tfm=T)
        P = P + 0.02 * np.random.randn(P.shape[0], 3)

        # add outlier noise
        P[np.arange(8)] += np.random.randn(8, 3)

        # plot the plane
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(*P.T)

        normal, center = q1_c(P)
        utils.plot_plane(ax, normal, center)
        plt.show()

    elif question == 'q2':
        scene_data = np.load('object3d.npy')
        pc, normals, rgb = np.array_split(scene_data, 3, axis=1)

        # segment region of interest (for testing ONLY)
        # roi = np.array(((-np.inf, 0.5), (0.2, 0.4), (0.1, np.inf)))
        roi = np.array(((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)))
        mask = utils.within_roi(pc, roi)

        center, radius = q2(pc[mask], normals[mask])

        # plot point cloud and fitted sphere
        ax = plt.figure().add_subplot(projection='3d')
        ax.view_init(30, -115)
        utils.plot_sphere(ax, center, radius)
        ax.scatter(*pc[::4].T, c=rgb[::4], s=0.04)
        plt.tight_layout()
        plt.show()

    elif question == 'q3':
        scene_data = np.load('object3d.npy')
        pc, normals, rgb = np.array_split(scene_data, 3, axis=1)

        # segment region of interest
        roi = np.array(((0.4, 0.6), (-np.inf, 0.2), (0.1, np.inf)))
        mask = utils.within_roi(pc, roi)

        center, axis, radius = q3(pc[mask], normals[mask])
        axis = axis / np.linalg.norm(axis)

        # plot point cloud and fitted cylinder
        ax = plt.figure().add_subplot(projection='3d')
        ax.view_init(30, -115)
        utils.plot_cylinder(ax, center, axis, radius)
        ax.scatter(*pc[::4].T, c=rgb[::4], s=0.04)
        plt.tight_layout()
        plt.show()

    elif question == 'q4_a':
        pc = np.load('bunny.npy')
        tfm = utils.make_tfm_matrix(tvec=(-0.1, 0.0, 0.3), rpy=(0.05, -0.1, 0.2))
        pc_transformed = utils.apply_transform(pc, tfm)

        T = q4_a(pc_transformed, pc)

        pc_untransformed = utils.apply_transform(pc_transformed, T)

        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(*pc.T, c='r', alpha=0.3)
        ax.scatter(*pc_untransformed.T, marker='^', c='g', alpha=0.3)
        plt.show()

    elif question == 'q4_b':
        np.random.seed(0)
        pc = np.load('bunny.npy')
        pc_noisy = pc + 0.001*np.random.randn(*pc.shape)
        ids = np.arange(len(pc))
        np.random.shuffle(ids)
        pc_shuffled = pc[ids]
        tfm = utils.make_tfm_matrix(tvec=(-0.1, 0.0, 0.3), rpy=(0.05, -0.1, 0.2))
        pc_shuffled_transformed = utils.apply_transform(pc_shuffled, tfm)
        pc_noisy_transformed = utils.apply_transform(pc_noisy, tfm)

        T_noisy = q4_a(pc_noisy_transformed, pc)
        T_shuffled = q4_a(pc_shuffled_transformed, pc)

        pc_noisy_untransformed = utils.apply_transform(pc_noisy_transformed, T_noisy)
        pc_shuffled_untransformed = utils.apply_transform(pc_shuffled_transformed, T_shuffled)

        f = plt.figure(figsize=(10, 5))
        ax = f.add_subplot(121, projection='3d')
        ax.scatter(*pc.T, c='r', alpha=0.3)
        ax.scatter(*pc_noisy_untransformed.T, marker='^', c='g', alpha=0.3)
        ax.set_title('with noise')

        ax = f.add_subplot(122, projection='3d')
        ax.scatter(*pc.T, c='r', alpha=0.3)
        ax.scatter(*pc_shuffled_untransformed.T, marker='^', c='g', alpha=0.3)
        ax.set_title('with shuffling')
        plt.show()

    elif question == 'q4_c':
        pc = np.load('bunny.npy')
        ids = np.arange(len(pc))
        np.random.seed(0)
        np.random.shuffle(ids)
        pc_transformed = pc[ids] + 0.001 * np.random.randn(*pc.shape)
        tfm = utils.make_tfm_matrix(tvec=(-0.1, 0.0, 0.3), rpy=(0.05, -0.1, 0.2))
        pc_transformed = utils.apply_transform(pc_transformed, tfm)

        T = q4_c(pc_transformed, pc)

        pc_untransformed = utils.apply_transform(pc_transformed, T)

        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(*pc.T, c='r', alpha=0.3)
        ax.scatter(*pc_untransformed.T, marker='^', c='g', alpha=0.3)
        plt.show()

    else:
        print('Invalid question: choose from '
              '{q1_a, q1_c, q2, q3, q4_a, q4_b, q4_c}.')


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append(0)

    hw4(sys.argv[1])
