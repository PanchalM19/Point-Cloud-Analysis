import numpy as np
from scipy.spatial.transform import Rotation


def make_tfm_matrix(tvec, rpy) -> np.ndarray:
    '''Returns 4x4 random homogeneous transformation matrix

    Arguments
    ---------
    tvec : array_like
        translation vector in 3d
    rpy : array_like
        roll pitch yaw in radians
    '''
    matrix = np.eye(4)
    matrix[:3, 3] = tvec
    matrix[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    return matrix


def apply_transform(pts, tfm):
    '''Applies homogenous transform to points

    pts: Nx3 array
    tfm: 4x4 transformation matrix
    '''
    return np.einsum('ij,nj->ni', tfm[:3, :3], pts) + tfm[:3, 3]


def make_plane3d(ngrids, scale=1.0, tfm=None):
    '''

    Arguments
    ---------
    ngrids : int
        number of points along each dimension of plane
    scale : float
        scale of the plane
    scale : np.ndarray, optional
        4x4 transformation matrix that is applied to the plane

    Returns
    -------
    plane : np.ndarray
        Nx3 array of points that form a plane in 3d space, N=ngrids**2
    '''
    # make evenly spaced grid in xy
    grid = np.mgrid[0:ngrids, 0:ngrids].reshape(2, -1).T

    # add z dimension
    grid = np.concatenate([grid, np.zeros((grid.shape[0], 1))], 1)

    # scale
    grid = scale * grid / (ngrids-1)

    if tfm is not None:
        grid = apply_transform(grid, tfm)

    return grid


def within_roi(pc, roi):
    '''
    Return new point cloud where all points fit within 3D bounding box
    (e.g. region of interest)

    Arguments
    ---------
    pc : np.ndarray
        Nx3 array of points
    roi : np.ndarray
        3x2 array specifying 3D bounding box (inclusive)
        ((xmin, xmax), (ymin, ymax), (zmin, zmax))

    Returns
    -------
    mask : np.ndarray
        boolean array indicating which of N points are within roi, shape=(N,)
    '''
    mask = np.bitwise_and.reduce([
        pc[:, 0] >= roi[0, 0],
        pc[:, 0] <= roi[0, 1],
        pc[:, 1] >= roi[1, 0],
        pc[:, 1] <= roi[1, 1],
        pc[:, 2] >= roi[2, 0],
        pc[:, 2] <= roi[2, 1],
    ])
    return mask


def plot_plane(ax, normal, center):
    '''Plots plane in 3D
    '''
    xx, yy = np.meshgrid(np.linspace(-1, 1, 2, endpoint=True),
                         np.linspace(0, 2, 2, endpoint=True),
                        )
    zz = (-normal[0] * xx - normal[1] * yy + center.dot(normal)) / normal[2]

    ax.plot_surface(xx, yy, zz, alpha=0.2, color='g')


def plot_sphere(ax, center, radius):
    '''Plots sphere in 3D
    '''
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u)*np.sin(v) + center[0]
    y = radius * np.sin(u)*np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    ax.plot_wireframe(x, y, z, color="r", alpha=0.4)


def plot_cylinder(ax, center, axis, radius):
    '''Plots cylinder in 3D
    '''
    if axis[2] < 0:
        axis *= -1

    t = np.linspace(0.0, 0.3, 8)
    theta = np.linspace(0, 2*np.pi, 20)
    theta_grid, t_grid = np.meshgrid(theta, t)
    x = radius * np.cos(theta_grid) + axis[0] * t_grid + center[0]
    y = radius * np.sin(theta_grid) + axis[1] * t_grid + center[1]
    z = t_grid * axis[2] + center[2]

    ax.plot_wireframe(x, y, z, color="r", alpha=0.4)
