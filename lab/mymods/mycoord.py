# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:53:21 2024
@author: santaro

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from pathlib import Path

from matplotlib import colormaps

"""
the shape of points is aligned to:
    (number of points, number of dimension)
    (number of frames, number of points, number of dimension)

"""

class CoordTransformer2d:
    def __init__(self, name='', local_origin=np.zeros(2), theta=0):
        self.name = name
        self.local_origin = np.asarray(local_origin)
        self.theta = np.atleast_1d(theta)
        self.R = CoordTransformer2d.make_rotation_matrix(self.theta)
        self.R_inv = CoordTransformer2d.make_rotation_matrix(-self.theta)

    @staticmethod
    def make_rotation_matrix(theta):
        theta = np.atleast_1d(theta)
        R = np.stack([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]
                      ], axis=0).transpose(2, 0, 1)
        return R

    @staticmethod
    def rotate(p, theta, center=np.zeros(2)):
        p = np.atleast_2d(p)
        num_frames = max(len(p), len(theta))
        if len(p) > 1 and len(theta) > 1 and len(p) != len(theta):
            raise ValueError(f"dimension mismatch: length of point ({len(p)} and theta {len(theta)}) must be 1 or equal.")
        R = CoordTransformer2d.make_rotation_matrix(theta)
        p_translated = p - center
        p_rotated = (R @ p_translated[:, :, np.newaxis]).reshape(num_frames, 2) #squeeze()
        p_result = p_rotated + center
        return p_result

    def transform_point(self, p, towhich="tolocal"):
        p = np.atleast_2d(p)
        num_frames = max(len(p), len(self.theta))
        if len(p) > 1 and len(self.theta) > 1 and len(p) != len(self.theta):
            raise ValueError(f"dimension mismatch: length of point ({len(p)} and self.theta {len(self.theta)}) must be 1 or equal.")
        if towhich == "tolocal":
            p_translated = p - self.local_origin
            p_transformed = (self.R_inv @ p_translated[:, :, np.newaxis]).reshape(num_frames, 2) #squeeze()
        elif towhich == 'toglobal':
            p_rotated = (self.R @ p[:, :, np.newaxis]).reshape(num_frames, 2) #squeeze()
            p_transformed = p_rotated + self.local_origin
        else:
            raise ValueError(f'Unknown transform direction : {towhich}')
        return p_transformed

    def transform_vector(self, v, towhich="tolocal"):
        v = np.atleast_2d(v)
        num_frames = max(len(v), len(self.theta))
        if len(v) > 1 and len(self.theta) > 1 and len(v) != len(self.theta):
            raise ValueError(f"dimension mismatch: length of vector ({len(v)} and self.theta {len(self.theta)}) must be 1 or equal.")
        if towhich == "tolocal":
            v_transformed = (self.R_inv @ v[:, :, np.newaxis]).reshape(num_frames, 2) #squeeze()
        elif towhich == 'toglobal':
            v_transformed = (self.R @ v[:, :, np.newaxis]).reshape(num_frames, 2) #squeeze()
        else:
            raise ValueError(f'Unknown transform direction : {towhich}')
        return v_transformed

    def transform_orientation(self, o, towhich="tolocal"):
        o = np.atleast_2d(o)
        if len(o) > 1 and len(self.theta) > 1 and len(o) != len(self.theta):
            raise ValueError(f"dimension mismatch: length of orientation ({len(o)} and self.theta {len(self.theta)}) must be 1 or equal.")
        if towhich == "tolocal":
            o_transformed = o - self.theta
        elif towhich == 'toglobal':
            o_transformed = o + self.theta
        else:
            raise ValueError(f'Unknown transform direction : {towhich}')
        return o_transformed

    def transform_coord(self, p, towhich="tolocal"):
        p = np.atleast_2d(p)
        if p.shape[1] == 4:
            p_transformed = self.transform_point(p[:, :2], towhich=towhich)
            v_transformed = self.transform_vector(p[:, 2:], towhich=towhich)
            p_transformed = np.concatenate([p_transformed, v_transformed], axis=-1)
        elif p.shape[1] == 2:
            p_transformed = self.transform_point(p[:, :2], towhich=towhich)
        else:
            raise ValueError(f"[Error] p must be 2-dimension in the {self.transform_coord.__name__}, p: {p.shape}")
        return p_transformed

    @staticmethod
    def polar2cartesian(p):
        p = np.atleast_2d(p)
        if p.shape[1] == 4: p = p[:, :2]
        x = p[:, 0] * np.cos(p[:, 1])
        y = p[:, 0] * np.sin(p[:, 1])
        p_transformed = np.stack([x, y], axis=1)
        return p_transformed
    @staticmethod
    def cartesian2polar(p):
        p = np.atleast_2d(p)
        if p.shape[1] == 4: p = p[:, :2]
        r = np.linalg.norm(p, axis=1)
        theta = np.arctan2(p[:, 1], p[:, 0])
        p_transformed = np.stack([r, theta], axis=1)
        return p_transformed

    def polar_coord(self, p, towhich='tocartesian'):
        p = np.atleast_2d(p)
        if p.shape[1] == 4: p = p[:, :2]
        if towhich == 'topolar':
            r = np.linalg.norm(p, axis=1)
            theta = np.arctan2(p[:, 1], p[:, 0])
            p_transformed = np.stack([r, theta], axis=1)
        elif towhich == 'tocartesian':
            x = p[:, 0] * np.cos(p[:, 1])
            y = p[:, 0] * np.sin(p[:, 1])
            p_transformed = np.stack([x, y], axis=1)
        else:
            raise ValueError(f'Unknown transform direction : {towhich}')
        return p_transformed

class CoordTransformer3d_np:
    def __init__(self, name='', local_origin=np.zeros(3), euler_angles=np.zeros(3), rot_order='zyx'):
        self.name = name
        self.local_origin = local_origin
        self.rot_order = rot_order
        self.euler_angles = np.atleast_2d(euler_angles) # ordered according to rotating sequence
        self.Rx, self.Ry, self.Rz = CoordTransformer3d.make_rotation_matrix(thetax=self.euler_angles[:, 0], thetay=self.euler_angles[:, 1], thetaz=self.euler_angles[:, 2])
        self.Rx_inv, self.Ry_inv, self.Rz_inv = CoordTransformer3d.make_rotation_matrix(thetax=-self.euler_angles[:, 0], thetay=-self.euler_angles[:, 1], thetaz=-self.euler_angles[:, 2])
        if rot_order == 'zyx':
            self.rotM = self.Rz @ self.Ry @ self.Rx
            self.rotM_inv = self.Rz_inv @ self.Ry_inv @ self.Rx_inv
        elif rot_order == 'xyz':
            self.rotM = self.Rx @ self.Ry @ self.Rz
            self.rotM_inv = self.Rx_inv @ self.Ry_inv @ self.Rz_inv

    @staticmethod
    def make_rotation_matrix(thetax, thetay, thetaz):
        thetax, thetay, thetaz = np.atleast_1d(thetax), np.atleast_1d(thetay), np.atleast_1d(thetaz)
        num_frames = len(thetax)
        zero = np.zeros(num_frames)
        one = np.ones(num_frames)
        Rx = np.stack([[one, zero, zero],
                    [zero, np.cos(thetax), -np.sin(thetax)],
                    [zero, np.sin(thetax), np.cos(thetax)]], axis=0).transpose(2, 0, 1)
        Ry = np.stack([[np.cos(thetay), zero, np.sin(thetay)],
                    [zero, one, zero],
                    [-np.sin(thetay), zero, np.cos(thetay)]], axis=0).transpose(2, 0, 1)
        Rz = np.stack([[np.cos(thetaz), -np.sin(thetaz), zero],
                    [np.sin(thetaz), np.cos(thetaz), zero],
                    [zero, zero, one]], axis=0).transpose(2, 0, 1)
        return Rx, Ry, Rz

    @staticmethod
    def rotate_euler(p, euler_angles, rot_order, center=np.zeros(3)):
        p = np.atleast_2d(p)
        euler_angles = np.atleast_2d(euler_angles)
        num_frames = max(len(p), len(euler_angles))
        if len(p) > 1 and len(euler_angles) > 1 and len(p) != len(euler_angles):
            raise ValueError(f"dimension mismatch: length of point ({len(p)} and euler_angles {len(euler_angles)}) must be 1 or equal.")
        thetax = np.full(num_frames, euler_angles[:, 0]) if euler_angles.shape[0] <= 1 else euler_angles[:, 0]
        thetay = np.full(num_frames, euler_angles[:, 1]) if euler_angles.shape[0] <= 1 else euler_angles[:, 1]
        thetaz = np.full(num_frames, euler_angles[:, 2]) if euler_angles.shape[0] <= 1 else euler_angles[:, 2]
        Rx, Ry, Rz = CoordTransformer3d.make_rotation_matrix(thetax=thetax, thetay=thetay, thetaz=thetaz)
        if rot_order == 'zyx':
            p_translated = p - center
            p_rotated = (Rz @ Ry @ Rx @ p_translated[:, :, np.newaxis]).reshape(num_frames, 3) #squeeze()
            p_transformed = p_rotated + center
        elif rot_order == 'xyz':
            p_translated = p - center
            p_rotated = (Rx @ Ry @ Rz @ p_translated[:, :, np.newaxis]).reshape(num_frames, 3) #squeeze()
            p_transformed = p_rotated + center
        else:
            raise ValueError(f'**** rot_order parameter is invalid: {rot_order}.')
        return p_transformed

    @staticmethod
    def rotate_extrinsic(p, euler_angles, rot_order, center=np.zeros(3)):
        p = np.atleast_2d(p)
        euler_angles = np.atleast_2d(euler_angles)
        num_frames = max(len(p), len(euler_angles))
        if len(p) > 1 and len(euler_angles) > 1 and len(p) != len(euler_angles):
            raise ValueError(f"dimension mismatch: length of point ({len(p)} and euler_angles {len(euler_angles)}) must be 1 or equal.")
        thetax = np.full(num_frames, euler_angles[:, 0]) if euler_angles.shape[0] <= 1 else euler_angles[:, 0]
        thetay = np.full(num_frames, euler_angles[:, 1]) if euler_angles.shape[0] <= 1 else euler_angles[:, 1]
        thetaz = np.full(num_frames, euler_angles[:, 2]) if euler_angles.shape[0] <= 1 else euler_angles[:, 2]
        Rx, Ry, Rz = CoordTransformer3d.make_rotation_matrix(thetax=thetax, thetay=thetay, thetaz=thetaz)
        if rot_order == 'zyx':
            p_translated = p - center
            p_rotated = (Rx @ Ry @ Rz @ p_translated[:, :, np.newaxis]).reshape(num_frames, 3) #squeeze()
            p_transformed = p_rotated + center
        elif rot_order == 'xyz':
            p_translated = p - center
            p_rotated = (Rz @ Ry @ Rx @ p_translated[:, :, np.newaxis]).reshape(num_frames, 3) #squeeze()
            p_transformed = p_rotated + center
        else:
            raise ValueError(f'**** rot_order parameter is invalid: {rot_order}.')
        return p_transformed

    def transform_point(self, p, towhich="tolocal"):
        p = np.atleast_2d(p)
        num_frames = max(len(p), len(self.euler_angles))
        if len(p) > 1 and len(self.euler_angles) > 1 and len(p) != len(self.euler_angles):
            raise ValueError(f"dimension mismatch: length of point ({len(p)} and self.euler_angles {len(self.euler_angles)}) must be 1 or equal.")
        if towhich == "tolocal":
            p_translated = p - self.local_origin
            p_transformed = (self.rotM_inv @ p_translated[:, :, np.newaxis]).reshape(num_frames, 3) #squeeze()
        elif towhich == 'toglobal':
            p_rotated = (self.rotM @ p[:, :, np.newaxis]).reshape(num_frames, 3) #squeeze()
            p_transformed = p_rotated + self.local_origin
        else:
            raise ValueError(f'****error: towhich parameter is invalid: {towhich}.')
        return p_transformed

    def transform_vector(self, v, towhich="tolocal"):
        v = np.atleast_2d(v)
        num_frames = max(len(v), len(self.euler_angles))
        if len(v) > 1 and len(self.euler_angles) > 1 and len(v) != len(self.euler_angles):
            raise ValueError(f"dimension mismatch: length of vector ({len(v)} and self.euler_angles {len(self.euler_angles)}) must be 1 or equal.")
        if towhich == "tolocal":
            v_transformed = (self.rotM_inv @ v[:, :, np.newaxis]).reshape(num_frames, 3) #squeeze()
        elif towhich == "toglobal":
            v_transformed = (self.rotM @ v[:, :, np.newaxis]).reshape(num_frames, 3) #squeeze()
        return v_transformed

    def transform_orientation(self, o, towhich="tolocal"):
        o = np.atleast_2d(o)
        num_frames = max(len(o), len(self.theta))
        if len(o) > 1 and len(self.theta) > 1 and len(o) != len(self.theta):
            raise ValueError(f"dimension mismatch: length of orientation ({len(o)} and self.theta {len(self.theta)}) must be 1 or equal.")
        if towhich == "tolocal":
            o_transformed = o - self.euler_angles
        elif towhich == 'toglobal':
            o_transformed = o + self.theta
        else:
            raise ValueError(f'Unknown transform direction : {towhich}')
        return o_transformed

    def transform_coord(self, p, towhich="tolocal"):
        p = np.atleast_2d(p)
        if p.shape[1] == 6:
            p_transformed = self.transform_point(p[:, :3], towhich=towhich)
            v_transformed = self.transform_vector(p[:, 3:], towhich=towhich)
            p_transformed = np.concatenate([p_transformed, v_transformed], axis=-1)
        elif p.shape[1] == 3:
            p_transformed = self.transform_point(p[:, :3], towhich=towhich)
        else:
            raise ValueError(f"[Error] p must be 3-dimension in the {self.transform_coord.__name__}, p: {p.shape}")
        return p_transformed

class CoordTransformer3d:
    def __init__(self, name='', local_origin=np.zeros(3), euler_angles=np.zeros(3), rot_order='zyx'):
        self.name = name
        self.local_origin = local_origin
        self.rot_order = rot_order
        self.euler_angles = np.atleast_2d(euler_angles) # ordered according to rotating sequence
        euler_angles_reordered = CoordTransformer3d.reorder_xyz(euler_angles, rot_order)
        self.rotation = R.from_euler(self.rot_order, euler_angles_reordered)
        self.rotM = self.rotation.as_matrix()
        self.rotM_inv = self.rotation.inv().as_matrix()
    @staticmethod
    def reorder_xyz(euler_angles, rot_order):
        euler_angles = np.atleast_2d(euler_angles)
        d = {"x": 0, "y": 1, "z": 2}
        idx = [d[k.lower()] for k in rot_order]
        return euler_angles[:, idx]
    @staticmethod
    def align_xyz(euler_angles, original_order):
        euler_angles = np.atleast_2d(euler_angles)
        d = {k.lower(): i for i, k in enumerate(original_order)}
        idx = [d["x"], d["y"], d["z"]]
        return euler_angles[:, idx]

    @staticmethod
    def make_affineM(euler_angles, rot_order, center=np.zeros(3)):
        euler_angles = np.atleast_2d(euler_angles)
        num_frames = len(euler_angles)
        euler_angles_reordered = CoordTransformer3d.reorder_xyz(euler_angles, rot_order)
        rotation = R.from_euler(rot_order, euler_angles_reordered)
        center = np.atleast_2d(np.asarray(center))
        rotM = rotation.as_matrix()
        affineM = np.tile(np.eye(4), (num_frames, 1, 1))
        affineM[:, :3, :3] = rotM
        translation = center - np.einsum("...jk,...k->...j", rotM, center)
        affineM[:, :3, 3] = translation
        return affineM

    @staticmethod
    def rotate_point(p, euler_angles, rot_order, center=np.zeros(3)):
        p = np.atleast_2d(p)
        euler_angles = np.atleast_2d(euler_angles)
        num_frames = max(len(p), len(euler_angles))
        if len(p) > 1 and len(euler_angles) > 1 and len(p) != len(euler_angles):
            raise ValueError(f"dimension mismatch: length of point ({len(p)} and euler_angles {len(euler_angles)}) must be 1 or equal.")
        affineM = CoordTransformer3d.make_affineM(euler_angles, rot_order, center=center)
        p_homogeneous = np.concatenate([p, np.ones((len(p), 1))], axis=1)
        p_transformed = (affineM @ p_homogeneous[:, :, np.newaxis]).squeeze(-1)
        return p_transformed[:, :3]

    @staticmethod
    def rotate_orientation(o, euler_angles, rot_order):
        o_reordered = CoordTransformer3d.reorder_xyz(o, rot_order)
        euler_angles_reordered = CoordTransformer3d.reorder_xyz(euler_angles, rot_order)
        r1 = R.from_euler(rot_order, o_reordered)
        r2 = R.from_euler(rot_order, euler_angles_reordered)
        combined_r = r2 * r1
        # orientaion = combined_r.as_euler("XYZ")
        # return orientation
        orientation = combined_r.as_euler(rot_order)
        orientation_reordered = CoordTransformer3d.align_xyz(orientation, rot_order)
        return orientation_reordered

    @staticmethod
    def get_basic_vector(euler_angles, rot_order):
        # u = np.eye(3)
        euler_angles_reordered = CoordTransformer3d.reorder_xyz(euler_angles, rot_order)
        r = R.from_euler(rot_order, euler_angles_reordered)
        # orientation = r.apply(u)
        orientation = r.as_matrix().transpose(0, 2, 1)
        return orientation

    def transform_point(self, p, towhich="tolocal"):
        p = np.atleast_2d(p)
        num_frames = max(len(p), len(self.euler_angles))
        if len(p) > 1 and len(self.euler_angles) > 1 and len(p) != len(self.euler_angles):
            raise ValueError(f"dimension mismatch: length of point ({len(p)} and self.euler_angles {len(self.euler_angles)}) must be 1 or equal.")
        if towhich == "tolocal":
            p_translated = p - self.local_origin
            p_transformed = self.rotation.inv().apply(p_translated)
        elif towhich == 'toglobal':
            p_rotated = self.rotation.apply(p)
            p_transformed = p_rotated + self.local_origin
        else:
            raise ValueError(f'****error: towhich parameter is invalid: {towhich}.')
        return p_transformed

    def transform_vector(self, v, towhich="tolocal"):
        v = np.atleast_2d(v)
        num_frames = max(len(v), len(self.euler_angles))
        if len(v) > 1 and len(self.euler_angles) > 1 and len(v) != len(self.euler_angles):
            raise ValueError(f"dimension mismatch: length of vector ({len(v)} and self.euler_angles {len(self.euler_angles)}) must be 1 or equal.")
        if towhich == "tolocal":
            v_transformed = self.rotation.inv().apply(v)
        elif towhich == "toglobal":
            v_transformed = self.rotation.apply(v)
        return v_transformed

    def transform_orientation(self, o, towhich="tolocal"):
        o = np.atleast_2d(o)
        num_frames = max(len(o), len(self.euler_angles))
        if len(o) > 1 and len(self.euler_angles) > 1 and len(o) != len(self.euler_angles):
            raise ValueError(f"dimension mismatch: length of orientation ({len(o)} and self.euler_angles {len(self.euler_angles)}) must be 1 or equal.")
        o_reordered = CoordTransformer3d.reorder_xyz(o, self.rot_order)
        input_rot = R.from_euler(self.rot_order, o_reordered)
        if towhich == "tolocal":
            new_rot = self.rotation.inv() * input_rot
            # new_rot = input_rot * self.rotation.inv()
        elif towhich == 'toglobal':
            new_rot = self.rotation * input_rot
            # new_rot = input_rot * self.rotation
        else:
            raise ValueError(f'Unknown transform direction : {towhich}')
        r = new_rot.as_euler(self.rot_order)
        r_reordered = CoordTransformer3d.align_xyz(r, self.rot_order)
        return r_reordered

    def transform_coord(self, p, towhich="tolocal"):
        p = np.atleast_2d(p)
        if p.shape[1] == 6:
            p_transformed = self.transform_point(p[:, :3], towhich=towhich)
            v_transformed = self.transform_vector(p[:, 3:], towhich=towhich)
            p_transformed = np.concatenate([p_transformed, v_transformed], axis=-1)
        elif p.shape[1] == 3:
            p_transformed = self.transform_point(p[:, :3], towhich=towhich)
        else:
            raise ValueError(f"[Error] p must be 3-dimension in the {self.transform_coord.__name__}, p: {p.shape}")
        return p_transformed


def visualize_points(ps, colors=['r', 'b', 'y', 'g', 'c']*20, markersizes=[4]*20, center=np.zeros(2), xyrange=2, centermarks=None, closeauto=False):
    fig, ax = plt.subplots(figsize=(12, 12))
    xylim = center + xyrange * np.array([-1, 1])
    ax.set(xlim=xylim, ylim=xylim)
    ax.set_aspect(1)
    ax.grid()
    ax.axhline(y=0, xmin=0.2, xmax=0.8, c='k', ls='--', lw=1)
    ax.axvline(x=0, ymin=0.2, ymax=0.8, c='k', ls='--', lw=1)
    if centermarks is not None:
        for _x, _y in centermarks:
            ax.plot([_x-xyrange*0.1, _x+xyrange*0.1], [_y, _y], c='k', ls='--', lw=1)
            ax.plot([_x, _x], [_y-xyrange*0.1, _y+xyrange*0.1], c='k', ls='--', lw=1)
    for _c, _p in enumerate(ps):
        # _p = _p[np.newaxis, :] if np.ndim(_p) == 1 else _p
        _p = np.atleast_2d(_p)
        ax.scatter(_p[:, 0], _p[:, 1], c=colors[_c], s=markersizes[_c])
        if _p.shape[1] == 4:
            vx = np.cos(_p[:, 2])
            vy = np.sin(_p[:, 3])
            ax.quiver(_p[:, 0], _p[:, 1], vx, vy, scale_units="xy", angles="xy", scale=1, width=0.001)
        elif _p.shape[1] == 6:
            vx = np.cos(_p[:, 3])
            vy = np.sin(_p[:, 4])
            ax.quiver(_p[:, 0], _p[:, 1], vx, vy, scale_units="xy", angles="xy", scale=1, width=0.001)
    if closeauto:
        plt.show(block=False)
        plt.pause(closeauto)
        plt.close()
    elif not closeauto:
        plt.show(block=True)


class BearingGeometoryGalculator:
    def __init__(self, p_Aring, p_Bring, p_balls, p_cage):
        self.p_Aring = p_Aring
        self.p_Bring = p_Bring
        self.p_balls = p_balls
        self.p_cage = p_cage

    def calc_ball_distribution(p_C):
        azms = np.arctan2(-p_C[:, 1, :], p_C[:, 2, :]) # arctan2(y-coord, x-coord), anlysis plane is YZ
        _azms = np.vstack([azms[1:], azms[0]+2*np.pi])
        dazms = _azms - azms
        dazms = np.where(dazms<0, dazms+2*np.pi, dazms)
        dazms = np.where(dazms>2*np.pi, dazms-2*np.pi, dazms)
        dazms_deg = np.degrees(dazms)
        return dazms_deg

def calc_ball_distribution(p_C):
    azms = np.arctan2(-p_C[:, 1, :], p_C[:, 2, :]) # arctan2(y-coord, x-coord), anlysis plane is YZ
    _azms = np.vstack([azms[1:], azms[0]+2*np.pi])
    dazms = _azms - azms
    dazms = np.where(dazms<0, dazms+2*np.pi, dazms)
    dazms = np.where(dazms>2*np.pi, dazms-2*np.pi, dazms)
    dazms_deg = np.degrees(dazms)
    return dazms_deg


def sciypy_test():
    #### scipy
    euler = np.radians(np.array([80, 0, 10]))
    rotation = R.from_euler("zyx", euler)
    print(f"rotation : {rotation.as_quat()}")
    print(f"rotation : {rotation.as_matrix()}")
    print(f"rotation : {np.degrees(rotation.as_euler("zyx"))}")

    mat = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    euler = R.from_matrix(mat)
    print(f"euler: {np.degrees(euler.as_euler("zyx"))}")

    rots = R.from_euler("z", [0, 90, 100, 400], degrees=True)
    print(f"rots: {rots.as_matrix().shape}")


if __name__ == '__main__':
    print('----- test -----\n')

    #### 2d
    # num = 8
    # theta = np.linspace(0, 2*np.pi, num)
    # transformer2d = CoordTransformer2d(name='sample', local_origin=np.zeros(2), theta=np.radians(30))
    # p = np.vstack([np.arange(num), np.zeros(num), np.zeros(num), np.zeros(num)]).T
    # p = transformer2d.polar_coord(p, towhich='topolar')
    # p2 = transformer2d.transform_coord(p, towhich='toglobal')
    # print(f"p.shape: {p.shape}")
    # print(f"p2.shape: {p2.shape}")
    # visualize_points([p, p2], xyrange=100)

    #### 3d
    # num = 100
    # euler_angles = np.vstack([np.zeros(num), np.zeros(num), np.linspace(0, 1*np.pi, num)]).T
    # transformer3d = CoordTransformer3d(name='sample3d', local_origin=np.zeros(3), euler_angles=euler_angles)
    # p = np.array([20, 0, 0])
    # p = np.array([20, 0, 0, 0, 0, 0])
    # euler_angles = np.array([0, 0, 1])
    # p2 = CoordTransformer3d.rotate_euler(p[:3], euler_angles=euler_angles, rot_order='zyx')
    # p2 = transformer3d.transform_coord(p)
    # visualize_points([p2, p], xyrange=100, markersizes=[1, 100])

    a = np.radians(np.array([0, 0, 10]))
    b = np.radians(np.array([0, 30, 40]))

    x = CoordTransformer3d.rotate_orientation(a, b, rot_order="zyx")
    print(f"x: {np.degrees(x)}")

    euler_angles = np.array([np.radians(45), 0, 0])
    v = CoordTransformer3d.get_basic_vector(euler_angles, rot_order="zyx")
    print(f"v: {v}")


    # angles = np.array([45, 0, 0])
    # angles = np.array([[30, 0, 0],
    #                    [20, 0, 0]])
    # angles = np.radians(angles)

    # m = CoordTransformer3d.make_affineM(angles, rot_order="zyx", center=np.zeros(3))
    # print(m.shape)
    # print(m)

    # p = np.array([0, 1, 0])
    # print(f"p: {p}")
    # print(f"angles: {np.degrees(angles)}")
    # p = CoordTransformer3d.rotate_euler(p, angles, rot_order="zyx", center=np.zeros(3))
    # print(f"p: {p}")