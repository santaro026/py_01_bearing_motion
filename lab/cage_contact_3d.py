"""
Created on Mon Feb 09 18:08:17 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.lines as mlines

from pathlib import Path
import pickle
import sys
import time

import multiprocessing as mp

from sympy import Point, Line, Circle, sqrt
from sympy import Point3D, Line3D, Plane
from sympy import Matrix, sin, cos, tan, rad, asin, acos, atan2

from mymods import mycoord, myplotter, mylogger
import config

x_axis = Line3D(Point3D(0, 0, 0), direction_ratio=[1, 0, 0])
y_axis = Line3D(Point3D(0, 0, 0), direction_ratio=[0, 1, 0])
z_axis = Line3D(Point3D(0, 0, 0), direction_ratio=[0, 0, 1])

def rotatex(point, theta, center=np.zeros(3)):
    px = point.x - center[0]
    py = point.y - center[1]
    pz = point.z - center[2]
    v = Matrix([px, py, pz])
    R = Matrix([
        [1, 0, 0],
        [0, cos(theta), -sin(theta)],
        [0, sin(theta), cos(theta)],
    ])
    v_rot = R * v
    x = v_rot[0] + center[0]
    y = v_rot[1] + center[1]
    z = v_rot[2] + center[2]
    return Point3D(x, y, z)
def rotatey(point, theta, center=np.zeros(3)):
    px = point.x - center[0]
    py = point.y - center[1]
    pz = point.z - center[2]
    v = Matrix([px, py, pz])
    R = Matrix([
        [cos(theta), 0, sin(theta)],
        [0, 1, 0],
        [-sin(theta), 0, cos(theta)],
    ])
    v_rot = R * v
    x = v_rot[0] + center[0]
    y = v_rot[1] + center[1]
    z = v_rot[2] + center[2]
    return Point3D(x, y, z)
def rotatez(point, theta, center=np.zeros(3)):
    px = point.x - center[0]
    py = point.y - center[1]
    pz = point.z - center[2]
    v = Matrix([px, py, pz])
    R = Matrix([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0, 0, 1],
    ])
    v_rot = R * v
    x = v_rot[0] + center[0]
    y = v_rot[1] + center[1]
    z = v_rot[2] + center[2]
    return Point3D(x, y, z)
def get_rigid_rotation_matrix(cage, theta):
    x = cage[0]
    y = cage[1]
    z = cage[2]
    yz_len = sqrt(y**2 + z**2)
    if yz_len == 0:
        return Matrix.eye(3)
    ux, uy, uz = 0, -z / yz_len, y / yz_len
    c = cos(theta)
    s = sin(theta)
    C = 1 - c
    R = Matrix([
        [c + ux**2*C,    ux*uy*C - uz*s,  ux*uz*C + uy*s],
        [uy*ux*C + uz*s, c + uy**2*C,    uy*uz*C - ux*s],
        [uz*ux*C - uy*s, uz*uy*C + ux*s, c + uz**2*C   ]
    ])
    return R
def rotate_rigid(point, R, center=np.zeros(3)):
    v = Matrix([point.x - center[0], point.y - center[1], point.z - center[2]])
    v_rot = R * v
    return Point3D([v_rot[0] + center[0], v_rot[1] + center[1], v_rot[2] + center[2]])

class ARing:
    def __init__(self, name="", PCD=50, ID=48, OD=52,  R=3.2):
        self.name = name
        self.PCD = PCD
        self.ID = ID
        self.OD = OD
        self.R = R
        self.groove = Circle((0, 0), self.PCD/2 + self.R)
    def visualize(self):
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        groove = patches.Circle((0, 0), self.groove.radius, color='k', fill=False)
        axs[0].add_patch(groove)
        axs[0].set(xlim=(-30, 30), ylim=(-30, 30))
        plt.show()

class BRing:
    def __init__(self, name="", PCD=50, ID=48, OD=52,  R=3.2):
        self.name = name
        self.PCD = PCD
        self.ID = ID
        self.OD = OD
        self.R = R
        self.groove = Circle((0, 0), self.PCD/2 - self.R)
    def visualize(self):
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        groove = patches.Circle((0, 0), self.groove.radius, color='k', fill=False)
        axs[0].add_patch(groove)
        axs[0].set(xlim=(-30, 30), ylim=(-30, 30))
        plt.show()

class Ball:
    def __init__(self, name="", PCD=50, num_balls=8, Dw=5.953):
        self.name = name
        self.PCD = PCD
        self.num_balls = num_balls
        self.Dw = Dw
        self.center = Point3D(0, 0, 0)
        self.pos_balls = np.linspace(0, 2*np.pi, self.num_balls, endpoint=False) + np.pi/2
        self.ball_centers = []
        for i in range(self.num_balls):
            _c = rotatex(Point3D(0, 0, 0).translate(0, self.PCD/2, 0), self.pos_balls[i])
            self.ball_centers.append(_c)
    def translate(self, xyz):
        self.center = self.center.translate(xyz[0], xyz[1], xyz[2])
        for i in range(self.num_balls):
            self.ball_centers[i] = self.ball_centers[i].translate(xyz[0], xyz[1], xyz[2])
    def rotatex(self, theta, center=np.zeros(3)):
        self.center = rotatex(self.center, theta, center=center)
        for i in range(self.num_balls):
            self.ball_centers[i] = rotatex(self.ball_centers[i], theta, center)
    def rotatey(self, theta, center=np.zeros(3)):
        self.center = rotatey(self.center, theta, center=center)
        for i in range(self.num_balls):
            self.ball_centers[i] = rotatey(self.ball_centers[i], theta, center)
    def rotatez(self, theta, center=np.zeros(3)):
        self.center = rotatez(self.center, theta, center=center)
        for i in range(self.num_balls):
            self.ball_centers[i] = rotatez(self.ball_centers[i], theta, center)
    def tilt(self, xyz, theta, center=np.zeros(3)):
        R = get_rigid_rotation_matrix(Point3D(xyz[0], xyz[1], xyz[2]), theta=theta)
        for i in range(self.num_balls):
            self.ball_centers[i] = self.ball_centers[i].translate(-xyz[0], -xyz[1], -xyz[2])
            self.ball_centers[i] = rotate_rigid(self.ball_centers[i], R, center)
            self.ball_centers[i] = self.ball_centers[i].translate(xyz[0], xyz[1], xyz[2])
    def visualize(self, plane="yz"):
        axis_dict = {'x': 0, 'y': 1, 'z': 2}
        _x = axis_dict[plane[0]]
        _y = axis_dict[plane[1]]
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        center = [self.center.x, self.center.y, self.center.y]
        axs[0].scatter(center[_x], center[_y], s=10, c='b')
        for i in range(self.num_balls):
            _c = np.array([self.ball_centers[i].x, self.ball_centers[i].y, self.ball_centers[i].z])
            _b = patches.Circle((_c[_x], _c[_y]), self.Dw/2, color='k', fill=False)
            axs[0].add_patch(_b)
            axs[0].scatter(_c[_x], _c[_y], s=10, c='k')
        axs[0].set(xlim=(-30, 30), ylim=(-30, 30))
        axs[0].axhline(y=0, c='k', lw=1)
        axs[0].axvline(x=0, c='k', lw=1)
        plt.show()

class SimpleCage:
    def __init__(self, name="", PCD=50, ID=48, OD=52, width=10, num_pockets=8, Dp=6.25):
        self.name = name
        self.cage = Point3D(0, 0, 0)
        self.PCD = PCD
        self.ID = ID
        self.OD = OD
        self.width = width
        self.num_pockets = num_pockets
        self.Dp = Dp
        self.pos_pockets = np.linspace(0, 2*np.pi, self.num_pockets, endpoint=False) + np.pi/2
        self.pocket_centers = []
        for i in range(self.num_pockets):
            _c = rotatex(Point3D(0, self.PCD/2, 0), self.pos_pockets[i])
            self.pocket_centers.append(_c)
        self.history = []
    def _generate_pocket_line(self):
        _theta_id = np.arcsin(self.Dp / self.ID)
        _theta_od = np.arcsin(self.Dp / self.OD)
        l0y = -(self.PCD - self.ID * np.cos(_theta_id))/2
        l1y = (self.OD * np.cos(_theta_od) - self.PCD)/2
        l0z = self.Dp/2
        l1z = -self.Dp/2
        l00 = Point3D(0, l0y, l0z)
        l01 = Point3D(0, l1y, l0z)
        l10 = Point3D(0, l0y, l1z)
        l11 = Point3D(0, l1y, l1z)
        self.pocket_lines = []
        for i in range(self.num_pockets):
            _l00 = rotatex(l00.translate(0, self.PCD/2, 0), self.pos_pockets[i])
            _l01 = rotatex(l01.translate(0, self.PCD/2, 0), self.pos_pockets[i])
            _l10 = rotatex(l10.translate(0, self.PCD/2, 0), self.pos_pockets[i])
            _l11 = rotatex(l11.translate(0, self.PCD/2, 0), self.pos_pockets[i])
            for k, v1, v2, v3 in self.history:
                if k == "translate":
                    _l00 = _l00.translate(v1[0], v1[1], v1[2])
                    _l01 = _l01.translate(v1[0] ,v1[1], v1[2])
                    _l10 = _l10.translate(v1[0], v1[1], v1[2])
                    _l11 = _l11.translate(v1[0], v1[1], v1[2])
                if k == "rotate":
                    _l00 = v1(_l00, v2, v3)
                    _l01 = v1(_l01, v2, v3)
                    _l10 = v1(_l10, v2, v3)
                    _l11 = v1(_l11, v2, v3)
            _l0 = Line3D(_l00, _l01)
            _l1 = Line3D(_l10, _l11)
            _ls = [_l0, _l1]
            self.pocket_lines.append(_ls)
    # def _generate_outline(self):
    #     for k, v1, v2 in self.history:
    #         translate_history = np.zeros(3)
    #         rotate_history = []
    #         if k == "translate":
    #             translate_history += np.asarray(v1)
    #         elif k == "rotate":
    #             rotate_history.append([v1, v2])
    def translate(self, xyz):
        for i in range(self.num_pockets):
            self.pocket_centers[i] = self.pocket_centers[i].translate(xyz[0], xyz[1], xyz[2])
        self.cage = self.cage.translate(xyz[0], xyz[1], xyz[2])
        self.history.append(["translate", xyz, None, None])
    def rotatex(self, theta, center=np.zeros(3)):
        for i in range(self.num_pockets):
            self.pocket_centers[i] = rotatex(self.pocket_centers[i], theta, center)
        self.cage = rotatex(self.cage, theta, center)
        self.history.append(["rotate", rotatex, theta, center])
    def rotatey(self, theta, center=np.zeros(3)):
        for i in range(self.num_pockets):
            self.pocket_centers[i] = rotatey(self.pocket_centers[i], theta, center)
        self.cage = rotatey(self.cage, theta, center)
        self.history.append(["rotate", rotatey, theta, center])
    def rotatez(self, theta, center=np.zeros(3)):
        for i in range(self.num_pockets):
            self.pocket_centers[i] = rotatez(self.pocket_centers[i], theta, center)
        self.cage = rotatez(self.cage, theta, center)
        self.history.append(["rotate", rotatez, theta, center])
    def visualize(self, plane="yz"):
        axis_dict = {'x': 0, 'y': 1, 'z': 2}
        _x = axis_dict[plane[0]]
        _y = axis_dict[plane[1]]
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        self._generate_pocket_line()
        for i in range(self.num_pockets):
            _p11 = np.array(self.pocket_lines[i][0].p1)
            _p12 = np.array(self.pocket_lines[i][0].p2)
            _p1 = np.column_stack([_p11, _p12])
            _p21 = np.array(self.pocket_lines[i][1].p1)
            _p22 = np.array(self.pocket_lines[i][1].p2)
            _p2 = np.column_stack([_p21, _p22])
            _l1 = mlines.Line2D(_p1[_x], _p1[_y], color='b')
            _l2 = mlines.Line2D(_p2[_x], _p2[_y], color='b')
            axs[0].add_line(_l1)
            axs[0].add_line(_l2)
            _c = np.array([self.pocket_centers[i].x, self.pocket_centers[i].y, self.pocket_centers[i].z])
            axs[0].scatter(_c[_x], _c[_y], s=10, c='b' )
        _c = [self.cage.x, self.cage.y, self.cage.z]
        _c = [_c[_x], _c[_y]]
        _id = patches.Circle(_c, self.ID/2, color='k', fill=False)
        _od = patches.Circle(_c, self.OD/2, color='k', fill=False)
        axs[0].add_patch(_id)
        axs[0].add_patch(_od)
        axs[0].set(xlim=(-30, 30), ylim=(-30, 30))
        plt.show()

class Bearing:
    def __init__(self, aring, bring, ball, cage):
        self.aring = aring
        self.bring = bring
        self.ball = ball
        self.cage = cage
        self.distances = np.zeros(self.ball.num_balls)
        self.n_vct = np.zeros((self.ball.num_balls, 3))
        self.n_vct_local = np.zeros((self.ball.num_balls, 3))
        self.p_contact = np.zeros((self.ball.num_balls, 3))
        self.normal = np.zeros((self.ball.num_balls, 3))
        self.fric = np.zeros((self.ball.num_balls, 3))
        self.force = np.zeros((self.ball.num_balls, 3))
    def calc_contact_ball2cylinder(self):
        for i in range(self.ball.num_balls):
            ball = self.ball.ball_centers[i]
            pocket = self.cage.pocket_centers[i]
            ball = rotatex(ball, -self.cage.pos_pockets[i]).translate(0, -self.cage.PCD/2, 0) # to pocket local coord
            pocket = rotatex(pocket, -self.cage.pos_pockets[i]).translate(0, -self.cage.PCD/2, 0) # to pocket local coord
            ball_zx = Point3D(ball.x, 0, ball.z)
            pocket_zx = Point3D(pocket.x, 0, pocket.z)
            self.distances[i] = self.cage.Dp/2 - pocket_zx.distance(ball_zx) - self.ball.Dw/2
            n_vct = rotatex(Point3D(ball.x, 0, ball.z), self.cage.pos_pockets[i])
            self.n_vct_local[i] = np.array([ball.x, 0, ball.z])
            self.n_vct[i] = np.array([n_vct.x, n_vct.y, n_vct.z])
            self.n_vct[i] = self.n_vct[i] / np.linalg.norm(self.n_vct[i])
            theta = atan2(ball.x, ball.z)
            _c2d = Circle((0, 0), self.cage.Dp/2)
            _cpz = _c2d.center.x + _c2d.radius * cos(theta)
            _cpx = _c2d.center.y + _c2d.radius * sin(theta)
            p_contact_local = Point3D(_cpx, ball.y, _cpz)
            p_contact = rotatex(p_contact_local.translate(0, self.cage.PCD/2, 0), self.cage.pos_pockets[i])
            self.p_contact[i] = np.array([p_contact.x, p_contact.y, p_contact.z])
    def calc_contact_force(self, k=10**8, d=1, eta=0.2):
        for i in range(self.ball.num_balls):
            if self.distances[i] <= 0:
                _f = -k * (self.distances[i]/1000) ** d
                n_vct = Matrix([self.n_vct[i, 0], self.n_vct[i, 1], self.n_vct[i, 2]])
                normal = _f * n_vct
                self.normal[i] = np.array([normal[0], normal[1], normal[2]])
                if self.n_vct_local[i, 2] >= 0:
                    dir_fric = rotatex(Point3D(0, 1, 0), self.cage.pos_pockets[i])
                elif self.n_vct_local[i, 2] <= 0:
                    dir_fric = rotatex(Point3D(0, -1, 0), self.cage.pos_pockets[i])
                self.fric[i] = eta * np.linalg.norm(self.normal[i]) * np.array([dir_fric.x, dir_fric.y, dir_fric.z])
                self.force[i] = self.normal[i] + self.fric[i]
    def visualize(self, plane="yz", vwidth=0.004, vscale=0.01):
        axis_dict = {'x': 0, 'y': 1, 'z': 2}
        _x = axis_dict[plane[0]]
        _y = axis_dict[plane[1]]
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        if self.aring:
            groove_a = patches.Circle((0, 0), self.aring.groove.radius, color='k', fill=False)
            axs[0].add_patch(groove_a)
        if self.bring:
            groove_b = patches.Circle((0, 0), self.bring.groove.radius, color='k', fill=False)
            axs[0].add_patch(groove_b)
        center = [self.ball.center.x, self.ball.center.y, self.ball.center.y]
        axs[0].scatter(center[_x], center[_y], s=10, c='b')
        for i in range(self.ball.num_balls):
            _c = np.array([self.ball.ball_centers[i].x, self.ball.ball_centers[i].y, self.ball.ball_centers[i].z])
            _b = patches.Circle((_c[_x], _c[_y]), self.ball.Dw/2, color='k', fill=False)
            axs[0].add_patch(_b)
            axs[0].scatter(_c[_x], _c[_y], s=10, c='k')
        self.cage._generate_pocket_line()
        for i in range(self.cage.num_pockets):
            _p11 = np.array(self.cage.pocket_lines[i][0].p1)
            _p12 = np.array(self.cage.pocket_lines[i][0].p2)
            _p1 = np.column_stack([_p11, _p12])
            _p21 = np.array(self.cage.pocket_lines[i][1].p1)
            _p22 = np.array(self.cage.pocket_lines[i][1].p2)
            _p2 = np.column_stack([_p21, _p22])
            _l1 = mlines.Line2D(_p1[_x], _p1[_y], color='b')
            _l2 = mlines.Line2D(_p2[_x], _p2[_y], color='b')
            axs[0].add_line(_l1)
            axs[0].add_line(_l2)
            _c = np.array([self.cage.pocket_centers[i].x, self.cage.pocket_centers[i].y, self.cage.pocket_centers[i].z])
            axs[0].scatter(_c[_x], _c[_y], s=10, c='b' )
            if self.distances[i] <= 0:
                cage_xyz = -np.array([self.ball.center.x, self.ball.center.y, self.ball.center.z], dtype=float)
                _n = np.cross(np.array([1, 0, 0]), cage_xyz)
                _n = _n / np.linalg.norm(_n)
                _f = np.dot(self.normal[i], _n) * _n
                axs[0].quiver(self.p_contact[i, _x], self.p_contact[i, _y], _f[_x], _f[_y], color='b', width=vwidth, scale_units="xy", angles="xy", scale=vscale)
                _f = np.dot(self.fric[i], _n) * _n
                axs[0].quiver(self.p_contact[i, _x], self.p_contact[i, _y], _f[_x], _f[_y], color='r', width=vwidth, scale_units="xy", angles="xy", scale=vscale)
                # axs[0].quiver(self.p_contact[i, _x], self.p_contact[i, _y], self.normal[i, _x], self.normal[i, _y], color='b', width=vwidth, scale_units="xy", angles="xy", scale=vscale)
                # axs[0].quiver(self.p_contact[i, _x], self.p_contact[i, _y], self.fric[i, _x], self.fric[i, _y], color='r', width=vwidth, scale_units="xy", angles="xy", scale=vscale)
                # axs[0].quiver(self.p_contact[i, _x], self.p_contact[i, _y], self.force[i, _x], self.force[i, _y], color='k', width=vwidth, scale_units="xy", angles="xy", scale=vscale)
        _c = [self.cage.cage.x, self.cage.cage.y, self.cage.cage.z]
        _c = [_c[_x], _c[_y]]
        _id = patches.Circle(_c, self.cage.ID/2, color='k', fill=False)
        _od = patches.Circle(_c, self.cage.OD/2, color='k', fill=False)
        axs[0].add_patch(_id)
        axs[0].add_patch(_od)
        axs[0].set(xlim=(-30, 30), ylim=(-30, 30))
        axs[0].axhline(y=0, c='k', lw=1)
        axs[0].axvline(x=0, c='k', lw=1)
        plt.show()

def calc_scalar_field(N=21, xyrange=0.1, theta=np.radians(0), name="scalar_field"):
    logger = mylogger.MyLogger(name, outdir=outdir, mode='w')
    logger.measure_time("main", 's')
    cage = SimpleCage()
    aring = None
    bring = None
    Y, Z = np.meshgrid(np.linspace(-xyrange, xyrange, N, endpoint=True), np.linspace(-xyrange, xyrange, N, endpoint=True))
    fric = np.zeros((N, N, 3))
    total = N * N
    c = 0
    for i in range(N):
        for j in range(N):
            ball = Ball()
            bearing = Bearing(aring, bring, ball, cage)
            ball_xyz = -np.array([0, Y[i, j], Z[i, j]])
            ball.tilt(ball_xyz, -theta)
            ball.translate(ball_xyz)
            bearing.calc_contact_ball2cylinder()
            bearing.calc_contact_force()
            cage_xyz = -np.array([bearing.ball.center.x, bearing.ball.center.y, bearing.ball.center.z], dtype=float)
            _n = np.cross(np.array([1, 0, 0]), cage_xyz)
            _n = _n / np.linalg.norm(_n)
            _fric = np.zeros(3)
            for k in range(bearing.ball.num_balls):
                _f = np.dot(bearing.fric[k], _n) * _n
                _fric += _f
            fric[i, j, :] = _fric
            c += 1
            progress = c / total * 100
            if c % 20 == 0:
                logger.info(f"progress: {progress:.1f} %, {c}/{total}")
    plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
    fig, axs = plotter.myfig()
    axs[0].quiver(Y, Z, fric[:, :, 1], fric[:, :, 2], scale_units="xy", angles="xy", scale=2, width=0.002, color='r')
    # axs[0].set(xlim=(-xyrange, xyrange), ylim=(-xyrange, xyrange))
    axs[0].set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    axs[0].axhline(y=0, c='k', lw=1)
    axs[0].axvline(x=0, c='k', lw=1)
    with open(outdir/f"{name}.pkl", "wb") as f:
        pickle.dump(fric, f)
    logger.measure_time("main", 'e')

def calc_scalar_field_parallel(N=21, xyrange=0.1, theta=np.radians(0), num_balls=8, name="scalar_field", ncpu=None, log_freq=100):
    import time
    st = time.perf_counter()
    logger = mylogger.MyLogger(name, outdir=outdir, mode='w')
    logger.measure_time("main", 's')
    Y, Z = np.meshgrid(np.linspace(-xyrange, xyrange, N, endpoint=True), np.linspace(-xyrange, xyrange, N, endpoint=True))
    total = N * N
    res_normal = []
    res_fric = []
    args_list = [(y, z, theta, num_balls) for y, z in zip(Y.ravel(), Z.ravel())]
    if ncpu is None: ncpu = mp.cpu_count() - 1
    logger.binfo(f"jobname: {name}\nncpu: {ncpu}/{mp.cpu_count()}\nnum_of_mesh: {N*N} ({N}*{N})\ntilt_angle: {np.degrees(theta):.2f}")
    with mp.Pool(processes=ncpu) as pool:
        for i, res in enumerate(pool.imap(compute_pixel, args_list), 1):
            normal, fric = res
            res_normal.append(normal)
            res_fric.append(fric)
            if i % log_freq == 0 or i == total:
                progress = i / total
                _now = time.perf_counter()
                logger.info(f"progress: {progress*100:.2f} % ({i}/{total}), estimated time: {(_now-st)*(1/progress-1):.2f} [sec]")
    num_pockets = res_fric[0].shape[0]
    res_normal = np.array(res_normal).reshape(N, N, num_pockets, 3)
    res_fric = np.array(res_fric).reshape(N, N, num_pockets, 3)
    with open(outdir/f"{name}_normal.pkl", "wb") as f:
        pickle.dump(res_normal, f)
    with open(outdir/f"{name}_fric.pkl", "wb") as f:
        pickle.dump(res_fric, f)
    logger.measure_time("main", 'e')

def calc_force_on_r_parallel(r=0.2, N=100, theta=np.radians(0), num_balls=8, name="force_on_r", log_freq=100, ncpu=None):
    st = time.perf_counter()
    logger = mylogger.MyLogger(name, outdir=outdir, mode='w')
    logger.measure_time("main", 's')
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = r * np.cos(t)
    z = r * np.sin(t)
    fric = np.zeros((N, 3))
    total = N
    res_normal = []
    res_fric = []
    args_list = [(y, z, theta, num_balls) for y, z in zip(y, z)]
    if ncpu is None: ncpu = mp.cpu_count() - 1
    logger.binfo(f"jobname: {name}\nncpu: {ncpu}/{mp.cpu_count()}\nnum_of_node: {N}\nrevolution_radius: {r}\ntilt_angle: {np.degrees(theta):.2f}")
    with mp.Pool(processes=ncpu) as pool:
        for i, res in enumerate(pool.imap(compute_pixel, args_list), 1):
            normal, fric = res
            res_normal.append(normal)
            res_fric.append(fric)
            if i % log_freq == 0 or i == total:
                progress = i / total
                _now = time.perf_counter()
                logger.info(f"progress: {progress*100:.2f} % ({i}/{total}), estimated time: {(_now-st)*(1/progress-1):.2f} [sec]")
    num_pockets = res_fric[0].shape[0]
    res_normal = np.array(res_normal).reshape(N, num_pockets, 3)
    res_fric = np.array(res_fric).reshape(N, num_pockets, 3)
    with open(outdir/f"{name}_normal.pkl", "wb") as f:
        pickle.dump(res_normal, f)
    with open(outdir/f"{name}_fric.pkl", "wb") as f:
        pickle.dump(res_fric, f)
    logger.measure_time("main", 'e')

def compute_pixel(args): #args = (y, z, theta)
    y, z, theta, num_balls = args
    aring = None
    bring = None
    cage = SimpleCage(num_pockets=num_balls)
    ball = Ball(num_balls=num_balls)
    bearing = Bearing(aring, bring, ball, cage)
    ball_xyz = -np.array([0, y, z])
    ball.tilt(ball_xyz, -theta)
    ball.translate(ball_xyz)
    bearing.calc_contact_ball2cylinder()
    bearing.calc_contact_force()
    return bearing.normal, bearing.fric

def resolve_force_2rc(force, position, axis=np.array([1, 0, 0])):
    u = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    pos = position - (np.sum(position * u, axis=-1, keepdims=True)) * u
    r_norm = np.linalg.norm(pos, axis=-1, keepdims=True)
    r_unit = np.divide(pos, r_norm, out=np.zeros_like(pos), where=r_norm!=0)
    fr = np.sum(force * r_unit, axis=-1, keepdims=True) * r_unit # instead of np.dot
    _c = np.cross(u, pos)
    c_norm = np.linalg.norm(_c, axis=-1, keepdims=True)
    c_unit = np.divide(_c, c_norm, out=np.zeros_like(pos), where=c_norm!=0)
    fc = np.sum(force * c_unit, axis=-1, keepdims=True) * c_unit
    return fr, fc

def load_scalar_field_data(jobname, datadir=config.ROOT/"results"/"scalar_field", step=1, vscale=1000, vwidth=0.002, xyrange=0.3, valpha=1, vmin=0, vmax=8000, cmap="viridis", mode="vector"):
    with open(datadir/f"{jobname}_normal.pkl", "rb") as f:
        normal = pickle.load(f)
    with open(datadir/f"{jobname}_fric.pkl", "rb") as f:
        fric = pickle.load(f)
    N = normal.shape[0]
    normal[np.isnan(normal)] = 0
    normal = np.sum(normal, axis=-2)[::step, ::step]
    normal_norm = np.linalg.norm(normal, axis=-1)
    fric[np.isnan(fric)] = 0
    fric = np.sum(fric, axis=-2)[::step, ::step]
    fric_norm = np.linalg.norm(fric, axis=-1)
    meshrange = 0.3
    Y, Z = np.meshgrid(np.linspace(-meshrange, meshrange, N, endpoint=True), np.linspace(-meshrange, meshrange, N, endpoint=True))
    Y, Z = Y[::step, ::step], Z[::step, ::step]
    U1, V1 = normal[:, :, 1], normal[:, :, 2]
    U2, V2 = fric[:, :, 1], fric[:, :, 2]
    mask = normal_norm!=0
    plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
    fig, axs = plotter.myfig(slide=True, xrange=(-xyrange, xyrange), yrange=(-xyrange, xyrange), xlabel="z [mm]", ylabel="y [mm]")
    if mode == "vector":
        # axs[0].quiver(Y[mask], Z[mask], U1[mask], V1[mask], scale_units="xy", angles="xy", scale=vscale, width=vwidth, color='b', alpha=valpha)
        axs[0].quiver(Y[mask], Z[mask], U2[mask], V2[mask], scale_units="xy", angles="xy", scale=vscale, width=vwidth, color='b', alpha=valpha)
    elif mode == "scalar":
        # axs[0].pcolormesh(Y, Z, normal_norm, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0].pcolormesh(Y, Z, fric_norm, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0].axhline(y=0, c='k', lw=1)
    axs[0].axvline(x=0, c='k', lw=1)
    axs[0].tick_params(axis='both', pad=14)
    axs[0].set_aspect(1)
    return fig, axs

def load_fc_on_r_data(jobs, datadir=config.ROOT/"results"/"scalar_field"):
    normals, frics = [], []
    lss, colors, labels = [], [], []
    for args in jobs:
        jobname = args["jobname"]
        lss.append(args["ls"])
        colors.append(args["color"])
        labels.append(args["label"])
        f_normal = outdir / f"{jobname}_normal.pkl"
        with open(f_normal, "rb") as f:
            normal = pickle.load(f)
            normals.append(normal)
        f_fric = outdir / f"{jobname}_fric.pkl"
        with open(f_fric, "rb") as f:
            fric = pickle.load(f)
            frics.append(fric)
    N = fric.shape[0]
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    ys = []
    for y in frics:
        y = np.sum(y, axis=-2)
        y = np.linalg.norm(y, axis=-1)
        ys.append(y)
    plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.RECTANGLE_FIG)
    fig, axs = plotter.myfig(slide=True, xsigf=0, ysigf=0, xlabel="cage position angle [degree]", ylabel="friction force", xrange=(0, 360), yrange=(0, 200))
    for i in range(len(jobs)):
        axs[0].plot(np.degrees(t), ys[i], lw=2, c=colors[i], ls=lss[i], label=labels[i])
    axs[0].axvline(x=0, c='k', lw=1)
    # axs[0].legend(fontsize=100)

    return fig, axs

def load_fc_on_r_data_old(inputfilelist, r=0.2):
    frics = []
    for inputfile in inputfilelist:
        with open(inputfile, "rb") as f:
            fric = pickle.load(f)
            frics.append(fric)
    xyrange = 0.3
    N = fric.shape[0]
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.RECTANGLE_FIG)
    fig, axs = plotter.myfig(slide=True, xsigf=0, ysigf=2, xlabel="angle of cage position [degree]", ylabel="circumferential component of the friction force", xrange=(0, 360), yrange=(0, 0.1))
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    for i in range(len(inputfilelist)):
        axs[0].plot(np.degrees(t), np.linalg.norm(frics[i], axis=-1), lw=1, c=colors[i])
    # axs[0].set(xlim=(-xyrange, xyrange), ylim=(-xyrange, xyrange))
    # axs[0].set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    # axs[0].set(xlim=(-0.2, 0.2), ylim=(-0.2, 0.2))
        axs[0].axvline(x=0, c='k', lw=1)
    return fig, axs

if __name__ == "__main__":
    print("---- run ----")
    # vwidth = 0.004
    # vscale = 0.002
    # xyz = np.array([0, 0.3, 0])
    # theta = np.arcsin((6.25-5.953) / 53.3)
    # theta = 0
    # xyz = -xyz
    # theta = -theta
    # plane = "yz"
    # print(f"theta: {np.degrees(theta):.2f}")
    # cage = SimpleCage()
    # cage.translate((0, 0, 10))
    # cage.rotatex(20)
    # cage.translate((0, 0, 10))
    # cage.visualize(plane="yz")
    # ball = Ball()
    # ball.tilt(xyz, theta)
    # ball.visualize(plane="zx")
    # aring = ARing()
    # aring = None
    # aring.visualize()
    # bring = BRing()
    # bring = None
    # bring.visualize()
    # bearing = Bearing(aring, bring, ball, cage)
    # ball.translate(xyz)
    # bearing.calc_contact_ball2cylinder()
    # bearing.calc_contact_force()
    # bearing.visualize(plane=plane, vwidth=vwidth, vscale=vscale)

    mp.set_start_method("spawn") # for linux
    outdir = config.ROOT / "results" / "scalar_field"
    outdir.mkdir(parents=True, exist_ok=True)
    # calc_scalar_field_parallel(N=21, xyrange=0.5, theta=np.radians(0), name="scalar_theta0_xy05_N21")
    # calc_scalar_field_parallel(N=301, xyrange=0.3, theta=np.radians(0.33), name="theta033_xy03_N301")

    onrlist = [
        # {"r": 0.15, "N": 400, "theta": 0, "num_balls": 4},
        # {"r": 0.15, "N": 400, "theta": 0.33, "num_balls": 4},
        # {"r": 0.16, "N": 400, "theta": 0, "num_balls": 4},
        # {"r": 0.16, "N": 400, "theta": 0.33, "num_balls": 4},
        # {"r": 0.17, "N": 400, "theta": 0, "num_balls": 4},
        # {"r": 0.17, "N": 400, "theta": 0.33, "num_balls": 4},
        # {"r": 0.15, "N": 400, "theta": 0, "num_balls": 8},
        # {"r": 0.15, "N": 400, "theta": 0.33, "num_balls": 8},
        # {"r": 0.16, "N": 400, "theta": 0, "num_balls": 8},
        # {"r": 0.16, "N": 400, "theta": 0.33, "num_balls": 8},
        # {"r": 0.17, "N": 400, "theta": 0, "num_balls": 8},
        # {"r": 0.17, "N": 400, "theta": 0.33, "num_balls": 8},
        # {"r": 0.18, "N": 400, "theta": 0, "num_balls": 8},
        # {"r": 0.18, "N": 400, "theta": 0.33, "num_balls": 8},
        # {"r": 0.19, "N": 400, "theta": 0, "num_balls": 8},
        # {"r": 0.19, "N": 400, "theta": 0.33, "num_balls": 8},
        # {"r": 0.2, "N": 400, "theta": 0, "num_balls": 8},
        # {"r": 0.2, "N": 400, "theta": 0.33, "num_balls": 8},
        # {"r": 0.14, "N": 400, "theta": 0, "num_balls": 22},
        # {"r": 0.14, "N": 400, "theta": 0.33, "num_balls": 22},
        # {"r": 0.15, "N": 400, "theta": 0, "num_balls": 22},
        # {"r": 0.15, "N": 400, "theta": 0.33, "num_balls": 22},
        # {"r": 0.16, "N": 400, "theta": 0, "num_balls": 22},
        # {"r": 0.16, "N": 400, "theta": 0.33, "num_balls": 22},
        # {"r": 0.17, "N": 400, "theta": 0, "num_balls": 22},
        # {"r": 0.17, "N": 400, "theta": 0.33, "num_balls": 22},
    ]
    for args in onrlist:
        r = args["r"]
        N = args["N"]
        theta = args["theta"]
        num_balls = args["num_balls"]
        _r = '0' + str(int(r*100)) if r > 0 else 0
        _theta = '0' + str(int(theta*100)) if theta > 0 else 0
        name = f"force_on_r{_r}_theta{_theta}_Z{num_balls}_N{N}"
        # calc_force_on_r_parallel(r=r, N=N, theta=np.radians(theta), num_balls=num_balls, name=name, log_freq=10, ncpu=2)

    filenamelist = ["scalar_theta0_xy03_N301_Z8", "scalar_theta033_xy03_N301_Z8"]
    fig, axs = [], []
    for fn in filenamelist:
        _fig, _axs = load_scalar_field_data(outdir/fn, step=1, vscale=100000, vwidth=0.001, valpha=0.4, xyrange=0.3, vmin=0, vmax=8000, cmap="viridis", mode="scalar")
        fig.append(_fig)
        axs.append(_axs)

    # jobs = [
    #     {"jobname": "force_on_r014_theta0_Z22_N400", "ls": "--", "color": 'b', "label": None},
    #     {"jobname": "force_on_r014_theta033_Z22_N400", "ls": "-", "color": 'b', "label": None},
    #     {"jobname": "force_on_r015_theta0_Z22_N400", "ls": "--", "color": 'r', "label": None},
    #     {"jobname": "force_on_r015_theta033_Z22_N400", "ls": "-", "color": 'r', "label": None},
    #     {"jobname": "force_on_r016_theta0_Z22_N400", "ls": "--", "color": 'g', "label": None},
    #     {"jobname": "force_on_r016_theta033_Z22_N400", "ls": "-", "color": 'g', "label": None},
    #     ]
    # fig, axs = load_fc_on_r_data(jobs, datadir=config.ROOT/"results"/"scalar_field")

    plt.show()


