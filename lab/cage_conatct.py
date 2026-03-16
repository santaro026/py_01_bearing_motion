"""
Created on Sun Feb 08 17:04:10 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.path import Path as mPath

from pathlib import Path

from sympy import Point, Circle, Line, Segment, sqrt
from sympy import Point3D, Line3D, Plane
from sympy import Matrix, sin, cos, tan

from santamods import mycoord, myplotter

def make_cage_points(num_frames, num_points, x_value, a, b, deform_angle, transformer, p0_angle=np.pi/2, endpoint=False):
    p_lcs = np.full((num_frames, num_points, 3), np.nan) # points of pockets on local coordinate system
    p_global = np.full((num_frames, num_points, 3), np.nan)
    pos_points = np.linspace(0, 2*np.pi, num_points, endpoint=endpoint) + p0_angle
    if deform_angle != 0:
        rot_euler_for_ellipse = np.vstack([deform_angle, np.zeros(num_frames), np.zeros(num_frames)]).T
    for i in range(num_points):
        _x = np.full(num_frames, x_value)
        _theta = np.full(num_frames, pos_points[i])
        _y = a * np.cos(_theta)
        _z = b * np.sin(_theta)
        p_lcs[:, i, :] = np.vstack([_x, _y, _z]).T
        if deform_angle != 0:
            p_lcs[:, i, :] = mycoord.CoordTransformer3D.rotate_euler(p_lcs[:, i, :], euler_angles=rot_euler_for_ellipse, rot_order="zyx")
        p_global[:, i, :] = transformer.transform_coord(p_lcs[:, i, :], towhich='toglobal')
    return p_global, p_lcs

# transformer = mycoord.CoordTransformer2D(coordsys_name="cage_coordsys", local_origin=np.zeros(2), theta=self.pos_pockets)

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
        self.pos_balls = np.linspace(0, 2*np.pi, self.num_balls, endpoint=False) + np.pi/2
        self.ball_circles = []
        self.ball_centers = []
        for i in range(self.num_balls):
            _b = Circle((0, 0), Dw/2).translate(self.PCD/2, 0).rotate(self.pos_balls[i], Point(0, 0))
            self.ball_circles.append(_b)
            _c = Point(0, 0).translate(self.PCD/2, 0).rotate(self.pos_balls[i], Point(0, 0))
            self.ball_centers.append(_c)
    def translate(self, xy):
        for i in range(self.num_balls):
            self.ball_circles[i][0] =  self.ball_circles[i][0].translate(xy[0], xy[1])
            self.ball_centers[i] = self.ball_centers[i].translate(xy[0], xy[1])
        for i in range(2):
            self.ring_outcircles[i] = self.ring_outcircles[i].translate(xy[0], xy[1])
    def rotate(self, theta):
        for i in range(self.num_balls):
            self.ball_circles[i][0] = self.ball_circles[i][0].rotate(theta)
            self.ball_centers[i] = self.ball_centers[i].rotate(theta)
    def visualize(self):
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        for i in range(self.num_balls):
            _b = patches.Circle(self.ball_circles[i].center, self.ball_circles[i].radius, color='k', fill=False)
            axs[0].add_patch(_b)
        axs[0].set(xlim=(-30, 30), ylim=(-30, 30))
        plt.show()

class SimpleCage:
    def __init__(self, name="", PCD=50, ID=48, OD=52, width=10, num_pockets=8, Dp=6.25):
        self.name = name
        self.PCD = PCD
        self.ID = ID
        self.OD = OD
        self.ring_outlines = [Circle((0, 0), self.ID/2), Circle((0, 0), self.OD/2)]
        self.width = width
        self.num_pockets = num_pockets
        self.Dp = Dp
        self.pos_pockets = np.linspace(0, 2*np.pi, self.num_pockets, endpoint=False) + np.pi/2
        _theta_id = np.arcsin(self.Dp / self.ID)
        _theta_od = np.arcsin(self.Dp / self.OD)
        l0x = -(self.PCD - self.ID * np.cos(_theta_id))/2
        l1x = (self.OD * np.cos(_theta_od) - self.PCD)/2
        l0y = self.Dp/2
        l1y = -self.Dp/2
        pocket_l0 = Line(Point(l0x, l0y), Point(l1x, l0y))
        pocket_l1 = Line(Point(l0x, l1y), Point(l1x, l1y))
        self.pockets_lines = []
        self.pockets_centers = []
        for i in range(self.num_pockets):
            _l0 = pocket_l0.translate(self.PCD/2, 0).rotate(self.pos_pockets[i], Point(0, 0))
            _l1 = pocket_l1.translate(self.PCD/2, 0).rotate(self.pos_pockets[i], Point(0, 0))
            _ls = [_l0, _l1]
            self.pockets_lines.append(_ls)
            _c = Point(0, 0).translate(self.PCD/2, 0).rotate(self.pos_pockets[i], Point(0, 0))
            self.pockets_centers.append(_c)
    def translate(self, xy):
        for i in range(self.num_pockets):
            self.pockets_lines[i][0] =  self.pockets_lines[i][0].translate(xy[0], xy[1])
            self.pockets_lines[i][1] = self.pockets_lines[i][1].translate(xy[0], xy[1])
            self.pockets_centers[i] = self.pockets_centers[i].translate(xy[0], xy[1])
        for i in range(2):
            self.ring_outlines[i] = self.ring_outlines[i].translate(xy[0], xy[1])
    def rotate(self, theta):
        for i in range(self.num_pockets):
            self.pockets_lines[i][0] = self.pockets_lines[i][0].rotate(theta)
            self.pockets_lines[i][1] = self.pockets_lines[i][1].rotate(theta)
            self.pockets_centers[i] = self.pockets_centers[i].rotate(theta)
        for i in range(2):
            self.ring_outlines[i] = self.ring_outlines[i].rotate(theta)
    def visualize(self):
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        for i in range(self.num_pockets):
            _p11 = np.array(self.pockets_lines[i][0].p1)
            _p12 = np.array(self.pockets_lines[i][0].p2)
            _p1 = np.column_stack([_p11, _p12])
            _p21 = np.array(self.pockets_lines[i][1].p1)
            _p22 = np.array(self.pockets_lines[i][1].p2)
            _p2 = np.column_stack([_p21, _p22])
            _l1 = mlines.Line2D(_p1[0], _p1[1], color='b')
            _l2 = mlines.Line2D(_p2[0], _p2[1], color='b')
            axs[0].add_line(_l1)
            axs[0].add_line(_l2)
            axs[0].scatter(self.pockets_centers[i].x, self.pockets_centers[i].y, s=10, c='b' )
        _id = patches.Circle(self.ring_outlines[0].center, self.ring_outlines[0].radius, color='k', fill=False)
        _od = patches.Circle(self.ring_outlines[1].center, self.ring_outlines[1].radius, color='k', fill=False)
        axs[0].add_patch(_id)
        axs[0].add_patch(_od)
        axs[0].set(xlim=(-30, 30), ylim=(-30, 30))
        plt.show()

class Bearing:
    def __init__(self, ARing, BRing, Ball, Cage):
        self.aring = ARing
        self.bring = BRing
        self.ball = Ball
        self.cage = Cage
    def calc_contact(self):
        self.distances = np.zeros((self.ball.num_balls, 2))
        self.n_vct = np.zeros((self.ball.num_balls, 2, 2))
        self.p_contact = np.zeros((self.ball.num_balls, 2, 2))
        for i in range(self.ball.num_balls):
            b = self.ball.ball_circles[i]
            l1 = self.cage.pockets_lines[i][0]
            l2 = self.cage.pockets_lines[i][1]
            self.distances[i] = np.array([l1.distance(b.center) - b.radius, l2.distance(b.center) - b.radius])
            self.n_vct[i] = np.array([l1.projection(b.center) - b.center, l2.projection(b.center) -b.center])
            self.n_vct[i] = self.n_vct[i] / np.linalg.norm(self.n_vct[i])
            self.p_contact[i] = np.array([l1.projection(b.center), l2.projection(b.center)])
    def calc_contact_force(self, k=1, d=1):
        self.forces = np.zeros((self.ball.num_balls, 2, 2))
        for i in range(self.ball.num_balls):
            if self.distances[i][0] <= 0:
                _f = -k * self.distances[i, 0] ** d
                self.forces[i, 0] = _f * self.n_vct[i, 0]
            if self.distances[i][1] <= 0:
                _f = -k * self.distances[i, 1] ** d
                self.forces[i, 1] = _f * self.n_vct[i, 1]
    def visualize(self):
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        if ARing:
            groove_a = patches.Circle((0, 0), self.aring.groove.radius, color='k', fill=False)
        if BRing:
            groove_b = patches.Circle((0, 0), self.bring.groove.radius, color='k', fill=False)
        for i in range(self.ball.num_balls):
            _b = patches.Circle(self.ball.ball_circles[i].center, self.ball.ball_circles[i].radius, color='k', fill=False)
            axs[0].add_patch(_b)
        for i in range(self.cage.num_pockets):
            _p11 = np.array(self.cage.pockets_lines[i][0].p1)
            _p12 = np.array(self.cage.pockets_lines[i][0].p2)
            _p1 = np.column_stack([_p11, _p12])
            _p21 = np.array(self.cage.pockets_lines[i][1].p1)
            _p22 = np.array(self.cage.pockets_lines[i][1].p2)
            _p2 = np.column_stack([_p21, _p22])
            _l1 = mlines.Line2D(_p1[0], _p1[1], color='b')
            _l2 = mlines.Line2D(_p2[0], _p2[1], color='b')
            axs[0].add_line(_l1)
            axs[0].add_line(_l2)
            axs[0].scatter(self.cage.pockets_centers[i].x, self.cage.pockets_centers[i].y, s=10, c='b' )
            if self.distances[i, 0] <= 0:
                axs[0].quiver(self.p_contact[i, 0, 0], self.p_contact[i, 0, 1], self.forces[i, 0, 0], self.forces[i, 0, 1], color='k', width=0.002)
            if self.distances[i, 1] <= 0:
                axs[0].quiver(self.p_contact[i, 1, 0], self.p_contact[i, 1, 1], self.forces[i, 1, 0], self.forces[i, 1, 1], color='k', width=0.002)
        axs[0].add_patch(groove_a)
        axs[0].add_patch(groove_b)
        _id = patches.Circle(self.cage.ring_outlines[0].center, self.cage.ring_outlines[0].radius, color='b', fill=False)
        _od = patches.Circle(self.cage.ring_outlines[1].center, self.cage.ring_outlines[1].radius, color='b', fill=False)
        axs[0].add_patch(_id)
        axs[0].add_patch(_od)
        axs[0].set(xlim=(-30, 30), ylim=(-30, 30))
        plt.show()


if __name__ == "__main__":
    print("---- run ----")

    cage = SimpleCage()
    cage.translate((0, 0.3))
    # cage.visualize()
    ball = Ball()
    # ball.visualize()
    aring = ARing()
    # aring.visualize()
    bring = BRing()
    # bring.visualize()

    bearing = Bearing(aring, bring, ball, cage)
    bearing.calc_contact()
    bearing.calc_contact_force()
    bearing.visualize()

