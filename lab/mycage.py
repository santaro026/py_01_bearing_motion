# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 10:22:23 2025
@author: santaro

useful functions for cage

Modification
- delete the legacy function to make cage sample data, because the more utilized class is implemented.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import polars as pl

from matplotlib.animation import ArtistAnimation
# import matplotlib
# matplotlib.use('Qt5Agg')
# matplotlib.use('TkAgg')

import config

import sys
from pathlib import Path
import time

from mymods import mycoord, myplotter


class SimpleCage:
    def __init__(self, name='', PCD=50, ID=48, OD=52, width=10, num_pockets=8, num_markers=8, num_mesh=99, Dp=6.25, Dw=5.953):
        self.name = name
        self.PCD = PCD
        self.ID = ID
        self.OD = OD
        self.width = width
        self.num_pockets = num_pockets
        self.num_markers = num_markers
        self.num_mesh = num_mesh
        self.num_nodes = num_mesh + 1
        self.Dp = Dp
        self.Dw = Dw
        self.dp = Dp - Dw
        self.pockets_pos = np.linspace(0, 2*np.pi, self.num_pockets, endpoint=False) + np.pi/2
        self.markers_pos = np.linspace(0, 2*np.pi, self.num_markers, endpoint=False) + np.pi/2
        self.nodes_pos = np.linspace(0, 2*np.pi, self.num_nodes, endpoint=True) + np.pi/2
        self.cage_zero = np.zeros((1, 3))
        markers_zero, _ = SimpleCage.make_pockets(np.zeros((1, 6)), self.num_pockets, 0, a=self.PCD/2, b=self.PCD/2, p0_angle=np.pi/6)
        self.markers_zero = markers_zero[:, :, :3]
    @staticmethod
    def omega2p(omega, r, dt, num_frames=None):
        omega = np.full(num_frames, omega) if np.ndim(omega) == 0 else omega
        angle_rev = np.cumsum(np.hstack([0, omega[1:]])) * dt
        y = r * np.cos(angle_rev + np.pi/2)
        z = r * np.sin(angle_rev + np.pi/2)
        x = np.zeros_like(y)
        return np.vstack([x, y, z]).T
    @staticmethod
    def p2omega(p, dt):
        angle = np.unwrap(np.arctan2(p[:, 2], p[:, 1]))
        omega = np.gradient(angle, dt)
        return omega
    @staticmethod
    def make_pockets(cage, num_pockets, x_value, a, b, deform_angle_local=0, p0_angle=np.pi/2, endpoint=False):
        num_frames = cage.shape[0]
        pockets_local = np.full((num_frames, num_pockets, 6), np.nan) # pockets of pockets on local coordinate system
        pockets_global = np.full((num_frames, num_pockets, 6), np.nan)
        pockets_pos = np.linspace(0, 2*np.pi, num_pockets, endpoint=endpoint) + p0_angle
        for i in range(num_pockets):
            _x = np.full(num_frames, x_value)
            _theta = np.full(num_frames, pockets_pos[i]) - deform_angle_local
            _y = a * np.cos(_theta) * np.cos(deform_angle_local) - b * np.sin(_theta) * np.sin(deform_angle_local)
            _z = a * np.cos(_theta) * np.sin(deform_angle_local) + b * np.sin(_theta) * np.cos(deform_angle_local)
            _Rx = np.full(num_frames, pockets_pos[i])
            _Ry = np.zeros(num_frames)
            _Rz = np.zeros(num_frames)
            pockets_local[:, i, :] = np.vstack([_x, _y, _z, _Rx, _Ry, _Rz]).T
            transformer = mycoord.CoordTransformer3d(local_origin=cage[:, :3], euler_angles=cage[:, 3:6], rot_order="zyx")
            pockets_global[:, i, :3] = transformer.transform_coord(pockets_local[:, i, :3], towhich='toglobal')
            pockets_global[:, :, 3:6] = pockets_local[:, :, 3:6] + cage[:, np.newaxis, 3:6]
        return pockets_global, pockets_local

    def time_series_data(self, sc="", fps=10000, cage_p=np.vstack([np.zeros(10001), 0.4*np.cos(np.linspace(0, 2*np.pi*40, 10001)), 0.4*np.sin(np.linspace(0, 2*np.pi*40, 10001))]).T, cage_R=np.vstack([np.linspace(0, 2*np.pi*40, 10001), np.zeros(10001), np.zeros(10001)]).T,  a=1, b=1, p0_angle=np.pi/2, deform_angle_local=0, noise_type="normal", noise_max=1):
        self.sc = sc
        self.fps = fps
        self.num_frames = cage_p.shape[0]
        self.duration = (self.num_frames - 1) / self.fps
        self.t = np.linspace(0, self.duration, self.num_frames, endpoint=True)
        self.dt = 1 / self.fps
        self.cage_p = cage_p
        self.cage_R = cage_R
        self.cage = np.concatenate([self.cage_p, self.cage_R], axis=-1)
        self.omega_rot = np.gradient(self.cage_R[:, 0], self.dt)
        self.angle_rev = np.unwrap(np.arctan2(self.cage_p[:, 2], self.cage_p[:, 1]))
        self.omega_rev = np.gradient(self.angle_rev, self.dt)
        self.r_rev = np.linalg.norm(self.cage_p[:, 1:], axis=1)
        self.omega_rot_avg = np.nanmean(self.omega_rot)
        self.rpm_avg = np.nanmean(self.omega_rot) / (2*np.pi) * 60
        #### define coordinate transformer
        self.euler_angles_avg = np.vstack([self.omega_rot_avg * self.t, np.zeros(self.num_frames), np.zeros(self.num_frames)]).T # euler angle for rotation frame with constant velocity
        self.transformerSI = mycoord.CoordTransformer3d(name="system_instantaneous", local_origin=np.zeros((self.num_frames, 3)), euler_angles=self.cage_R, rot_order='zyx')
        self.transformerSA = mycoord.CoordTransformer3d(name="system_average", local_origin=np.zeros((self.num_frames, 3)), euler_angles=self.euler_angles_avg, rot_order="zyx")
        self.transformerCI= mycoord.CoordTransformer3d(name="cage_instantaneous", local_origin=self.cage_p, euler_angles=self.cage_R, rot_order='zyx')
        self.transformerCA = mycoord.CoordTransformer3d(name="cage_average", local_origin=self.cage_p, euler_angles=self.euler_angles_avg, rot_order="zyx")
        #### generate points on cage
        self.pockets, self.pockets_local = SimpleCage.make_pockets(self.cage, num_pockets=self.num_pockets, x_value=0, a=a*self.PCD/2, b=b*self.PCD/2, p0_angle=p0_angle, deform_angle_local=deform_angle_local)
        self.markers, self.markers_local = SimpleCage.make_pockets(self.cage, num_pockets=self.num_markers, x_value=0, a=a*self.PCD/2, b=b*self.PCD/2, p0_angle=p0_angle, deform_angle_local=deform_angle_local)
        self.nodes, self.nodes_local = SimpleCage.make_pockets(self.cage, num_pockets=self.num_nodes, x_value=0, a=a*self.PCD/2, b=b*self.PCD/2, p0_angle=p0_angle, deform_angle_local=deform_angle_local, endpoint=True)
        #### add noise to the marker
        rng = np.random.default_rng(seed=0)
        if noise_type == 'uniform':
            noise = rng.uniform(-1, 1, (self.num_frames, self.num_markers, 3)) * noise_max
        elif noise_type == 'normal':
            noise = rng.normal(0, 1/3, (self.num_frames, self.num_markers, 3)) * noise_max
        self.markers_p_noise_local = np.full((self.num_frames, self.num_markers, 3), np.nan)
        self.markers_p_noise_local[:, :, :] = self.markers_local[:, :, :3] + noise
        self.markers_p_noise = np.full((self.num_frames, self.num_markers, 3), np.nan)
        for i in range(self.num_markers):
            self.markers_p_noise[:, i, :] = self.transformerCI.transform_coord(self.markers_p_noise_local[:, i, :], towhich='toglobal')

    def time_series_data2(self, sc="", fps=10000, duration=1, omega_rot=40*np.pi, omega_rev=40*np.pi, r_rev=0.4, initial_pos=np.pi/2, a=1, b=1, p0_angle=np.pi/2, omega_deform=0, initial_deform_dir=np.pi/2, noise_type="normal", noise_max=1):
        num_frames = int(fps * duration) + 1
        t = np.linspace(0, duration , num_frames)
        dt = 1 / fps
        x = np.zeros(num_frames)
        omega_deform_local = omega_deform - omega_rot
        if a != 1 and b != 1:
            d = (1 - min(a, b)) * self.PCD/2
        else:
            d = 0
        y_deform = d * np.cos(omega_deform*t + initial_deform_dir)
        z_deform = d * np.sin(omega_deform*t + initial_deform_dir)
        y = r_rev * np.cos(omega_rev*t + initial_pos) + y_deform
        z = r_rev * np.sin(omega_rev*t + initial_pos) + z_deform
        cage_p = np.vstack([x, y, z]).T
        Rx = np.cumsum(np.hstack([0, np.full(num_frames-1, omega_rot)])) * dt
        Ry = np.zeros(num_frames)
        Rz = np.zeros(num_frames)
        cage_R = np.vstack([Rx, Ry, Rz]).T
        deform_angle_local = omega_deform_local * t
        self.time_series_data(sc=sc, fps=fps, cage_p=cage_p, cage_R=cage_R, a=a, b=b, p0_angle=p0_angle, deform_angle_local=deform_angle_local, noise_type=noise_type, noise_max=noise_max)

    def export_with_TEMAformat(self, outdir=None):
        #### export with visualization test format
        if outdir is None: outdir = config.ROOT / 'data'
        outdir.mkdir(exist_ok=True, parents=True)
        outfname = f'{self.name}_{self.sc}_{int(self.rpm_avg)}rpm_{int(self.fps)}fps'
        header_markers = ['time']
        for i in range(self.num_markers):
            for j in ['y', 'z']:
                header_markers.append(f'p{i}{j}')
        t = self.t[:, np.newaxis]
        markers = self.markers[:, :, 1:3].reshape(self.num_frames, -1)
        data_markers = np.concatenate([t, markers], axis=-1)
        df_markers = pl.from_numpy(data_markers, schema=header_markers)
        markers_noise = self.markers_p_noise[:, :, 1:3].reshape(self.num_frames, -1)
        data_markers_noise = np.concatenate([t, markers_noise], axis=-1)
        df_markers_noise = pl.from_numpy(data_markers_noise, schema=header_markers)
        markers_zero = self.markers_zero.reshape(self.num_markers, 3)
        markers_zeroy = np.concatenate([markers_zero[:, 1], np.zeros(1)], axis=0).T
        markers_zeroz = np.concatenate([markers_zero[:, 2], np.zeros(1)], axis=0).T
        zero_data = []
        _markers_name = []
        for i in range(self.num_markers+1):
            _markers_name.append(f'point#{i}')
        zero_data.append(_markers_name)
        zero_data.append(markers_zeroy)
        zero_data.append(markers_zeroz)
        _bring = []
        for i in range(self.num_markers):
            _bring.append('-')
        _bring.append(1)
        zero_data.append(_bring)
        zero_data = np.array(zero_data).T
        df_zero_data = pl.from_numpy(zero_data, schema=["item", "y", "z", "area"])
        df_markers.write_csv(outdir / f"{outfname}_markers.csv")
        df_markers_noise.write_csv(outdir / f"{outfname}_markers_noise.csv")
        df_zero_data.write_csv(outdir / f"{self.name}_zero.csv")

    def export_time_series_data(self, outdir=None):
        #### export primary data
        if outdir is None: outdir = Path(__file__).resolve().parent / 'data'
        if outfname == None: outfname = f'{self.name}_sc{self.shooting_code}_{self.rpm_avg}rpm_{self.fps}fps'
        cage = np.vstack([np.arange(self.num_frames), # 0
                        self.t, # 1
                        self.x, # 2
                        self.y, # 3
                        self.z, # 4
                        self.Rx, # 5
                        self.Ry, # 6
                        self.Rz, # 7
                        ])
        header_cage = ['frame', 'time [sec]', 'x', 'y', 'z', 'Rx [rad]', 'Ry [rad]', 'Rz [rad]']
        mytools.save_csv(cage, header=header_cage, outfname=f'{outfname}_cage.csv')
        header_markers = ['frame', 'time [sec]']
        for i in range(self.num_markers):
            for j in list(['x', 'y']):
                header_markers.append(f'p{i}{j}')
        markers = np.vstack([np.arange(self.num_frames), self.t, self.markers_p]).T
        mytools.save_csv(markers, header=header_markers, outfname=f'{outfname}_markers.csv')
        markers_noise = np.vstack([np.arange(self.num_frames), self.t, self.markers_p_noise]).T
        mytools.save_csv(markers_noise, header=header_markers, outfname=f'{outfname}_markers_noise.csv')
        header_pockets = ['frame', 'time [sec]']
        for i in range(self.num_markers):
            for j in list(['x', 'y']):
                header_pockets.append(f'p{i}_{j}')
        pockets = np.vstack([np.arange(self.num_frames), self.t, self.pockets_p]).T
        mytools.save_csv(pockets, header=header_pockets, outfname=f'{outfname}_pockets.csv')

if __name__ == '__main__':
    print('---- test ----')

    st = time.perf_counter()

    import specification_loader
    spec_loader = specification_loader.SpecificationLoader()
    sample_code = specification_loader.SampleCode
    spec = spec_loader.specification_factory(sample_code.SIMPLE50)
    print(spec)

    cage = SimpleCage(name=spec.name, PCD=spec.PCD, ID=spec.ID, OD=spec.OD, width=spec.width, num_pockets=spec.num_pockets, num_markers=spec.num_markers, num_mesh=100, Dp=spec.Dp, Dw=spec.Dw)

    skip = 2
    interval = 10
    import time_param_loader
    param_loader = time_param_loader.TimeParamLoader()
    MotinoCode = time_param_loader.MotionCode
    # param = param_loader.param_factory(MotinoCode.ROT_REV)
    # param = param_loader.param_factory(MotinoCode.ROT_REV_ELLIPSE)
    # param = param_loader.param_factory(MotinoCode.ROT10_REV)
    # param = param_loader.param_factory(MotinoCode.ROT_GRAVITY)
    # param = param_loader.param_factory(MotinoCode.ROT_REV10)
    # param = param_loader.param_factory(MotinoCode.ROT_REV5_ELLIPSE5)
    # param = param_loader.param_factory(MotinoCode.ROT_REV10_ELLIPSE10)

    enums = list(time_param_loader.MotionCode)

    for enu in enums:
        param = param_loader.param_factory(enu)
        cage.time_series_data2(sc=f"{param.name}", fps=param.fps, duration=param.duration, omega_rot=param.omega_rot, omega_rev=param.omega_rev, r_rev=param.r_rev, a=param.a, b=param.b, omega_deform=param.omega_deform, noise_type=param.noise_type, noise_max=param.noise_max, initial_pos=param.initial_pos)
        outdir = config.ROOT / "sampledata" / f"{cage.name}"
        print(f"enum: {param.name}")
        cage.export_with_TEMAformat(outdir=outdir)

    et = time.perf_counter()
    print(f'elapsed time: {round((et-st)/1000, 4)} [sec]')

    plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.TRAJECTORY_WITH_TIMESERIES)
    fig, axs = plotter.myfig(xrange=[(-30, 30), (-30, 30)], yrange=[(-30, 30), (-30, 30)])
    # plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.TRAJECTORY)
    # fig, axs = plotter.myfig(xrange=(-40, 40), yrange=(-40, 40))

    cage_center_scale = 1
    cage_center = cage.cage[:, :3] * cage_center_scale
    cage_center_local = cage.transformerSI.transform_point(cage.cage[:, :3], towhich="tolocal")
    data_list = [
        {"id": 0, "data": [cage_center[:, 1], cage_center[:, 2]], "color": 'k', "markersize": 10, "malpha": 1, "lw": 1, "lalpha": 0.4, "disp_max": 100},
        {"id": 1, "data": [cage_center_local[:, 1], cage_center_local[:, 2]], "color": 'k', "markersize": 10, "malpha": 1, "lw": 1, "lalpha": 0.4, "disp_max": 100},
    ]

    _c0 = {"id": 0, "data": [cage.pockets[:, 0, 1], cage.pockets[:, 0, 2]], "color": 'r', "markersize": 400, "malpha": 1, "lw": 1, "lalpha": 0.4, "disp_max": 0}
    _c1 = {"id": 0, "data": [cage.pockets[:, 1, 1], cage.pockets[:, 1, 2]], "color": 'b', "markersize": 400, "malpha": 1, "lw": 1, "lalpha": 0.4, "disp_max": 0}
    data_list.append(_c0)
    data_list.append(_c1)
    for i in range(cage.num_pockets):
        _c2 = {"id": 0, "data": [cage.pockets[:, i, 1], cage.pockets[:, i, 2]], "color": 'k', "markersize": 400, "malpha": 0.1, "lw": 1, "lalpha": 0.4, "disp_max": 0}
        data_list.append(_c2)
    pockets_local = np.full((cage.num_frames, cage.num_pockets, 3), np.nan)
    for i in range(cage.num_pockets):
        pockets_local[:, i, :] = cage.transformerSI.transform_coord(cage.pockets[:, i, :3], towhich="tolocal")
        _c2 = {"id": 1, "data": [pockets_local[:, i, 1], pockets_local[:, i, 2]], "color": 'k', "markersize": 400, "malpha": 0.1, "lw": 1, "lalpha": 0.4, "disp_max": 0}
        data_list.append(_c2)
    _c0 = {"id": 1, "data": [pockets_local[:, 0, 1], pockets_local[:, 0, 2]], "color": 'r', "markersize": 400, "malpha": 1, "lw": 1, "lalpha": 0.4, "disp_max": 0}
    _c1 = {"id": 1, "data": [pockets_local[:, 1, 1], pockets_local[:, 1, 2]], "color": 'b', "markersize": 400, "malpha": 1, "lw": 1, "lalpha": 0.4, "disp_max": 0}
    data_list.append(_c0)
    data_list.append(_c1)

    cage_ori = mycoord.CoordTransformer3d.get_basic_vector(cage.cage[:, 3:], rot_order="zyx")
    cage_ori_avg = mycoord.CoordTransformer3d.get_basic_vector(cage.euler_angles_avg, rot_order="zyx")
    cage_local = np.full((cage.num_frames, 6), np.nan)
    cage_local[:, :3] = cage.transformerCI.transform_point(cage.cage[:, :3], towhich="tolocal")
    cage_local[:, 3:6] = cage.transformerCI.transform_orientation(cage.cage[:, 3:6], towhich="tolocal")
    cage_ori_local = mycoord.CoordTransformer3d.get_basic_vector(cage_local, rot_order="zyx")
    vct_list = [
        #### system basic vector
        {"id": 0, "mode": "force", "data": [cage_center[:, 1], cage_center[:, 2], cage_ori[:, 1, 1], cage_ori[:, 1, 2]], "width": 0.005, "scale": 0.2, "color": 'g', "alpha": 1},
        {"id": 0, "mode": "force", "data": [cage_center[:, 1], cage_center[:, 2], cage_ori[:, 2, 1], cage_ori[:, 2, 2]], "width": 0.005, "scale": 0.2, "color": 'b', "alpha": 1},
        {"id": 0, "mode": "force", "data": [np.zeros(cage.num_frames), np.zeros(cage.num_frames), cage_ori[:, 1, 1], cage_ori[:, 1, 2]], "width": 0.005, "scale": 0.2, "color": 'g', "alpha": 1},
        {"id": 0, "mode": "force", "data": [np.zeros(cage.num_frames), np.zeros(cage.num_frames), cage_ori[:, 2, 1], cage_ori[:, 2, 2]], "width": 0.005, "scale": 0.2, "color": 'b', "alpha": 1},
        #### cage basic vector
        {"id": 1, "mode": "force", "data": [cage_center_local[:, 1], cage_center_local[:, 2], cage_ori_local[:, 1, 1], cage_ori_local[:, 1, 2]], "width": 0.005, "scale": 0.2, "color": 'g', "alpha": 1},
        {"id": 1, "mode": "force", "data": [cage_center_local[:, 1], cage_center_local[:, 2], cage_ori_local[:, 2, 1], cage_ori_local[:, 2, 2]], "width": 0.005, "scale": 0.2, "color": 'b', "alpha": 1},
    ]
    pockets_local = np.full((cage.num_frames, cage.num_pockets, 6), np.nan)
    for i in range(cage.num_pockets):
        _ori = mycoord.CoordTransformer3d.get_basic_vector(cage.pockets[:, i, 3:], rot_order="zyx")
        _vy = {"id": 0, "mode": "force", "data": [cage.pockets[:, i, 1], cage.pockets[:, i, 2], _ori[:, 1, 1], _ori[:, 1, 2]], "width": 0.002, "scale": 0.5, "color": 'g', "alpha": 1}
        _vz = {"id": 0, "mode": "force", "data": [cage.pockets[:, i, 1], cage.pockets[:, i, 2], _ori[:, 2, 1], _ori[:, 2, 2]], "width": 0.002, "scale": 0.5, "color": 'b', "alpha": 1}
        vct_list.append(_vy)
        vct_list.append(_vz)
        pockets_local[:, i, :3] = cage.transformerSI.transform_point(cage.pockets[:, i, :3], towhich="tolocal")
        pockets_local[:, i, 3:6] = cage.transformerSI.transform_orientation(cage.pockets[:, i, 3:6], towhich="tolocal")
        _ori_local = mycoord.CoordTransformer3d.get_basic_vector(cage.pockets_local[:, i, 3:], rot_order="zyx")
        _vy = {"id": 1, "mode": "force", "data": [pockets_local[:, i, 1], pockets_local[:, i, 2], _ori_local[:, 1, 1], _ori_local[:, 1, 2]], "width": 0.002, "scale": 0.5, "color": 'g', "alpha": 1}
        _vz = {"id": 1, "mode": "force", "data": [pockets_local[:, i, 1], pockets_local[:, i, 2], _ori_local[:, 2, 1], _ori_local[:, 2, 2]], "width": 0.002, "scale": 0.5, "color": 'b', "alpha": 1}
        vct_list.append(_vy)
        vct_list.append(_vz)
    nodes_local = np.full((cage.num_frames, cage.num_nodes, 3), np.nan)
    for i in range(cage.num_nodes):
        nodes_local[:, i, :] = cage.transformerSI.transform_coord(cage.nodes[:, i, :3], towhich="tolocal")

    def get_ellipse_axline(points, major_length=100, minor_length=10):
        ellip_vct = points[:, :cage.num_nodes//2+1, :3] - points[:, cage.num_nodes//2:, :3]
        ellip_norm = np.linalg.norm(ellip_vct, axis=2)
        ellip_vct = ellip_vct / ellip_norm[:, :, np.newaxis]
        major_idx = np.argmax(ellip_norm, axis=1)
        minor_idx = np.argmin(ellip_norm, axis=1)
        major_vct = np.take_along_axis(ellip_vct, major_idx[:, None, None], axis=1).squeeze(1)
        minor_vct = np.take_along_axis(ellip_vct, minor_idx[:, None, None], axis=1).squeeze(1)
        major_line = np.stack([-major_length * major_vct, major_length * major_vct], axis=1)
        minor_line = np.stack([-minor_length * minor_vct, minor_length * minor_vct], axis=1)
        return major_line, minor_line

    major_line, minor_line = get_ellipse_axline(cage.nodes)
    major_line_local, minor_line_local = get_ellipse_axline(nodes_local)
    fline_list = [
        {"id": 0, "data": [cage.nodes[:, :, 1], cage.nodes[:, :, 2]], "color": 'orange', "alpha": 0.4, "lw": 10},
        {"id": 0, "data": [major_line[:, :, 1], major_line[:, :, 2]], "color": 'k', "alpha": 1, "lw": 1},
        {"id": 0, "data": [minor_line[:, :, 1], minor_line[:, :, 2]], "color": 'k', "alpha": 1, "lw": 1},
        {"id": 1, "data": [nodes_local[:, :, 1], nodes_local[:, :, 2]], "color": 'orange', "alpha": 0.4, "lw": 10},
        {"id": 1, "data": [major_line_local[:, :, 1], major_line_local[:, :, 2]], "color": 'k', "alpha": 1, "lw": 1},
        {"id": 1, "data": [minor_line_local[:, :, 1], minor_line_local[:, :, 2]], "color": 'k', "alpha": 1, "lw": 1},
    ]

    # axs[2].plot(cage.t, np.degrees(cage.cage_R[:, 0]), lw=1)
    # axs[2].plot(cage.t, np.degrees(cage.cage_R[:, 1]), lw=1)
    # axs[2].plot(cage.t, np.degrees(cage.cage_R[:, 2]), lw=1)
    # axs[2].set(xlim=(0, 1), ylim=(-10, np.degrees(np.max(cage.cage_R[:, 0]))))

    for i in range(2):
        axs[i].axvline(x=0, lw=0.5, c='k')
        axs[i].axhline(y=0, lw=0.5, c='k')
        circle = patches.Circle((0, 0), 27, fill=False, edgecolor='k', ls="--")
        axs[i].add_patch(circle)
        circle = patches.Circle((0, 0), 2, fill=False, edgecolor='r', ls="--")
        axs[i].add_patch(circle)
    axs[0].set_aspect(1)
    axs[1].set_aspect(1)
    axs[3].axis("off")
    # animator = myplotter.MyAnimator(fig, axs, data_list=data_list, vct_list=vct_list, fline_list=fline_list)
    # ani = animator.make_func_ani(skip=skip, interval=interval)
    # plt.show()



