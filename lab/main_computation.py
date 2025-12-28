# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 08:37:04 2025
@author: honda-shin

Completed on Wed Jul 02 15:51:10 2025

Notes:
main script for analyzing the data of high speed camera

modification
- revise the expression "theta_rotframe = t * angular_velocity_avg + (initial_phase_avg - np.pi/2)" about display phase setting of markers

"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal

import re
from pathlib import Path
from datetime import datetime

from sintamods import myfft

from data_handler import DataHandler
import testcondition_loader
import data_processor
import plot_drawer
import progress_tracker

import helperfuncs
import config

def run_main_computation(inf_markers: Path, inf_zero: Path, inf_audio: Path, testcond: testcondition_loader.TestConditions, outdir=None, settings=None):
    tgtdir = inf_markers.parent
    outdir = outdir / tgtdir.name / inf_markers.stem if outdir is not None else config.ROOT / 'results' / 'output'
    logger = progress_tracker.Logger() # for logging

    calc_settings = {
        "xyrange_probmap": [-0.5, 0.5],
        "bins_probmap": 50,
        "Bring_area": np.pi * (49.1/2)**2, # calculated value based on the value in drawings
    }

    # it affects later parts that i shoud modify
    xyrange_probmap = calc_settings["xyrange_probmap"]
    bins = calc_settings["bins_probmap"]
    # Dp, dp, Dl, dl, note_Dpl = helperfuncs.calc_clearance(testcond)
    ##

    fpath = Path(__file__).resolve()
    logger.msg(f'main computation script: {fpath.relative_to(config.ROOT)}')
    logger.rectime(0, name='main')

    #### load the data
    logger.rectime(0, name='load_data')
    sc, rpm, fps, rec_markers = DataHandler.extract_info_from_filename(inf_markers)
    if inf_audio is not None:
        rec_audio = (_rec := re.search(r'REC(\d+)', inf_audio.name)) and int(_rec.group(1))
        if rec_markers != rec_audio:
            print(f"**** load rec files are not match, markers refer to {rec_markers}, audio refer to {rec_audio}.")
            return 0
    rpm_actual = rpm * testcond.mortar_prop[0] + testcond.mortar_prop[1]

    if testcond.test_code < 0:
        #### load test data
        p_zero, area_zero, Bcenter_zero, num_points_zero = helperfuncs.read_zero(inf_zero, skip_rows=1, num_index=1, sep=',')
        p_markers, num_points, num_frames = helperfuncs.read_markers(inf_markers, skip_rows=1, num_index=1, sep=',')
    else:
        #### load the high-speed camera data
        p_zero, area_zero, Bcenter_zero, num_points_zero = helperfuncs.read_zero(inf_zero, skip_rows=3, num_index=1, sep='\t')
        p_markers, num_points, num_frames = helperfuncs.read_markers(inf_markers, skip_rows=3, num_index=1, sep='\t')

    #### 9/1 13:50
    frame = np.arange(num_frames)
    t = np.arange(num_frames) / fps
    dt = t[1] -t[0]

    timedata_cage = data_processor.TimeSeriesDataProcessor(num_frames, fps)

    Bring_area = calc_settings["Bring_area"]
    scaling_factor_pxl2mm = np.sqrt(Bring_area / area_zero) if area_zero > 0 else 1
    p_zero *= scaling_factor_pxl2mm
    p_markers *= scaling_factor_pxl2mm
    #### load the audio data
    if inf_audio is not None:
        sound = DataHandler.read_matdata(inf_audio)
        inf_audio_name = inf_audio.name
        sound_shape = sound.shape
        num_samplings = len(sound)
        if testcond.test_code == 14:
            sampling_rate = 24000
        else:
            sampling_rate = 48000
        duration = num_samplings / sampling_rate
        t_sound = np.linspace(0, duration, num_samplings)
        okruns_sound, okruns_id_sound, ngruns_sound, ngruns_id_sound, sound_pw, sound_rms, threshold_sound = helperfuncs.detect_noise(sound, sampling_rate, window_time=0.01, threshold_factor=2, threshold=1.5)
        tranges_noisy_sound = np.array(okruns_sound) / sampling_rate if okruns_sound is not None else None
        # tranges_silent_sound = np.array(ngruns_sound) / sampling_rate  if ngruns_sound is not None else None
    elif inf_audio is None:
        inf_audio_name = None
        sound_shape = None
        num_samplings = None
        sampling_rate = None
        duration = None
        t_sound = t
        sound = np.full_like(t, 0)
        okruns_sound, okruns_id_sound, ngruns_sound, ngruns_id_sound, sound_pw, sound_rms, threshold_sound = helperfuncs.detect_noise(sound, fps, window_time=0.01, threshold_factor=2, threshold=1.5)
        tranges_noisy_sound = np.array(okruns_sound) / sampling_rate if okruns_sound is not None else None
    logger.rectime(1, name='load_data')

    logger.msg(f'test information: {tgtdir.name}')
    logger.msg(f'\nzero data: {inf_zero.name}')
    logger.msg(f'data shape: {p_zero.shape}')
    logger.msg(f'the number of points: {num_points_zero} [points]')
    logger.msg(f'Bring area of zero: {area_zero} [pixel**2]')
    logger.msg(f'reference area from drawings: {Bring_area} [mm**2] (dimeter: 49.1 [m])')
    logger.msg(f'Bring center of zero: {Bcenter_zero} [pixel]')
    logger.msg(f'scaling factor: {scaling_factor_pxl2mm} [mm/pixel]')
    logger.msg(f'\nmarkers data: {inf_markers.name}')
    logger.msg(f'data shape: {p_markers.shape}')
    logger.msg(f'the number of points: {num_points} [points]')
    logger.msg(f'the number of frames: {num_frames} [frames]')
    logger.msg(f'frame rate of high speed camera: {fps} [frame/sec]')
    logger.msg(f'commanded rotation speed of Bring: {rpm} [rpm]')
    logger.msg(f'predicted actual rotation speed of Bring: {rpm_actual} [rpm]')
    logger.msg(f'\naudio data: {inf_audio_name}')
    logger.msg(f'data shape: {sound_shape}')
    logger.msg(f'sampling rate of mic: {sampling_rate} [/sec]')
    logger.msg(f'duration of mic: {duration} [sec]')

    notell = f'{tgtdir.name} sc: {sc}\nbring: {rpm}[rpm], camera: {fps}[fps] rec: {rec_markers}\npxl2mm: {round(scaling_factor_pxl2mm, 3)}[pxl/mm]'
    notelr = f'Dp: {Dp}, Dl: {Dl}\n{note_Dpl}'

    #### calculate cage center
    logger.rectime(0, name='calc_cage_center')
    res_lsm_zero, _ = myfitting.lsm_for_circle(p_zero.reshape(num_points, 2).T)
    r_markers_zero = res_lsm_zero[2] # radius of markers in a static state
    ## determine the mass center of cage with least square fitting
    res_lsm_main_markers, res_lsm_distance_markers = myfitting.lsm_for_circles(p_markers)
    p_center = res_lsm_main_markers[0:2]
    r_markers = res_lsm_main_markers[2] # radius of circle determined by the markers when rotationg
    radii_markers = res_lsm_distance_markers
    ## determine the system center that is assumed to correspond to the center of the cage center trajectory
    res_lsm_main_center , _ = myfitting.lsm_for_circle(p_center)
    system_center = res_lsm_main_center[0:2]
    revolution_radius = res_lsm_main_center[2]
    centrifugal_expansion = r_markers - r_markers_zero
    logger.msg(f'\nresult of calculation')
    logger.msg(f'the system center in the images: {system_center} [mm]')
    logger.msg(f'time-averaged revolution radius over the entire shooting time: {revolution_radius} [mm]')
    logger.msg(f'radius of zero markers in a static state: {r_markers_zero} [mm]')
    logger.msg(f'time-averaged radius of markers over the entire shooting time: {np.nanmean(r_markers)} [mm]')
    logger.msg(f'time-averaged centrifugal expansion of markers over the entire shooting time: {np.nanmean(centrifugal_expansion)} [mm]')
    logger.rectime(1, name='calc_cage_center')

    #### transform the coordinates
    logger.rectime(0, name='coord_conversion')
    p_center -= system_center.reshape(2, -1)
    p_markers -= np.tile(system_center.reshape(2, -1), (num_points, 1))

    # angle_m0 = np.arctan2(p_markers[1]-p_center[1], p_markers[0]-p_center[0])
    # angular_velocity_m0 = np.gradient(angle_m0, t) # it is so noisy

    angles_markers = np.zeros((num_points, num_frames))
    for i in range(num_points):
        angles_markers[i] = np.arctan2(p_markers[i*2+1]-p_center[1], p_markers[i*2]-p_center[0])
    initial_phase_m0 = angles_markers[0, 0]
    angles_markers_ini = angles_markers[:, 0][:, np.newaxis]
    angles_markers = np.unwrap(angles_markers)
    angles_markers -= angles_markers_ini # calclate angular displacement
    angle_cage = np.nanmean(angles_markers, axis=0)
    angular_velocity = np.gradient(angle_cage, t)
    angular_velocity_avg = np.nanmean(angular_velocity)
    angle_rotframe = angular_velocity_avg * t
    angle_cage_rotframe = angle_cage - angle_rotframe
    angular_deviation = abs(angle_cage) - abs(angle_rotframe) # angular deviation from the ideal orientation in a rotating frame with constant velocity
    orbital_ratio = angular_velocity / rpm_actual
    theta_rotframe = angle_rotframe - (np.pi/2 - initial_phase_m0) # rotframe with constant angular velocity
    theta_rotframe2 = angle_cage - (np.pi/2 - initial_phase_m0) # rotframe with fluctuating angular velocity based on cage
    ## transform coordinates to a rotating frame with constant angular velocity, using the system center as the origin
    p_center_rotframe = mycoord.transform_coord_2d(p=p_center, local_origin=np.zeros(2), theta=theta_rotframe, towhich='tolocal')
    p_markers_rotframe = np.zeros((num_points*2, num_frames))
    for i_p in range(num_points):
        p_markers_rotframe[2*i_p:2*(i_p+1), :] = mycoord.transform_coord_2d(p=p_markers[2*i_p:2*(i_p+1), :], local_origin=np.zeros(2), theta=theta_rotframe, towhich='tolocal')
    ## transform coordinates to a rotating frame2 with actual cage angular velocity, using the system center as the origin
    p_center_rotframe2 = mycoord.transform_coord_2d(p=p_center, local_origin=np.zeros(2), theta=theta_rotframe2, towhich='tolocal')
    p_markers_rotframe2 = np.zeros((num_points*2, num_frames))
    for i_p in range(num_points):
        p_markers_rotframe2[2*i_p:2*(i_p+1), :] = mycoord.transform_coord_2d(p=p_markers[2*i_p:2*(i_p+1), :], local_origin=np.zeros(2), theta=theta_rotframe2, towhich='tolocal')
    res_lsm_main_markers_rotframe2, res_lsm_distance_markers_rotframe2 = myfitting.lsm_for_circles(p_markers_rotframe2)
    # p_center_rotframe2 = res_lsm_main_markers_rotframe2[0:2] # it must be (0, 0)
    # r_markers_rotframe2 = res_lsm_main_markers_rotframe2[2] # radius of markers when rotationg
    p_center_rotframe2_polar = mycoord.transform_polar_2d(p_center_rotframe2, towhich='topolar')
    p_center_rotframe2_polar[1] = np.where(p_center_rotframe2_polar[1]<0, p_center_rotframe2_polar[1]+np.radians(360), p_center_rotframe2_polar[1])
    p_center_rotframe2_polar[1] = np.unwrap(p_center_rotframe2_polar[1])
    ## transform coordinates to a rotating frame3 with actual cage angular velocity, using the cage center as the origin
    p_markers_rotframe3 = np.zeros((num_points*2, num_frames))
    for i_p in range(num_points):
        p_markers_rotframe3[2*i_p:2*(i_p+1), :] = mycoord.transform_coord_2d(p=p_markers[2*i_p:2*(i_p+1), :], local_origin=p_center, theta=theta_rotframe2, towhich='tolocal')
    res_lsm_main_markers_rotframe3, res_lsm_distance_markers_rotframe3 = myfitting.lsm_for_circles(p_markers_rotframe3)
    p_center_rotframe3 = res_lsm_main_markers_rotframe3[0:2] # it must be (0, 0)
    r_markers_rotframe3 = res_lsm_main_markers_rotframe3[2] # radius of markers when rotationg
    scale_mks3 = 10
    change_p_markers_rotframe3 = p_markers_rotframe3 - p_markers_rotframe3[:, 0][:, np.newaxis]
    p_markers_rotframe3_xscale = p_markers_rotframe3 + change_p_markers_rotframe3 * scale_mks3
    logger.msg(f'time-averaged rotational angular velocity over the entire shooting time: {angular_velocity_avg/2/np.pi*60} [rpm]')
    logger.rectime(1, name='coord_conversion')

    #### calculate the deformation
    logger.rectime(0, name='calc_deformation')
    dias, ddias, rnd, elp_id, elp_angle = myfitting.calc_elliptical_deformation(p_markers, p_zero)
    dias_rotframe3, ddias_rotframe3, rnd_rotframe3, elp_id_rotframe3, elp_angle_rotframe3 = myfitting.calc_elliptical_deformation(p_markers_rotframe3, p_zero) # the results are same as those in inertial frame except for angular components
    ddia0_demeaned = ddias[0] - np.nanmean(ddias[0])
    okruns_ddia0, okruns_id_ddia0, ngruns_ddia0, ngruns_id_ddia0, ddia0_pw, ddia0_rms, threshold_ddia0 = helperfuncs.detect_noise(ddia0_demeaned, fps, window_time=0.01, threshold_factor=10, threshold=0.1)
    tranges_noisy_ddia0 = np.array(okruns_ddia0) / fps if okruns_ddia0 is not None else None
    tranges_silent_ddia0 = np.array(ngruns_ddia0) / fps if ngruns_ddia0 is not None else None
    logger.rectime(1, name='calc_deformation')

    logger.msg(f'time-averaged roundness over the entire shooting time: {np.nanmean(rnd)}')
    logger.msg(f'threshold of diameter change: {threshold_ddia0}')
    logger.msg(f'noisy time range detected from diameter:\n{tranges_noisy_ddia0}')
    if inf_audio is not None:
        logger.msg(f'threshold of sound rms: {threshold_sound}')
        logger.msg(f'noisy time range detected from audio data:\n{tranges_noisy_sound}')
        tranges_noisy = helperfuncs.merge_ranges(tranges_noisy_sound, tranges_noisy_ddia0, tmax=t[-1])
        # tranges_silent = helperfuncs.merge_ranges(tranges_silent_sound, tranges_silent_ddia0, tmax=t[-1])
        logger.msg(f'noisy time range:\n{tranges_noisy}')
    elif inf_audio == None:
        tranges_noisy = tranges_noisy_ddia0
        # tranges_silent = tranges_silent_ddia0
        logger.msg(f'noisy time range: {tranges_noisy}')
        # logger.msg(f'silent time range: {tranges_silent}')
    #### existence probability
    franges_noisy, id_noisy = helperfuncs.cnvt_trange2frange(tranges_noisy, fps) # indices during noisy period based on camera data scale
    # franges_silent, id_silent = helperfuncs.cnvt_trange2frange(tranges_silent, fps)

    logger.rectime(0, name='existence_probability')
    probability_map, yedges, zedges = np.histogram2d(p_center_rotframe2[0], p_center_rotframe2[1], bins=bins, range=[xyrange_probmap, xyrange_probmap], density=False)
    probability_map /= fps/1000
    if id_noisy is None:
        probability_map_noisy, _, _ = np.histogram2d([], [], bins=bins, range=[xyrange_probmap, xyrange_probmap], density=False)
        id_silent = np.arange(num_frames)
    else:
        id_noisy = np.concatenate(id_noisy)
        probability_map_noisy, _, _ = np.histogram2d(p_center_rotframe2[0][id_noisy], p_center_rotframe2[1][id_noisy], bins=bins, range=[xyrange_probmap, xyrange_probmap], density=False)
        if len(id_noisy) == num_frames:
            id_silent = np.array([])
        elif len(id_noisy) < num_frames:
            mask_silent = np.full(num_frames, True)
            mask_silent[id_noisy] = False
            id_silent = np.where(mask_silent)[0]
    probability_map_noisy /= fps/1000
    if id_silent is None:
        probability_map_silent, _, _ = np.histogram2d([], [], bins=bins, range=[xyrange_probmap, xyrange_probmap], density=False)
    else:
        probability_map_silent, _, _ = np.histogram2d(p_center_rotframe2[0][id_silent], p_center_rotframe2[1][id_silent], bins=bins, range=[xyrange_probmap, xyrange_probmap], density=False)
    probability_map_silent /= fps/1000
    probability = [probability_map, probability_map_noisy, probability_map_silent, yedges, zedges]
    total_duration = (num_frames-1) / fps
    noisy_duration = (len(id_noisy)-1) / fps if id_noisy is not None else 0
    silent_duration = (len(id_silent)-1) / fps if id_silent is not None else 0
    durations = [total_duration, noisy_duration, silent_duration]
    logger.msg(f'total shooting duration: {total_duration}')
    logger.msg(f'duration of noisy segment: {noisy_duration}')
    logger.msg(f'duration of silent segment: {silent_duration}')
    logger.rectime(1, name='existence_probability')

    #### save the data
    logger.rectime(0, name='save_csv')
    header_cage, header_cage2, header_deformation, header_rotation_speed = helperfuncs.make_headers(num_points)
    data_list = [
        np.vstack([frame, t, p_center, angle_cage, p_markers, r_markers, radii_markers]), # cage in inertia frame
        np.vstack([frame, t, p_center_rotframe, angle_cage_rotframe, p_markers_rotframe]), # cage in rotating frame
        np.vstack([frame, t, p_center_rotframe2, np.zeros(num_frames), p_markers_rotframe2]), # cage in rotating frame2
        np.vstack([frame, t, p_center_rotframe3, np.zeros(num_frames), p_markers_rotframe3]), # cage in rotating frame3
        np.vstack([frame, t, dias, ddias, rnd, elp_id[0], elp_angle[0], elp_id[1], elp_angle[1], elp_angle[2]]), # deformation in inertia frame
        np.vstack([frame, t, dias_rotframe3, ddias_rotframe3, rnd_rotframe3, elp_id_rotframe3[0], elp_angle_rotframe3[0], elp_id_rotframe3[1], elp_angle_rotframe3[1], elp_angle_rotframe3[2]]), # defromation in rotating frame3
        np.vstack([frame, t, angle_cage, angular_deviation, angles_markers, angular_velocity, orbital_ratio]) # rotation speed
    ]
    headers_list = [header_cage, header_cage2, header_cage2, header_cage2, header_deformation, header_deformation, header_rotation_speed]
    outfname_list = ['res_cage', 'res_cage_rotframe', 'res_cage_rotframe2', 'res_cage_rotframe3', 'res_deformation', 'res_deformation_rotframe3', 'res_rotation_speed']
    outfcsv_list = [f'{_name}.csv' for _name in outfname_list]
    for i in range(len(data_list)):
        mytools.save_csv(data_list[i], header=headers_list[i], outdir=outdir, outfname=outfcsv_list[i], pkl=True, mkchildir=True)
    logger.rectime(1, name='save_csv')

    #### save figures
    logger.rectime(0, name='save_fig')
    plotter = helperfuncs.PlotterForCageVisualization(config.ROOT/'assets'/'plot_settings.json', testcond, notell=notell, notelr=notelr)
    ### trajectory
    fig_trj, ax_trj = plotter.trajectory(p_center, title='trajectory of cage center')
    fig_trj_rotf2, ax_trj_rotf2 = plotter.trajectory(p_center_rotframe2, title='trajectory of cage center in rotating frame2', is_markpocket=True)
    fig_trj_mks_rotf3, ax_trj_mks_rotf3 = plotter.trajectories(p_markers_rotframe3, title='markers in rotating frame3')
    fig_trj_mksx_rotf3, ax_trj_mksx_rotf3 = plotter.trajectories(p_markers_rotframe3_xscale, title='markers in rotating frame3')
    ### vs time
    ## in inertial frame
    tlist = [t, t, t_sound]
    fig_cnt, axs_cnt = plotter.vstime3(tlist, [p_center[0], p_center[1], sound], ylabel=['y [mm]', 'z [mm]', 'sound pressure [MPa]'], xrange=(t[0]-0.2, t[-1]+0.2), yrange=[plotter.xyrange_trj, plotter.xyrange_trj, (-100, 100)], ysigf=[2, 2, 0], ytick=[0.1, 0.1, 20], title='coordinates of cage center')
    fig_mk0, axs_mk0 = plotter.vstime3(tlist, [p_markers[0], p_markers[1], sound], ylabel=['y [mm]', 'z [mm]', 'sound pressure [MPa]'], xrange=(t[0]-0.2, t[-1]+0.2), yrange=[plotter.xyrange_markers, plotter.xyrange_markers, (-100, 100)], ysigf=0, ytick=[10, 10, 20], title='coordinates of marker0')
    ## in rotating frame2 with actual angular velocity and system center, to visualize the cage eccentricity
    fig_cnt_rotf2, axs_cnt_rotf2 = plotter.vstime3(tlist, [p_center_rotframe2[0], p_center_rotframe2[1], sound], ylabel=['y [mm]', 'z [mm]', 'sound pressure [MPa]'], xrange=(t[0]-0.2, t[-1]+0.2), yrange=[plotter.xyrange_trj, plotter.xyrange_trj, (-100, 100)], ysigf=[2, 2, 0], ytick=[0.1, 0.1, 20], title='coordinates of cage center in rotating frame2')
    _min = np.nanmin(np.degrees(p_center_rotframe2_polar[1]))
    _max = np.nanmax(np.degrees(p_center_rotframe2_polar[1]))
    _diff = abs(_max - _min)
    _deg_range = (_min-20, _max+20) if _diff > 360 else (-20, 380)
    _ytick_deg = int(round(_diff/10, 2 - len(str(int(_diff))))) if _diff > 360 else 45
    fig_cnt_rotf2_polar, axs_cnt_rotf2_polar = plotter.vstime3(tlist, [p_center_rotframe2_polar[0], np.degrees(p_center_rotframe2_polar[1]), sound], ylabel=['r [mm]', 'theta [deg]', 'sound pressure [MPa]'], xrange=(t[0]-0.2, t[-1]+0.2), yrange=[(0, plotter.xyrange_trj[1]), _deg_range, (-100, 100)], ysigf=[2, 0, 0], ytick=[0.2, _ytick_deg, 20], ytick_0center=[False, True, True], title='polar coordinates of cage center in rotating frame2')
    _mid = (np.nanmax(p_markers_rotframe3[1]) + np.nanmin(p_markers_rotframe3[1])) / 2
    _yrange = (round(_mid-1, 2), round(_mid+1, 2))
    fig_mk0_rotf3, axs_mk0_rotf3 = plotter.vstime3(tlist, [p_markers_rotframe3[0], p_markers_rotframe3[1], sound], ylabel=['y [mm]', 'z [mm]', 'sound pressure [MPa]'], xrange=(t[0]-0.2, t[-1]+0.2), yrange=[(-1, 1), _yrange, (-100, 100)], ysigf=[2, 2, 0], ytick=[0.2, 0.2, 20], ytick_0center=[True, False, True], title='coordinates of marker0 in rotating frame3')
    ## deformation
    _indicator = rnd ** 2
    _alpha = (_indicator - np.nanmin(_indicator)) / (np.nanmax(_indicator) - np.nanmin(_indicator))
    _alpha = np.where(np.isnan(_alpha), 0, _alpha)
    fig_deform, axs_deform = plotter.vstime3(tlist, [rnd, ddias[0], sound], ylabel=['roundness [mm]', 'change in diameter of axis0 [mm]', 'sound pressure [MPa]'], xrange=(t[0]-0.2, t[-1]+0.2), yrange=[(0, 1), (-1, 1), (-100, 100)], ysigf=[2, 2, 0], ytick=[0.1, 0.2, 20], title='deformation of cage')
    fig_deform3, axs_deform3 = plotter.vstime3(tlist, [elp_angle_rotframe3[2], elp_angle_rotframe3[0], sound], lws=[0.1, 0.1, 0.4], alphas=[_alpha, _alpha, 1], ylabel=['difference between major and minor axis [degree]', 'angle of minor axis [mm]', 'sound pressure [MPa]'], xrange=(t[0]-0.2, t[-1]+0.2), yrange=[(-200, 200), (-200, 200), (-100, 100)], ysigf=[0, 0, 0], ytick=[45, 45, 20], title='deformation of cage', plottype=['scatter', 'scatter', 'plot'])
    ## rotation speed
    _rpm = int(rpm_actual*0.447)
    _rpm = int(_rpm/100) * 100
    fig_rotation_speed, axs_rotation_speed = plotter.vstime3(tlist, [np.abs(60*angular_velocity/2/np.pi), np.degrees(angular_deviation), sound], ylabel=['rotation speed [rpm]', 'angular deviation [degree]', 'sound pressure [MPa]'], xrange=(t[0]-0.2, t[-1]+0.2), yrange=[(_rpm-1000, _rpm+500), (-270, 100), (-100, 100)], ytick_0center=[False, True, True], ysigf=[0, 0, 0], ytick=[200, 45, 20], title='rotation speed of cage')
    axs_rotation_speed[0].axhline(y=rpm_actual*0.447, lw=1, c='b', alpha=0.4)
    ## detect noise
    fig_sound, axs_sound = plotter.vstime3([t, t_sound, t_sound], [ddia0_rms, sound_rms, sound], ylabel=['rms of diamter0', 'rms of sound', 'sound pressure [MPa]'], xrange=(t[0]-0.2, t[-1]+0.2), yrange=[(0, 0.1), None, (-100, 100)], ysigf=[2, 2, 0], ytick=[0.02, None, 20], title='detection of noise')
    ax_fill_list = [axs_cnt[2], axs_cnt_rotf2[2], axs_cnt_rotf2_polar[2], axs_mk0[2], axs_mk0_rotf3[2], axs_rotation_speed[2], axs_deform[2], axs_deform3[2], axs_sound[2]]
    for _ax in ax_fill_list:
        _ax.fill_between(t_sound, 0, 0.9, where=helperfuncs.get_mask(t_sound, tranges_noisy), color='r', edgecolor=None, alpha=0.1, transform=_ax.get_xaxis_transform())
        _ax.fill_between(t_sound, 0.9, 0.95, where=helperfuncs.get_mask(t_sound, tranges_noisy_sound), color='c', edgecolor=None, alpha=0.1, transform=_ax.get_xaxis_transform())
        _ax.fill_between(t_sound, 0.95, 1.0, where=helperfuncs.get_mask(t_sound, tranges_noisy_ddia0), color='g', edgecolor=None, alpha=0.1, transform=_ax.get_xaxis_transform())
    axs_deform[1].fill_between(t, 0, 1, where=helperfuncs.get_mask(t, tranges_noisy_ddia0), color='g', edgecolor=None, alpha=0.1, transform=axs_deform[1].get_xaxis_transform())
    axs_sound[0].fill_between(t, 0, 1, where=helperfuncs.get_mask(t, tranges_noisy_ddia0), color='g', edgecolor=None, alpha=0.1, transform=axs_sound[0].get_xaxis_transform())
    axs_sound[0].axhline(y=threshold_ddia0, xmin=0, xmax=1, lw=0.4, c='k')
    axs_sound[1].fill_between(t_sound, 0, 1, where=helperfuncs.get_mask(t_sound, tranges_noisy_sound), color='c', edgecolor=None, alpha=0.1, transform=axs_sound[1].get_xaxis_transform())
    axs_sound[1].axhline(y=threshold_sound, xmin=0, xmax=1, lw=0.4, c='k')
    fig_list = [fig_trj, fig_trj_rotf2, fig_trj_mks_rotf3, fig_trj_mksx_rotf3, fig_cnt, fig_cnt_rotf2, fig_cnt_rotf2_polar,  fig_mk0, fig_mk0_rotf3, fig_deform, fig_deform3, fig_rotation_speed, fig_sound]

    outfname_list2 = ['fig_trajectory_center', 'fig_trajectory_center_rotf2', 'fig_trajectory_mks_rotf3', f'fig_trajectory_mks_rotf3_x{scale_mks3}', 'fig_coord_cnt', 'fig_coord_cnt_rotf2', 'fig_coord_cnt_rotf2_polar', 'fig_coord_mk0', 'fig_corrd_mk0_rotf3', 'fig_deformation', 'fig_deformation_rotframe3', 'fig_rotation_speed', 'fig_sound']
    outfpng_list = [f'{_name}.png' for _name in outfname_list2]
    for i in range(len(fig_list)):
        mytools.save_fig(fig_list[i], outdir=outdir, outfname=outfpng_list[i], pkl=True, mkchildir=True)
    logger.rectime(1, name='save_fig')

    #### fft for cage center, marker, deformation, sound
    logger.rectime(0, name='fft')
    ft_list = [p_center[0], p_markers[0], ddias[0], np.degrees(angular_deviation)]
    ylabel0_list = ['cage center', 'marker0', 'change in diameter0', 'angular deviation']
    yrange0_list = [(-1, 1), (-30, 30), (-1, 1), (-90, 90)]
    outfname_fft_list = ['res_fft_cage_center.png', 'res_fft_marker0.png', 'res_fft_diameter0.png', 'res_fft_angular_deviation.png']
    outfname_spectrogram_list = ['res_spectrogram_cage_center.png', 'res_spectrogram_marker0.png', 'res_spectrogram_diameter0.png', 'res_spectrogram_angular_deviation.png']
    ysigf_list = [2, 0, 2, 0]
    fftdir = outdir / 'fft'
    fftdir.mkdir(parents=True, exist_ok=True)
    for i_fft in range(len(ft_list)):
        myfft.fft_main(ft=ft_list[i_fft], dt=dt, num_one_frame=2**12, split_rate=None, overlap=0.5, window_func='hann', title='', ylabel=[ylabel0_list[i_fft], 'amplitude'], xtick=[0.2, 500], xrange=[None, (-100, 8200)], yrange=[yrange0_list[i_fft], None], ysigf=[ysigf_list[i_fft], 2], show_peaks=1, findpeak_height='auto', findpeak_distance=4, save_plot=1, outdir=fftdir, outfname=outfname_fft_list[i_fft], notell=notell)
        myfft.make_spectrogram(ft=ft_list[i_fft], sampling_rate=fps, window='hann', nperseg=2**8, noverlap=0.5, xtick=[0.2, 0.2], yrange=[yrange0_list[i_fft], (0, 8100)], ysigf=[ysigf_list[i_fft], 0], ylabel=[ylabel0_list[i_fft], 'frequency [Hz]'], save_plot=1, outdir=fftdir, outfname=outfname_spectrogram_list[i_fft], notell=notell)
    if inf_audio is not None:
        myfft.fft_main(ft=sound, dt=1/sampling_rate, num_one_frame=2**12, split_rate=None, overlap=0.5, window_func='hann', title='', ylabel=['Sound Pressure [MPa]', 'amplitude'], xtick=[0.2, 500], xrange=[None, (-100, 8100)], yrange=[None, None], ysigf=2, show_peaks=1, findpeak_height='auto', findpeak_distance=4, save_plot=1, outdir=fftdir, outfname='res_fft_sound.png', notell=notell)
        myfft.make_spectrogram(ft=sound, sampling_rate=sampling_rate, window='hann', nperseg=2**10, noverlap=0.5, xtick=[0.2, 0.2], yrange=[None, (0, 8100)], ysigf=[2, 0], ylabel=['Sound Pressure [MPa]', 'frequency [Hz]'], save_plot=1, outdir=fftdir, outfname='res_spectrogram_sound.png', notell=notell)
    logger.rectime(1, name='fft')

    logger.rectime(1, name='main')

    logger.export(outdir=outdir, outfname=f'tc{testcond.test_code:02}_{fps}fps_{rpm}rpm')

    #### apply smoothing fileter
    # p_markers = signal.savgol_filter(p_markers, window_length=25, polyorder=3, deriv=0, delta=dt)
    # ic(p_markers.shape)

    #### perform vector analysis
    # now pass

    summary = {
        'shooting code': sc,
        'x coord of the system center [mm]': system_center[0],
        'y coord of the system center [mm]': system_center[1],
        'rotation speed of Bring [rpm]': rpm,
        'time-averaged rotation speed of cage [rpm]': angular_velocity_avg/2/np.pi*60,
        'time-averaged radius of cage revolution [mm]': revolution_radius,
        'time-averaged centrifugal expansion of cage markers [mm]': np.nanmean(centrifugal_expansion),
        'time-averaged roundness of cage markers [mm]': np.nanmean(rnd),
        'total duration [sec]': total_duration,
        'duration of noisy segment [sec]': noisy_duration,
        'duration of silent segment [sec]': silent_duration
    }

    plt.close('all')
    return summary, probability, durations


if __name__ == '__main__':
    print('\n---- test ----\n')

    tc = 4
    num = 12

    runmode = 'test'
    # testcode = 'tc*'

    date_str = datetime.today().strftime('%y%m%d')

    if runmode == 'test':
        outdir = config.ROOT / 'results' / f'{date_str}_cage_visualization_{config.VERSION}_{runmode}'
        datadir = config.ROOT / 'testdata'
    elif runmode == 'prod' or runmode == 'production':
        outdir = config.ROOT / 'results' / f'{date_str}_cage_visualization_{config.VERSION}'
    outdir.mkdir(parents=True, exist_ok=True)

    tgtdir = datadir
    inf_markers = list(tgtdir.glob('sc*'))[0]
    inf_zero = list(tgtdir.glob('zero*'))[0]
    testcond = helperfuncs.testcond_factory(helperfuncs.TestEnum['TESTDATA1'])

    run_main_computation(inf_markers, inf_zero, inf_audio=None, testcond=testcond, outdir=outdir)







