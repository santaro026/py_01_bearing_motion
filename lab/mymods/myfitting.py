""" Created on Wed Dec 17 22:03:47 2025
@author: santaro


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

from pathlib import Path
import warnings

class MyFitting:
    def __init__(self, points, name=""):
        self.name = name
        self._points = points
    @property
    def points(self):
        return self._points

    def lsm_for_line(self, allow_nan=False):
        return lsm_for_line(self.points, allow_nan=allow_nan)
    def kasa_circle(self, allow_nan=False):
        return kasa_circle(self.points, allow_nan=allow_nan)
    def taubin_circle(self, allow_nan=False):
        return taubin_circle(self.points, allow_nan=allow_nan)
    def lsm_for_ellipse(self, allow_nan=False):
        return lsm_for_ellipse(self.points, allow_nan=allow_nan)
    def fitzgibbon_ellipse(self, allow_nan=False):
        return fitzgibbon_ellipse(self.points, allow_nan=allow_nan)

    def compare(self, fitting_list=[], allow_nan=False):
        results = []
        for f in fitting_list:
            res, info = f(allow_nan=allow_nan)
            result = {
                "fitting": f.__name__,
                "result": res,
                "info": info,
            }
            results.append(result)
        import myplotter
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        ax = axs[0]
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(len(results))]
        for i, res in enumerate(results):
            fitting = res["fitting"]
            result = res["result"]
            info = res["info"]
            if "line" in fitting:
                x, y = self.make_line(*result, n=10)
            elif "circle" in fitting:
                x, y = self.make_circle(*result, n=500)
            elif "ellipse" in fitting:
                print(f"********** {fitting}: {result}")
                x, y = self.make_ellipse(*result, n=500)
            ax.plot(x, y, lw=1, c=colors[i], alpha=1, label=fitting)
        ax.scatter(self.points[:, 0], self.points[:, 1], s=100, c='grey')
        ax.legend()
        ax.set_aspect(1)
        return fig, ax

    def make_line(self, a, b, n):
        x_min = np.min(self.points[:, 0])
        x_max = np.max(self.points[:, 0])
        x = np.linspace(x_min, x_max, n)
        y = a * x + b
        return x, y
    def make_circle(self, cx, cy, r, n):
        t = np.linspace(0, 2*np.pi, n, endpoint=True)
        x = r * np.cos(t) + cx
        y = r * np.sin(t) + cy
        return x, y
    def make_ellipse(self, cx, cy, a, b, theta, n):
        t = np.linspace(0, 2*np.pi, n, endpoint=True)
        x = a * np.cos(t) + cx
        y = b * np.sin(t) + cy
        return x, y


#### least squares fitting of line
def lsm_for_line(points, allow_nan=False):
    if points.shape[1] != 2:
        raise ValueError("argument of points must be shape of (N, 2).")
    has_nan_or_inf = np.isnan(points).any() or np.isinf(points).any()
    if not allow_nan and has_nan_or_inf:
        return np.full(3, np.nan), None
    valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    valid_id = np.where(valid_mask)[0]
    x = points[valid_id, 0]
    y = points[valid_id, 1]
    num_points = len(x)
    M1 = np.vstack([x, np.ones(num_points)]).T
    ab, residuals, rank, s = np.linalg.lstsq(M1, y, rcond=None) # ab is slope and intercept
    info = {
        "rss": float(residuals[0]) if residuals.size > 0 else None,
        "rank": int(rank),
        "singular_values": s,
        "num_points": len(x),
        "valid_indices": valid_id
    }
    # M2 = np.array(y)
    # ab = np.linalg.inv(M1.T @ M1) @ M1.T @ M2
    return ab, info

#### least squares fitting of circles
def kasa_circle(points, allow_nan=False):
    if points.shape[1] != 2:
        raise ValueError("argument of points must be shape of (N, 2).")
    has_nan_or_inf = np.isnan(points).any() or np.isinf(points).any()
    if not allow_nan and has_nan_or_inf:
        return np.full(3, np.nan), None
    valid_points_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    valid_points_id = np.where(valid_points_mask)[0]
    if len(valid_points_id) < 3:
        raise ValueError(f"need at least 3 valid points")
    x = points[valid_points_id, 0]
    y = points[valid_points_id, 1]
    num_points = len(x)
    M1 = np.vstack([x, y, np.ones(num_points)]).T
    M2 = -(x**2 + y**2)
    coef, residuals, rank, s = np.linalg.lstsq(M1, M2, rcond=None)
    A, B, C = coef
    # A, B, C = np.dot(np.linalg.inv(np.dot(M1.T, M1)), np.dot(M1.T, M2))
    cx, cy = -A/2, -B/2
    r = (cx**2 + cy**2 - C)**0.5
    xyr = np.array([cx, cy, r])
    d_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    geom_err = np.abs(d_center - r)
    info = {
        "rss": float(residuals[0]) if residuals.size > 0 else None,
        "rank": int(rank),
        "singular_values": s,
        "num_points": len(x),
        "num_valid_points": len(valid_points_id),
        "valid_points_indices": valid_points_id,
        "radii": d_center,
        "geom_error_mean": float(np.mean(geom_err)),
        "geom_error_std": float(np.std(geom_err)),
        "geom_error_max": float(np.max(geom_err)),
    }
    return xyr, info

def lsm_for_circle(*args, **kwargs):
    warnings.warn(
        f"{lsm_for_ellipse.__name__} is deprecated and will be removed in a future version."
        f"please use kasa_circle instead.",
        category=DeprecationWarning,
        stacklevel=2
    )
    return kasa_circle(*args, **kwargs)

def taubin_circle(points, allow_nan=False):
    if points.shape[1] != 2:
        raise ValueError("argument of points must be shape of (N, 2).")
    has_nan_or_inf = np.isnan(points).any() or np.isinf(points).any()
    if not allow_nan and has_nan_or_inf:
        return np.full(3, np.nan), None
    valid_points_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    valid_points_id = np.where(valid_points_mask)[0]
    if len(valid_points_id) < 3:
        raise ValueError(f"need at least 3 valid points")
    x = points[valid_points_id, 0]
    y = points[valid_points_id, 1]
    num_points = len(x)
    mx = np.mean(x)
    my = np.mean(y)
    zx = x - mx
    zy = y - my
    z = zx**2 + zy**2
    mz = np.mean(z)
    Z = np.column_stack([z, zx, zy, np.ones(len(x))])
    M = (Z.T @ Z) / len(x)
    P = np.array([[8*mz, 0, 0, 2],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [2, 0, 0, 0]])
    # eigenvalues, eigenvectors = linalg.eig(M)
    eigenvalues, eigenvectors = linalg.eig(M, P)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    A_vec = None
    idx = np.argsort(np.abs(eigenvalues))
    for i in idx:
        if eigenvalues[i] > -1e-12:
            A_vec = eigenvectors[:, i]
            break
    if A_vec is None:
        print(f"valid eigenvalues was not found.")
        return np.full(3, np.nan), None
    a, b1, b2, c = A_vec
    cx_rel = -b1 / (2 * a)
    cy_rel = -b2 / (2 * a)
    r = np.sqrt((b1**2 + b2**2 - 4 * a * c) / (4 * a**2))
    cx = cx_rel + mx
    cy = cy_rel + my
    xyr = np.array([cx, cy, r])
    d_center = np.sqrt((x - cx)**2 + (cy - y)**2)
    geom_err = np.abs(d_center - r)
    rss = np.sum(geom_err**2)
    info = {
        "rss": rss,
        "rank": None, # not implemented
        "singular_values": None, # not implemented
        "num_points": len(x),
        "num_valid_points": len(valid_points_id),
        "valid_points_indices": valid_points_id,
        "radii": d_center,
        "geom_error_mean": float(np.mean(geom_err)),
        "geom_error_std": float(np.std(geom_err)),
        "geom_error_max": float(np.max(geom_err)),
    }
    return xyr, info

def lsm_for_circles(points):
    """
    Least squares fitting of circles for sequential frames.
    it doesn't use linalg.lstsq of numpy to avoid for loops for frames, is useful when the number of frames is greater than that of points.

    """
    if points.shape[2] != 2:
        raise ValueError("argument of points must be shape of (M, N, 2).")
    valid_frames_mask = (~np.isnan(points).any(axis=2) & ~np.isinf(points).any(axis=2)).all(axis=1)
    valid_frames_id = np.where(valid_frames_mask)[0]
    if len(valid_frames_id) < 3:
        raise ValueError(f"need at least 3 valid points")
    num_frames = points.shape[0]
    num_points = points.shape[1]
    xyrs = np.full((num_frames, 3), np.nan)
    d_center = np.full((num_frames, num_points), np.nan)
    xs = points[valid_frames_id, :, 0]
    ys = points[valid_frames_id, :, 1]
    num_frames_valid = xs.shape[0]
    M1 = np.stack([xs, ys, np.ones((num_frames_valid, num_points))], axis=2)
    M1T = M1.transpose(0, 2, 1)
    M2 = -(xs**2 + ys**2)[:, :, np.newaxis]
    inv_M1T_M1 = np.linalg.inv(M1T @ M1)
    # inv_M1T_M1 = np.linalg.pinv(M1T @ M1)
    M1T_M2 = M1T @ M2
    ABC = (inv_M1T_M1 @ M1T_M2).T.squeeze()
    # A, B, C = ABC[0, 0], ABC[0, 1], ABC[0, 2]
    # print(ABC.shape)
    A, B, C = ABC[0], ABC[1], ABC[2]
    # Compute the circle centers (cx, cy) and radius (r)
    cx = -A / 2
    cy = -B / 2
    r = np.sqrt(cx**2 + cy**2 - C)  # Shape (num_frames,)
    xyrs[valid_frames_id, :] = np.column_stack([cx, cy, r])
    d_center[valid_frames_id, :] = np.sqrt((xs - cx[:, np.newaxis])**2 + (ys - cy[:, np.newaxis])**2)
    info = {
        "num_points": num_points,
        "num_frames": num_frames,
        "num_valid_frames": num_frames_valid,
        "valid_indices": valid_frames_id,
        "radii": d_center,
    }
    return xyrs, info

def lsm_for_ellipse(points, allow_nan=False):
    """
    Least squares fitting of ellipse
    points: shape of (num_points, 2)
    """
    if points.shape[1] != 2:
        raise ValueError("argument of points must be shape of (N, 2).")
    has_nan_or_inf = np.isnan(points).any() or np.isinf(points).any()
    if not allow_nan and has_nan_or_inf:
        return np.full(5, np.nan), None
    valid_points_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    valid_points_id = np.where(valid_points_mask)[0]
    if len(valid_points_id) < 3:
        raise ValueError(f"need at least 3 valid points")
    x = points[valid_points_id, 0]
    y = points[valid_points_id, 1]
    num_points, _ = points.shape
    num_points_valid = len(valid_points_id)
    M1 = np.vstack([x*y, y**2, x, y, np.ones(num_points_valid)]).T
    M2 = -x**2
    coef, residuals, rank, s = np.linalg.lstsq(M1, M2, rcond=None)
    A, B, C, D, E = coef
    cx = (A*D - 2*B*C) / (4*B - A**2)
    cy = (A*C - 2*D) / (4*B - A**2)
    if abs(A/(1-B)) > 10**14: # first aid to avoid singular error
        a , b = np.nan, np.nan
        # a , b = 0, 0
        theta = np.radians(45)
    else:
        theta = np.arctan(A/(1-B)) / 2
        # theta = np.atan2(A, (1-B)) / 2
        sin = np.sin(theta)
        cos = np.cos(theta)
        a = np.sqrt(
            (cx*cos + cy*sin)**2 - E*cos**2
                - ((cx*sin - cy*cos)**2 - E*sin**2)
                    *(sin**2 - B*cos**2) / (cos**2 - B*sin**2)
            )
        b = np.sqrt(
            (cx*sin - cy*cos)**2 - E*sin**2
                - ((cx*cos + cy*sin)**2 - E*cos**2)
                    *(cos**2 - B*sin**2) / (sin**2 - B*cos**2)
            )
    result = np.array([cx, cy, a, b, theta])
    d_center = np.sqrt((x - cx)**2 +(y - cy)**2)
    geom_err = np.full(num_points, np.nan)
    sin = np.sin(theta)
    cos = np.cos(theta)
    ux = points[:, 0] - cx
    uy = points[:, 1] - cy
    x_local = ux * cos + uy * sin
    y_local = -ux * sin + uy * cos
    for i in range(num_points):
        geom_err[i], _ = calc_mindist_p2ellipse(np.array([x_local[i], y_local[i]]), a, b)
    info = {
        "rss": float(residuals[0]) if residuals.size > 0 else None,
        "rank": int(rank),
        "singular_values": s,
        "num_points": len(x),
        "num_valid_points": num_points_valid,
        "valid_points_indices": valid_points_id,
        "radii": d_center,
        "geom_error_mean": float(np.mean(geom_err)),
        "geom_error_std": float(np.std(geom_err)),
        "geom_error_max": float(np.max(geom_err)),
    }
    return result, info

def fitzgibbon_ellipse(points, allow_nan=False):
    if points.shape[1] != 2:
        raise ValueError("argument of points must be shape of (N, 2).")
    has_nan_or_inf = np.isnan(points).any() or np.isinf(points).any()
    if not allow_nan and has_nan_or_inf:
        return np.full(5, np.nan), None
    valid_points_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    valid_points_id = np.where(valid_points_mask)[0]
    if len(valid_points_id) < 5:
        raise ValueError(f"need at least 5 valid points")
    x = points[valid_points_id, 0]
    y = points[valid_points_id, 1]
    num_points, _ = points.shape
    num_points_valid = len(valid_points_id)
    mx = np.mean(x)
    my = np.mean(y)
    scale = np.max(np.abs(points))
    x_n = (x - mx) / scale # normalize data to improve numerical stability
    y_n = (y - my) / scale
    D = np.vstack([x_n**2, x_n*y_n, y_n**2, x_n, y_n, np.ones_like(x_n)]).T
    S = np.dot(D.T, D) # (6, 6) matrix for scatter of data points
    C = np.zeros((6, 6)) # constraint matrix to enforce the condition for ellipse
    C[0, 2] = 2
    C[1, 1] = -1
    C[2, 0] = 2
    eigvals, eigvecs = linalg.eig(S, C) # solve the generalized eigenvalue problem S*a = lambda*C*a
    # pos_idx = np.where((eigvals > 0) & (np.isfinite(eigvals)))[0]
    pos_idx = np.where((eigvals.real > -1e-12) & (np.isfinite(eigvals)))[0]
    if len(pos_idx) == 0:
        print(f"markers: {points}")
        print(f"eigenvalues: {eigvals}")
        print("no valid eigenvalue found, check the input points.")
        return np.full(6, np.nan), None
        # raise ValueError("no valid eigenvalue found, check the input points.")
    elif len(pos_idx) > 1:
        print("multiple valid eigenvalues found, check the input points.")
        return np.full(6, np.nan), None
        # raise ValueError("multiple valid eigenvalues found, check the input points.")
    abcdef_norm = eigvecs[:, pos_idx].real.flatten()
    a_n, b_n, c_n, d_n, e_n, f_n = abcdef_norm[:]
    a, b, c = a_n / scale**2, b_n / scale**2, c_n / scale**2
    d = d_n / scale - (2 * a * mx) - (b * my)
    e = e_n / scale - (2 * c * my) - (b * mx)
    f = f_n - (d_n * mx / scale) - (e_n * my / scale) + (a * mx**2) + (c * my**2) + (b * mx * my)
    abcdef = np.array([a, b, c, d, e, f])
    residuals = D @ abcdef_norm[:, np.newaxis]
    rank = np.linalg.matrix_rank(D)
    xyabtheta = abcdef2xyabtheta(abcdef)
    d_center = np.sqrt((x - xyabtheta[0])**2 + (y - xyabtheta[1])**2)

    dx = x - xyabtheta[0]
    dy = y - xyabtheta[1]
    cos = np.cos(xyabtheta[4])
    sin = np.sin(xyabtheta[4])
    x_rot = dx * cos + dy * sin
    y_rot = -dx * sin + dy * cos
    # geom_err, _ = calc_mindist_p2ellipse(np.array([x - xyabtheta["center"][0], y - xyabtheta["center"][1]]), xyabtheta["axes"][0], xyabtheta["axes"][1])
    geom_err, _ = calc_mindist_p2ellipse(np.array([x_rot, y_rot]).T, xyabtheta[2], xyabtheta[3])
    _, s, _ = linalg.svd(D)
    info = {
        "rss": float(np.sum(residuals**2)),
        "rank": int(rank),
        "singular_values": s,
        "num_points": len(x),
        "num_valid_points": num_points_valid,
        "valid_points_indices": valid_points_id,
        "radii": d_center,
        "geom_error": geom_err,
        "geom_error_mean": float(np.mean(geom_err)),
        "geom_error_std": float(np.std(geom_err)),
        "geom_error_max": float(np.max(geom_err)),
        "eigenvalues": eigvals[pos_idx],
        "eigenvectors": eigvecs[:, pos_idx],
        "abcdef": abcdef,
    }
    return xyabtheta, info

def abcdef2xyabtheta(abcdef):
    a, b, c, d, e, f = abcdef
    delta = b**2 - 4*a*c
    cx = (b*e - 2*c*d) / (-delta)
    cy = (b*d - 2*a*e) / (-delta)
    theta = 0.5 * np.arctan2(b, a-c)
    up = 2 * (a * cx**2 + b * cx * cy + c * cy**2 -f)
    down1 = a + c - np.sqrt((a - c)**2 + b**2)
    down2 = a + c + np.sqrt((a - c)**2 + b**2)
    semi1 = np.sqrt(max(0, up / down1))
    semi2 = np.sqrt(max(0, up / down2))
    major = max(semi1, semi2)
    minor = min(semi1, semi2)
    return np.array([cx, cy, major, minor, theta])

def calc_mindist_p2ellipse(points, a, b, mode="newton", tol=1e-12, max_iter=100):
    """
    Calculates the minimum distance from a point (p, q) to an ellipse with semi-axes a and b.
    this is invalid in certain parameters like (0.001, 0.001), 1, 2 because of inner_sqrt become negative.

    """
    if points.ndim == 1:
        points = points[np.newaxis, :]
    if np.isscalar(a): a = np.full(len(points), a)
    if np.isscalar(b): b = np.full(len(points), b)
    p, q = points[:, 0], points[:, 1]
    if mode == "newton":
        is_converge = False
        # theta = np.arctan2(Q*a, P*b) # initial value
        theta = np.arctan2(q*a, p*b) # initial value
        for _ in range(max_iter):
            ct = np.cos(theta)
            st = np.sin(theta)
            f = (a**2 - b**2) * ct * st - p * a * st + q * b * ct
            f_prime = (a**2 - b**2) * (ct**2 - st**2) - p * a * ct - q * b * st
            step = f / f_prime
            theta_new = theta - step
            if np.all(np.abs(step) < tol):
                theta = theta_new
                is_converge = True
                break
            theta = theta_new
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        r = np.sqrt((x-p)**2 + (y-q)**2)
        theta = np.arctan2(y/b, x/a)
    elif mode == "algebra":
        # Calculate intermediate variables
        A = (-a**2 - a*p + b**2) / (b * q)
        B = (-a**2 + a*p + b**2) / (b * q)
        D = (a**2 - b**2)**2 - (a**2 * p**2) - (b**2 * q**2)
        term1 = -432 * a * b * p * q * (a**2 - b**2)
        term2 = 12 * D
        inner_sqrt = np.sqrt(term1**2 - 4 * (term2**3))
        C = np.cbrt(term1 + inner_sqrt)
        cbrt2 = np.cbrt(2)
        denom_bqC = b * q * C
        denom_bq_cbrt = 3 * cbrt2 * b * q
        term_D = (4 * cbrt2 * D) / denom_bqC
        term_C = C / denom_bq_cbrt
        sqrt_part1 = np.sqrt(A**2 + term_D + term_C)
        inner_numerator = 2 * A**3 - 4 * B
        nested_radical = np.sqrt(
            2 * A**2 - term_D - term_C - (inner_numerator / sqrt_part1)
        )
        # solev
        theta_val = (A / 2) - (0.5 * sqrt_part1) + (0.5 * nested_radical)
        theta = 2 * np.arctan(theta_val)
        r = np.sqrt((a * np.cos(theta) - p)**2 + (b * np.sin(theta) - q)**2)
    return r.squeeze(), theta.squeeze()

#### calculate elliptical deformation
def calc_elliptical_deformation(points, points_ref):
    """
    evaluate elliptical deformation; roundness, major and minor axis, ect
    parameter:
    points: in the shape of (num_frames, num_points, 2)
    points_ref: in the shape of (num_points, 2)

    """
    if points.ndim == 2:
        points = points[np.newaxis, :, :]
        num_frames, num_points, _ = points.shape
    elif points.ndim == 3:
        num_frames, num_points, _ = points.shape
    if points_ref.ndim == 2:
        points_ref = points_ref[np.newaxis, :, :]
    num_axes = num_points // 2
    #### calclate diameters
    diameters_vct = points[:, :num_axes, :] - points[:, num_axes:, :]
    diameters_norm = np.linalg.norm(diameters_vct, axis=2)
    diameters_theta = np.arctan2(diameters_vct[:, :, 1], diameters_vct[:, :, 0])
    diameters_ref_vct = points_ref[:, :num_axes, :] - points_ref[:, num_axes:, :]
    diameters_ref_norm = np.linalg.norm(diameters_ref_vct, axis=2)
    delta_diameters = diameters_norm - diameters_ref_norm
    roundness = (np.amax(delta_diameters, axis=1) - np.amin(delta_diameters, axis=1)) / 2
    direction_id = np.array([np.argmin(delta_diameters, axis=1), np.argmax(delta_diameters, axis=1)]).T
    direction = np.array([diameters_theta[np.arange(num_frames), direction_id[:, 0]], diameters_theta[np.arange(num_frames), direction_id[:, 1]]]).T
    deformation_angle = np.abs(direction[:, 1] - direction[:, 0]) % np.pi
    results = {
        "diameters_norm": diameters_norm,
        "delta_diameters": delta_diameters,
        "roundness": roundness,
        "direction_id": direction_id,
        "direction": direction,
        "deformation_angle": deformation_angle,
    }
    return results



# def calc_cumulative_angles(angles, threshold=300, unit='deg'):
    full_angle = 360 if unit=='deg' else 2*np.pi
    if np.ndim(angles) == 1:
        d_angles = angles[1:] - angles[:-1]
        flag_forward = np.hstack([0, np.where(d_angles<-threshold, 1, 0)])
        flag_backward = np.hstack([0, np.where(d_angles>threshold, -1, 0)])
        flag = np.cumsum(flag_forward) + np.cumsum(flag_backward)
        angles_corrected = angles + flag * full_angle
    elif np.ndim(angles) > 1:
        d_angles = angles[:, 1:] - angles[:, :-1]
        flag_forward = np.hstack([np.zeros((len(angles), 1)), np.where(d_angles<np.full((len(angles), 1), -threshold), 1, 0)])
        flag_backward = np.hstack([np.zeros((len(angles), 1)), np.where(d_angles>np.full((len(angles), 1), threshold), -1, 0)])
        flag = np.cumsum(flag_forward, axis=1) + np.cumsum(flag_backward, axis=1)
        angles_corrected = angles + flag * full_angle
    return angles_corrected, flag

# sample data generator
class SimpleCage:
    def __init__(self, name='', PCD=50, ID=48, OD=52, width=10, num_pockets=8, num_markers=8, num_mesh=100, Dp=6.25, Dw=5.953):
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
        self.pos_pockets = np.linspace(0, 2*np.pi, self.num_pockets, endpoint=False) + np.pi/2
        self.pos_markers = np.linspace(0, 2*np.pi, self.num_markers, endpoint=False) + np.pi/2
        self.pos_node = np.linspace(0, 2*np.pi, self.num_nodes, endpoint=True) + np.pi/2

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
    def make_cage_points(num_frames, num_points, x_value, a, b, deform_angle, transformer, p0_angle=np.pi/2):
        p_lcs = np.full((num_frames, num_points, 3), np.nan) # points of pockets on local coordinate system
        p_global = np.full((num_frames, num_points, 3), np.nan)
        pos_points = np.linspace(0, 2*np.pi, num_points, endpoint=False) + p0_angle
        if deform_angle != 0: rot_euler_for_ellipse = np.vstack([deform_angle, np.zeros(num_frames), np.zeros(num_frames)]).T
        for i in range(num_points):
            _x = np.full(num_frames, x_value)
            _theta = np.full(num_frames, pos_points[i])
            _y = a * np.cos(_theta)
            _z = b * np.sin(_theta)
            p_lcs[:, i, :] = np.vstack([_x, _y, _z]).T
            if deform_angle != 0: p_lcs[:, i, :] = mycoord.CoordTransformer3D.rotate_euler(p_lcs[:, i, :], euler_angles=rot_euler_for_ellipse, rot_order="zyx")
            p_global[:, i, :] = transformer.transform_coord(p_lcs[:, i, :], towhich='toglobal')
        return p_global, p_lcs

    def time_series_data(self, fps=10000, p_cage=np.vstack([np.zeros(10001), 0.4*np.cos(np.linspace(0, 2*np.pi*40, 10001)), 0.4*np.sin(np.linspace(0, 2*np.pi*40, 10001))]).T, R_cage=np.vstack([np.linspace(0, 2*np.pi*40, 10001), np.zeros(10001), np.zeros(10001)]).T,  a=1, b=1, noise_type="normal", noise_max=1, p0_angle=np.pi/2):
        self.fps = fps
        self.num_frames = p_cage.shape[0]
        self.duration = (self.num_frames - 1) / self.fps
        self.t = np.linspace(0, self.duration, self.num_frames, endpoint=True)
        self.dt = 1 / self.fps
        self.p_cage = p_cage
        self.R_cage = R_cage
        self.omega_rot = np.gradient(R_cage[:, 0], self.dt)
        self.angle_rev = np.unwrap(np.arctan2(self.p_cage[:, 2], self.p_cage[:, 1]))
        self.omega_rev = np.gradient(self.angle_rev, self.dt)
        self.r_rev = np.linalg.norm(self.p_cage[:, 1:], axis=1)
        self.omega_rot_avg = np.nanmean(self.omega_rot)
        self.rpm_avg = np.nanmean(self.omega_rot) / (2*np.pi) * 60
        self.euler_angles1 = np.vstack([self.omega_rot_avg * self.t, np.zeros(self.num_frames), np.zeros(self.num_frames)]).T # euler angle for rotation frame with constant velocity
        self.euler_angles2 = np.vstack([self.R_cage[:, 0], np.zeros(self.num_frames), np.zeros(self.num_frames)]).T # euler angle for rotation frame with instant velocity
        self.transformer_SI = mycoord.CoordTransformer3D(coordsys_name="cage_coordsys", local_origin=np.zeros((self.num_frames, 3)), euler_angles=self.R_cage, rot_order='zyx')
        self.p_cage_lcs = self.transformer_SI.transform_coord(self.p_cage, towhich="tolocal")
        self.transformer_CI= mycoord.CoordTransformer3D(coordsys_name="cage_coordsys", local_origin=self.p_cage, euler_angles=self.R_cage, rot_order='zyx')
        #### generate points on cage
        self.p_pockets, self.p_pockets_lcs = SimpleCage.make_cage_points(num_frames=self.num_frames, num_points=self.num_pockets, x_value=0, a=a, b=b, deform_angle=0, transformer=self.transformer_CI, p0_angle=p0_angle)
        self.p_markers, self.p_markers_lcs = SimpleCage.make_cage_points(num_frames=self.num_frames, num_points=self.num_markers, x_value=self.width/2, a=a, b=b, deform_angle=0, transformer=self.transformer_CI, p0_angle=p0_angle)
        self.p_nodes, self.p_nodes_lcs = SimpleCage.make_cage_points(num_frames=self.num_frames, num_points=self.num_nodes, x_value=0, a=a, b=b, deform_angle=0, transformer=self.transformer_CI, p0_angle=p0_angle)
        #### add noise to the marker
        rng = np.random.default_rng(seed=0)
        if noise_type == 'uniform':
            noise = rng.uniform(-1, 1, (self.num_frames, self.num_markers, 3)) * noise_max
        elif noise_type == 'normal':
            noise = rng.normal(0, 1/3, (self.num_frames, self.num_markers, 3)) * noise_max
        self.p_markers_noise_lcs = self.p_markers_lcs + noise
        self.p_markers_noise = np.full((self.num_frames, self.num_markers, 3), np.nan)
        for i in range(self.num_markers):
            self.p_markers_noise[:, i, :] = self.transformer_CI.transform_coord(self.p_markers_noise_lcs[:, i, :], towhich='toglobal')

    def time_series_data2(self, fps=10000, duration=1, omega_rot=40*np.pi, omega_rev=40*np.pi, r_rev=0.4, a=1, b=1, omega_deform=0, noise_type="normal", noise_max=1, p0_angle=np.pi/2):
        num_frames = int(fps * duration + 1)
        # t = np.linspace(0, duration , num_frames)
        t = np.arange(num_frames) / fps
        dt = 1 / fps
        x = np.zeros(num_frames)
        y = r_rev * np.cos(omega_rev*t)
        z = r_rev * np.sin(omega_rev*t)
        p_cage = np.vstack([x, y, z]).T
        Rx = np.cumsum(np.hstack([0, np.full(num_frames-1, omega_rot)])) * dt
        Ry = np.zeros(num_frames)
        Rz = np.zeros(num_frames)
        R_cage = np.vstack([Rx, Ry, Rz]).T
        self.time_series_data(fps=fps, p_cage=p_cage, R_cage=R_cage, a=a, b=b, noise_type=noise_type, noise_max=noise_max, p0_angle=p0_angle)




if __name__ == '__main__':
    print('---- test ----')
    #### ellipse
    rng = np.random.default_rng()
    num_nodes = 20
    node = np.linspace(0, 2*np.pi, num_nodes)
    a = 7
    b = 4
    x = a * np.cos(node) + rng.normal()
    y = b * np.sin(node) + rng.normal()
    p = np.column_stack([x, y])
    print(f"data points: {p.shape}")

    myfitting = MyFitting(p)
    fittings = [myfitting.lsm_for_line, myfitting.kasa_circle, myfitting.taubin_circle, myfitting.lsm_for_ellipse, myfitting.fitzgibbon_ellipse]
    fig, ax = myfitting.compare(fittings)
    plt.show()


    # xyr, info = lsm_for_circle(p)
    # xyr_taubin, info_taubin = taubin_circle(p)
    # print(f"kasa: {xyr}")
    # print(f"taubin: {xyr_taubin}")

    # abcdef, xyabtheta, info = fitzgibbon_ellipse(p)

    # x_cfit = xyr[2] * np.cos(node) + xyr[0]
    # y_cfit = xyr[2] * np.sin(node) + xyr[1]

    # x_tcfit = xyr[2] * np.cos(node) + xyr_taubin[0]
    # y_tcfit = xyr[2] * np.sin(node) + xyr_taubin[1]

    # x_efit = xyabtheta["axes"][0] * np.cos(node) + xyabtheta["center"][0]
    # y_efit = xyabtheta["axes"][1] * np.sin(node) + xyabtheta["center"][1]

    # xyrange = 10
    # import myplotter
    # plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
    # fig, axs = plotter.myfig()
    # ax = axs[0]
    # ax.plot(x, y, lw=8, c='k', alpha=0.2)

    # ax.plot(x_cfit, y_cfit, lw=4, c='r', ls='--', alpha=0.4)
    # ax.plot(x_tcfit, y_tcfit, lw=2, c='m', ls='--', alpha=0.4)
    # ax.plot(x_efit, y_efit, lw=4, c='b', ls='--', alpha=0.4)

    # ax.set(xlim=(-xyrange, xyrange), ylim=(-xyrange, xyrange))
    # ax.set_aspect(1)
    # ax.axhline(y=0, lw=0.4, c='k')
    # ax.axvline(x=0, lw=0.4, c='k')
    # plt.show()




    #### line
    # t = np.linspace(0, 10, 20)
    # rng = np.random.default_rng(seed=0)
    # x = t + rng.uniform(-1, 1, 20)
    # y = x*2 + rng.uniform(-1, 1, 20)
    # fitter = MyFitting(np.vstack([x, y]).T)
    # fitter.lsm_for_line()
    # fitter.check()

    # ps = np.array([0.001, 10])
    # d, theta = calc_mindist_p2ellipse(ps, 2, 1, max_iter=100000)
    # print(d, np.degrees(theta))


    """
    duration = 0.2
    fps = 10000
    num_frames = int(duration * fps) + 1
    rng = np.random.default_rng(seed=0)
    import mycage
    cage = mycage.SimpleCage(name='', PCD=50, ID=48, OD=52, width=10, num_pockets=8, num_markers=8, num_mesh=100, Dp=6.25, Dw=5.953)
    t = np.arange(num_frames) / fps
    a = 4
    b = 1
    cage.time_series_data2(fps=fps, duration=duration, omega_rot=40*np.pi, omega_rev=40*np.pi, r_rev=2, a=a, b=b, omega_deform=10*np.pi, noise_type="normal", noise_max=0.01, p0_angle=np.pi/15)
    import mycoord
    transformer = mycoord.CoordTransformer3D(coordsys_name="cage_coordsys", local_origin=np.zeros((1, 3)), euler_angles=np.zeros(3), rot_order='zyx')
    points_zero = cage.make_cage_points(num_frames=1, num_points=8, x_value=5, a=25, b=25, deform_angle=0, transformer=transformer)[0][:, :, 1:]
    cut = 0
    markers = cage.p_markers_noise[:, cut:, 1:]
    markers_defect = cage.p_markers_noise[:, :, 1:].copy()
    markers_defect[500:1000, 0, 0] = np.nan

    xyr = np.zeros((num_frames, 3))
    for i in range(num_frames):
        xyr[i], info = lsm_for_circle(markers[i])
    # xyr_defect = np.zeros((num_frames, 3))
    # for i in range(num_frames):
    #     xyr_defect[i], info = lsm_for_circle(markers_defect[i], allow_nan=False)

    xyabtheta = np.zeros((num_frames, 5))
    for i in range(num_frames):
        xyabtheta[i], info = lsm_for_ellipse(markers[i], allow_nan=False)
    # xyabtheta_defect = np.zeros((num_frames, 5))
    # for i in range(num_frames):
    #     xyabtheta_defect[i], info = lsm_for_ellipse(markers_defect[i], allow_nan=False)

    # xyr2, info = lsm_for_circles(markers)
    # xyr2_defect, info = lsm_for_circles(markers_defect)
    # results = calc_elliptical_deformation(markers, points_zero)

    abcdef = np.zeros((num_frames, 6))
    xyabtheta_fitzgibbon = np.zeros((num_frames, 5))
    infos = []
    for i in range(num_frames):
        abcdef[i], info = fitzgibbon_ellipse(markers[i], allow_nan=False)
        _xyabtheta = abcdef2xyabtheta(abcdef[i])
        xyabtheta_fitzgibbon[i] = _xyabtheta["center"][0], _xyabtheta["center"][1], _xyabtheta["axes"][0], _xyabtheta["axes"][1], _xyabtheta["angle"]
    print(info)

    datalist = [
        # {"id": 0, "data": xyr[:, 0], "lw": 4, "c": 'g', "alpha": 0.2},
        # {"id": 1, "data": xyr[:, 1], "lw": 4, "c": 'g', "alpha": 0.2},
        # {"id": 2, "data": xyr[:, 2], "lw": 4, "c": 'g', "alpha": 0.4},
        # {"id": 0, "data": xyr_defect[:, 0], "lw": 1, "c": 'b', "alpha": 0.8},
        # {"id": 1, "data": xyr_defect[:, 1], "lw": 1, "c": 'b', "alpha": 0.8},
        # {"id": 2, "data": xyr_defect[:, 2], "lw": 1, "c": 'b', "alpha": 0.8},
        # {"id": 0, "data": xyr2[:, 0], "lw": 2, "c": 'r', "alpha": 0.8},
        # {"id": 1, "data": xyr2[:, 1], "lw": 2, "c": 'r', "alpha": 0.8},
        # {"id": 2, "data": xyr2[:, 2], "lw": 2, "c": 'r', "alpha": 0.8},
        # {"id": 0, "data": xyr2_defect[:, 0], "lw": 1, "c": 'b', "alpha": 0.8},
        # {"id": 1, "data": xyr2_defect[:, 1], "lw": 1, "c": 'b', "alpha": 0.8},
        # {"id": 2, "data": xyr2_defect[:, 2], "lw": 1, "c": 'b', "alpha": 0.8},

        {"id": 0, "data": xyabtheta[:, 0], "lw": 4, "c": 'b', "alpha": 0.4},
        {"id": 1, "data": xyabtheta[:, 1], "lw": 4, "c": 'b', "alpha": 0.4},
        {"id": 2, "data": xyabtheta[:, 2], "lw": 4, "c": 'b', "alpha": 0.4},
        # {"id": 2, "data": xyabtheta[:, 3], "lw": 1, "c": 'g', "alpha": 1},
        # {"id": 0, "data": xyabtheta_defect[:, 0], "lw": 8, "c": 'm', "alpha": 0.2},
        # {"id": 1, "data": xyabtheta_defect[:, 1], "lw": 8, "c": 'm', "alpha": 0.2},
        # {"id": 2, "data": xyabtheta_defect[:, 2], "lw": 8, "c": 'm', "alpha": 0.2},
        # {"id": 2, "data": xyabtheta_defect[:, 3], "lw": 8, "c": 'm', "alpha": 0.2},

        {"id": 0, "data": xyabtheta_fitzgibbon[:, 0], "lw": 1, "c": 'r', "alpha": 1},
        {"id": 1, "data": xyabtheta_fitzgibbon[:, 1], "lw": 1, "c": 'r', "alpha": 1},
        {"id": 2, "data": xyabtheta_fitzgibbon[:, 2], "lw": 1, "c": 'r', "alpha": 1},
        {"id": 2, "data": xyabtheta_fitzgibbon[:, 3], "lw": 1, "c": 'm', "alpha": 1, "ls": '--'},

        # {"id": 0, "data": results["diameters_norm"][:, 0], "lw": 1, "c": 'r', "alpha": 0.8},
        # {"id": 0, "data": results["diameters_norm"][:, 1], "lw": 1, "c": 'b', "alpha": 0.8},
        # {"id": 0, "data": results["diameters_norm"][:, 2], "lw": 1, "c": 'g', "alpha": 0.8},
        # {"id": 1, "data": results["roundness"], "lw": 1, "c": 'r', "alpha": 0.8},
        # {"id": 2, "data": np.degrees(results["direction"][:, 0]), "lw": 1, "c": 'r', "alpha": 0.8},
        # {"id": 2, "data": np.degrees(results["direction"][:, 1]), "lw": 1, "c": 'b', "alpha": 0.8},
        # {"id": 2, "data": np.degrees(results["deformation_angle"]), "lw": 1, "c": 'k', "alpha": 0.8},
    ]
    print(xyabtheta_fitzgibbon[:, 3])
    f = np.arange(num_frames)
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    # axs[0].set_ylim(40, 60)
    # axs[1].set_ylim(-1, 10)
    axs[2].set_ylim(0, 100)
    # axs[2].set_ylim(0, 30)
    for i in range(3):
        axs[i].set_xlim(0, num_frames)
    for i in range(len(datalist)):
        _d = datalist[i]
        axs[_d["id"]].plot(f, _d["data"], lw=_d["lw"], c=_d["c"], alpha=_d["alpha"])

    diff_center = xyr[:, :2] - xyabtheta_fitzgibbon[:, :2]
    error_center_circle = xyr[:, :2] - cage.p_cage[:, 1:]
    error_center_fitz = xyabtheta_fitzgibbon[:, :2] - cage.p_cage[:, 1:]
    error_a_circle = xyr[:, 2] - min(a * cage.PCD/2, b * cage.PCD/2)
    error_b_circle = xyr[:, 2] - max(a * cage.PCD/2, b * cage.PCD/2)
    error_a_fitz = xyabtheta_fitzgibbon[:, 2] - min(a * cage.PCD/2, b * cage.PCD/2)
    error_b_fitz = xyabtheta_fitzgibbon[:, 3] - max(a * cage.PCD/2, b * cage.PCD/2)

    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    axs[0].plot(f, diff_center[:, 0], lw=1, c='k', alpha=0.4)
    axs[0].plot(f, error_center_circle[:, 0], lw=1, c='g', alpha=0.4)
    axs[0].plot(f, error_center_fitz[:, 0], lw=1, c='r', alpha=0.4)

    axs[1].plot(f, diff_center[:, 1], lw=1, c='k', alpha=0.4)
    axs[1].plot(f, error_center_circle[:, 1], lw=1, c='g', alpha=0.4)
    axs[1].plot(f, error_center_fitz[:, 1], lw=1, c='r', alpha=0.4)

    axs[2].plot(f, error_a_circle, lw=1, c='k', alpha=0.4)
    # axs[2].plot(f, error_b_circle, lw=1, c='k', alpha=0.4)
    axs[2].plot(f, error_a_fitz, lw=1, c='r', alpha=0.4)
    # axs[2].plot(f, error_b_fitz, lw=1, c='r', alpha=0.4)

    axs[0].set(ylim=(-0.1, 0.1))
    axs[1].set(ylim=(-0.1, 0.1))
    axs[2].set(ylim=(-40, 40))


    f = 200
    node = np.linspace(0, 1, 100)
    x_circle = xyr[f, 2] * np.cos(2*np.pi*node) + xyr[f, 0]
    y_circle = xyr[f, 2] * np.sin(2*np.pi*node) + xyr[f, 1]

    def rotate_points(p, theta):
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])[np.newaxis, :, :]
        p = p[:, :, np.newaxis]
        p_rotated = R @ p
        return p_rotated

    x_ellipse = xyabtheta[f, 2] * np.cos(2*np.pi*node)
    y_ellipse = xyabtheta[f, 3] * np.sin(2*np.pi*node)
    ellipse = np.vstack([x_ellipse, y_ellipse]).T
    p = rotate_points(ellipse, xyabtheta[f, 4]).squeeze() + xyabtheta[f, :2][np.newaxis, :]

    # x_ellipse = xyabtheta_defect[f, 2] * np.cos(2*np.pi*node)
    # y_ellipse = xyabtheta_defect[f, 3] * np.sin(2*np.pi*node)
    # ellipse = np.vstack([x_ellipse, y_ellipse]).T
    # p_defect = rotate_points(ellipse, xyabtheta_defect[f, 4]).squeeze() + xyabtheta_defect[f, :2][np.newaxis, :]

    # abcdef = fitzgibbon_ellipse(markers[f], allow_nan=False)
    # ellipse_fitzgibbon = abcdef2xyabtheta(abcdef)
    major = xyabtheta_fitzgibbon[f, 2]
    minor = xyabtheta_fitzgibbon[f, 3]
    x_ellipse_fitzgibbon = major * np.cos(2*np.pi*node)
    y_ellipse_fitzgibbon = minor * np.sin(2*np.pi*node)
    markers_fitzgibbon = np.vstack([x_ellipse_fitzgibbon, y_ellipse_fitzgibbon]).T
    p_fitzgibbon = rotate_points(markers_fitzgibbon, xyabtheta_fitzgibbon[f, 4]).squeeze() + xyabtheta_fitzgibbon[f, :2][np.newaxis, :]

    print(f"circle center: {xyr[f, :2]}")
    print(f"ellipse center: {xyabtheta[f, :2]}")
    print(f"fitzgibbon ellipse center: {xyabtheta_fitzgibbon[f, :2]}")

    fig_trj, ax_trj = plt.subplots(figsize=(8, 8))
    ax_trj.set_aspect(1)
    ax_trj.grid()

    ax_trj.scatter(markers[f, 0, 0], markers[f, 0, 1], c='r', s=100, alpha=1)
    ax_trj.scatter(markers[f, 1, 0], markers[f, 1, 1], c='b', s=100, alpha=1)
    ax_trj.scatter(markers[f, 2:, 0], markers[f, 2:, 1], c='k', s=40, alpha=1)

    ax_trj.plot(x_circle, y_circle, lw=4, c='g', alpha=0.4)
    ax_trj.scatter(xyr[f, 0], xyr[f, 1], s=200, c='g', alpha=1, marker='x', zorder=100)

    ax_trj.plot(p[:, 0], p[:, 1], lw=1, c='b', alpha=1, ls='--')
    ax_trj.scatter(xyabtheta[f, 0], xyabtheta[f, 1], s=200, c='b', alpha=0.2, marker='+', zorder=99)

    ax_trj.plot(p_fitzgibbon[:, 0], p_fitzgibbon[:, 1], lw=1, c='r', alpha=1)
    ax_trj.scatter(xyabtheta_fitzgibbon[f, 0], xyabtheta_fitzgibbon[f, 1], s=200, c='r', alpha=0.2, marker='o', zorder=99)

    xymax = max(np.abs(np.min(markers[f, :, :])), np.abs(np.max(markers[f, :, :]))) * 1.1
    ax_trj.set(xlim=(-xymax, xymax), ylim=(-xymax, xymax))

    print(f"fitzgibbon a, b: {xyabtheta_fitzgibbon[f, 2:4]}")
    print(f"fitzgibbon theta: {np.degrees(xyabtheta_fitzgibbon[f, 4])}")

    plt.show()



    """