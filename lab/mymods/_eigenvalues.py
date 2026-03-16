"""
Created on Sat Mar 07 10:07:57 2026
@author: santaro



"""

import numpy as np
from scipy import linalg
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

import myplotter


def simple_spring(m1=1, m2=1, k=1):
    print("---- simple_spring ----")
    M = np.array([[m1, 0],
                [0, m2]])
    K = np.array([[k, -k],
                [-k, k]])
    # eigenvalues, eigenvectors = linalg.eig(K, M)
    eigenvalues, eigenvectors = linalg.eigh(K, M)
    print(f"Eigenvalues: {eigenvalues.shape}, {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors.shape}, {eigenvectors}")
    for i in range(len(eigenvalues)):
        freq = np.sqrt(eigenvalues[i])
        print(f"---- solution {i} ----")
        print(f"Eigenvalue : {eigenvalues[i]:.4f}")
        print(f"Frequency : {freq:.4f}")
        print(f"Eigenvector :{eigenvectors[:, i]}")

def simple_spring2(m1=1, m2=1, k=1):
    print("---- simple_spring2 ----")
    M = np.array([[m1, 0],
                [0, m2]])
    K = np.array([[k, -k],
                [-k, k]])
    # eigenvalues, eigenvectors = linalg.eig(K, M)
    eigenvalues, eigenvectors = linalg.eigh(K, M)
    omega = np.sqrt(eigenvalues) # natural frequencies
    v1, v2 = eigenvectors[:, 0], eigenvectors[:, 1] # natural modes
    print(f"Natural frequencies: {omega}")
    print(f"Natural modes:\n{eigenvectors}")
    t = np.linspace(0, 20, 1000)
    u0 = np.array([1, -1])
    c = np.linalg.solve(eigenvectors, u0) # modal coordinates
    print(f"Modal coordinates: {c}")
    mode1_motion = c[0] * np.outer(v1, np.cos(omega[0] * t))
    mode2_motion = c[1] * np.outer(v2, np.cos(omega[1] * t))
    total_motion = mode1_motion + mode2_motion
    plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.LANDSCAPE_FIG_21)
    fig, axs = plotter.myfig()
    axs[0].plot(t, total_motion[0], c='k', lw=2, label="actual motion")
    axs[0].plot(t, mode1_motion[0], c='r', ls='--', lw=1, label="mode 1")
    axs[0].plot(t, mode2_motion[0], c='b', ls='--', lw=1, label="mode 2")
    axs[0].set_title("Mass 1 Motion")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Displacement")
    axs[0].legend()
    axs[1].plot(t, total_motion[1], c='k', lw=2, label="actual motion")
    axs[1].plot(t, mode1_motion[1], c='r', ls='--', lw=1, label="mode 1")
    axs[1].plot(t, mode2_motion[1], c='b', ls='--', lw=1, label="mode 2")
    axs[1].set_title("Mass 2 Motion")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Displacement")
    axs[1].legend()
    plt.show()

def pca1():
    print("---- pca1 ----")
    rng = np.random.default_rng(seed=42)
    x = rng.multivariate_normal(mean=[0, 0], cov=[[3, 2], [2, 3]], size=10000)
    data = x
    x = rng.normal(loc=0, scale=1, size=10000)
    y = 0.8 * x + np.random.normal(scale=0.5, size=x.shape)
    data = np.vstack([x, y]).T
    print(f"Data shape: {data.shape}")
    data_centered = data - np.mean(data, axis=0)
    cov_matrix = np.cov(data_centered, rowvar=False)
    print(f"Covariance matrix:\n{cov_matrix}")

    eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")

    plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
    fig, axs = plotter.myfig()
    ax = axs[0]
    ax.scatter(data[:, 0], data[:, 1], alpha=0.5, label="Data")
    for i in range(len(eigenvalues)):
        vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i])
        ax.arrow(0, 0, vec[0], vec[1], color='r', width=0.05, head_width=0.2, label=f"PC{i+1}")
    ax.set_title("PCA on 2D Data")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    plt.show()



if __name__ == "__main__":
    print(" ---- run ----")
    # simple_spring()
    # simple_spring2()

    pca1()



