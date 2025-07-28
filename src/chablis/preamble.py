import numpy as np
import scipy as scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.colors import LogNorm, Normalize
import matplotlib.scale as mscale
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.markers import MarkerStyle
#import seaborn as sns
import scipy.integrate as integrate
from scipy import optimize
from scipy import sparse # for the baseline correction
from scipy.sparse.linalg import spsolve # for the baseline correction
from scipy.interpolate import CubicSpline # for interpolating the charge data
from scipy.interpolate import splrep, BSpline # for smoothed interpolation
from scipy.optimize import minimize
import functools
import pandas as pd
import glob
import os
import itertools
from datetime import datetime
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 11pt font in plots to match thesis
    "axes.labelsize": 11,
    "font.size": 11,
    # Make the legend/label fonts a bit smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

plt.rcParams.update(tex_fonts)

import importlib.util

def get_cwd():
    notebook_dir = os.getcwd()
    return notebook_dir

def fig_size(width = 437.46112, fraction = 0.8, subplots=(1, 1), aspect_ratio=1):
    #width (float) = document textwidth or columnwidth in pts
    #fraction (float, optional) = fraction of width the figure will occupy

    fig_width_pt = width * fraction # fig width in pts
    inches_per_pt = 1 / 72.27 # pt to inch conversion
    golden_ratio = (5**.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt # fig width in inches
    fig_height_in = fig_width_in * aspect_ratio * golden_ratio * (subplots[0] / subplots[1]) # fig height in inches
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim

# a little test