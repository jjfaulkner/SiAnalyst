import numpy as np
import scipy as scipy
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
import functools
import pandas as pd
import glob
import os
import itertools
#import ipywidgets as widgets
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

def get_cwd():
    notebook_dir = os.getcwd()
    return notebook_dir