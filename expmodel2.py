MODULE_DIR = '/Users/ravitpichayavet/Documents/GaTechIE/GraduateResearch/CTC_CVRP/Modules'
GUROBI_LICENSE_DIR = '/Users/ravitpichayavet/gurobi.lic'
MAIN_DIR = '../ComputationalExperiment/'

ARG_COLOR = '#104375'
DEPOT_COLOR = '#D0F6FF'
NODE_COLOR = '#484848'

import sys
sys.path.insert(0,MODULE_DIR)
import importlib
from datetime import datetime

import visualize_sol as vis_sol
import initialize_path as init_path
import random_instance as rand_inst
import utility as util
import model as md
import bnp as bnp
import experiment as exp

import numpy as np
from gurobipy import *
import os
os.environ['GRB_LICENSE_FILE'] = GUROBI_LICENSE_DIR


from itertools import combinations,permutations 
import random 
import pandas as pd
import itertools
import plotly.graph_objects as go
import networkx as nx
import plotly.offline as py 
import pickle as pk
import nltk
import time
import copy

from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial import distance
import logging

import pybnb
from copy import deepcopy
from math import ceil
# logging.basicConfig(filename='myapp.log',filemode='w',format='%(message)s',level=logging.INFO)
# print = logging.info


EXPERIMENT_ID = 4
constant_dict = dict()
NO_DEMAND_NODE = None # wait to be set
# VEHICLE's PARAMETERS
constant_dict['truck_capacity'] = 20
constant_dict['truck_speed'] = 20

# INIT SOLUTION's PARAMETERS
constant_dict['init_max_nodes_proute'] = 3
constant_dict['init_max_vehicles_proute'] = 3

# PROBLEM's PARAMETERS
constant_dict['fixed_setup_time'] = 0
constant_dict['max_vehicles'] = None # wait to be set as Phase 1 solution
constant_dict['time_window'] = 2
constant_dict['tw_avg_factor'] = 1

# PLOTING CONFIG
edge_plot_config = {'line_width':1.5, 'line_color':ARG_COLOR, 'dash':None, 'name':''}


NO_DEMAND_NODE = 36

# DP CONFIG
DOM_RULE = 2
DP_MODE = "SIMUL_M"
DP_TIME_LIM = 150
constant_dict['max_nodes_proute_DP'] = 12
# constant_dict['max_vehicles_proute_DP'] = None
constant_dict['dp_time_limit'] = DP_TIME_LIM
constant_dict['max_vehicles_proute_DP'] = 12

# INIT SOLUTION's PARAMETERS
constant_dict['init_max_nodes_proute'] = 3
constant_dict['init_max_vehicles_proute'] = 3


print(constant_dict)
# BNP CONFIG
BNP_TIME_LIM = 18000

# Create network components: nodes, arcs
labeling_dict = vis_sol.create_nodes(0,NO_DEMAND_NODE)
docking,customers,depot,depot_s,depot_t,all_depot,nodes,arcs = labeling_dict.values()

TYPE_LIST = [1]; 
pref="BnP"
for _t in TYPE_LIST:
    INST_TYPE = _t
    file_no_list = [1]
#     file_no_list = range(1,)
#     file_no_list = range(5,6)
    suffix = 'L2norm'
    # Instance's directory
    TYPE_DIR = MAIN_DIR+'InstancesForExperiment/L2_norm/TYPE{0}/{1}N/'.format(INST_TYPE,NO_DEMAND_NODE)
    INST_NAME_LIST = [ 'InstanceType{0}_{1}n_{2}_{3}'.format(INST_TYPE,
                                                        NO_DEMAND_NODE,
                                                        fn,suffix) for fn in file_no_list]
    # Solution's plot directory
    SOL_DIR = MAIN_DIR+'FinalResults/Experiment{}/{}/SolutionPlots/{}N/'.format(EXPERIMENT_ID,
                                                                            DP_MODE,
                                                                            NO_DEMAND_NODE)
    # Results directory
    RESULT_DIR = MAIN_DIR+'FinalResults/Experiment{}/{}/Results/'.format(EXPERIMENT_ID,DP_MODE)
    EXP_NAME = 'IV_TYPE{0}_{1}N{2}'.format(INST_TYPE,NO_DEMAND_NODE,"_"+pref)
    LOG_F_NAME = RESULT_DIR+pref+'%s_log'%NO_DEMAND_NODE
    
    exp.runExperimentMinAveTimeSpentFromInstList(
                                _inst_list = INST_NAME_LIST,
                                _const_dict = constant_dict,
                                _edge_config = edge_plot_config,
                                _inst_dir = TYPE_DIR,
                                _sol_dir = SOL_DIR,
                                _result_dir = RESULT_DIR,
                                _exp_name = EXP_NAME,
                                _dom_rule = DOM_RULE,
                                _dp_mode = DP_MODE,
                                _log_file_name = LOG_F_NAME,
                                _bnp_time_limit = BNP_TIME_LIM)
