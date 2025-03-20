"""
This script contains code to generate data plotted in figure 3EF using pypet
This script produces the file allexp.csv.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helper_funcs import *
from helper_exploration import *
import time
from pypet import Environment, cartesian_product, Trajectory
import logging
import os
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

generations = 300000   # Number of generations to simulate 
bins = 50 # bins x bins runs done acros sig_u x sig_s parameter space 

# STEP1 OF PYPET PIPELINE 
# Add parameters for exploring sig_u, sig_s, sig_eps and tau2
def add_parameters_all(traj):

    
    """Adds all parameters to `traj`"""
    print('Adding Parameters')

    # Following lines adds common paramaters
    traj.f_add_parameter('com.sig_e', np.sqrt(0.5).item(),
                         comment='Common phenotypic variance of both species')
    traj.f_add_parameter('com.sig_s', np.sqrt(1000.0).item(),
                         comment='Strength of stabilising selection')
    traj.f_add_parameter('com.sig_u', np.sqrt(10.0).item(),
                         comment='Utilisation curve variance')
    traj.f_add_parameter('com.sig_eps', np.sqrt(8).item(),
                        comment='Strength of environmental fluctuations')
    traj.f_add_parameter('com.rho', 0.5,
                        comment='Autocorrelation between developmental'
                            'environment and selection environment')
    traj.f_add_parameter('com.r', 0.1,
                        comment='Growth rate')
    traj.f_add_parameter('com.seed', 0,
                        comment='Value of seed for choosing random values')
    traj.f_add_parameter('com.tot', generations,
                        comment='Number of generations ot run the simulation')

    traj.f_add_parameter('com.tau1', np.float64(0.5),
                        comment='fraction of generation between'
                            'development and selection for species 1')
    traj.f_add_parameter('com.tau2', np.float64(0.2),
                        comment='fraction of generation between'
                            'development and selection for species 2')
    traj.f_add_parameter('sp.A', np.array([5.0, 5.0]),
                         comment='Optimal genetic trait value')
    traj.f_add_parameter('sp.B', np.array([3.0, 3.0]),
                         comment='Optimal plasticity')
    traj.f_add_parameter('sp.a0', np.array([5.01, 4.99]),
                         comment='Initial genetic trait value')
    traj.f_add_parameter('sp.b0', np.array([2.5, 2.51]),
                         comment='Initial plasticity value')
    traj.f_add_parameter('sp.kar', np.array([60000.0, 60000.0]),
                         comment='Carrrying capacities')
    traj.f_add_parameter('sp.n0', traj.kar/2,
                         comment='Inital populations, default half of carrying')
    traj.f_add_parameter('sp.Gaa', np.array([0.5, 0.5]),
                         comment='variance of trait a')
    traj.f_add_parameter('sp.Gbb', np.array([0.045, 0.045]),
                         comment='variance of trait b')
    # growth parameter: 0 -> static population, 1 -> growing population
    traj.f_add_parameter('sp.grow', np.array([1, 1]),
                         comment='growth parameter')
    # plasticity parameter: -2 -> no fluctuations, -1 -> no plasticity
    #                       0 -> constant plasticity, 1 -> evolving plasticity
    traj.f_add_parameter('sp.plast', np.array([1, 1]),
                         comment='plasticity parameter')

# STEP2 OF PYPET PIPELINE 
# Add parameter exploration for sig_u, sig_s, sig_eps and tau2
def param_exp_all(traj):
    """ Here is where you change all the kinda exploration you wanna do!"""
    print('Exploring across tau1 vs tau2')
    
    explore_dict = {'sig_u': [np.sqrt(i).item() for i in np.arange(1.0, 81.0, 80.0/bins)],
                    'sig_s': [np.sqrt(i).item() for i in np.arange(441.0, 1600.0, 1159.0/bins)],
                    'tau2': [i for i in np.float64(np.array([0.4, 0.9]))],
                    # 'a0': [np.array([5.01, i]) for i in np.array([4.99, 5.99])],
                    'sig_eps': [np.sqrt(i).item() for i in np.array([2.0, 6.0, 10.0])]}
    
    explore_dict = cartesian_product(explore_dict, ('sig_s','sig_u','sig_eps','tau2'))
                    
    
    traj.f_explore(explore_dict)
    print('added exploration')

# STEP3 OF PYPET PIPELINE
# Define main run
def main(fn, fld, traje):
    filename = os.path.join('hdf5', fld, fn)
    env = Environment(trajectory=traje,
                      comment='Setting up the pypet pipeline for our '
                            'temporal model of character displacement. ',
                      add_time=False, # We don't want to add the current time to the name,
                      log_stdout=True,
                      log_config='DEFAULT',
                      multiproc=True,
                      ncores=24,
                      wrap_mode='QUEUE',
                      filename=filename,
                      overwrite_file=True)
    traj = env.trajectory
    
    add_parameters_all(traj)
    
    # Let's explore
    param_exp_all(traj)

    # Run the experiment
    env.run(run_main_exp)

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':    
    
   
    fn = 'allexp.hdf5'
    fld = "figure3"    
    
    traje = 'dummy'
    print('starting')
    main(fn, fld, traje)
    # STEP4 OF PYPET PIPELINE
    post_proc(fn, fld, traje)
    print('finished')
