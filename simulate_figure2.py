# -*- coding: utf-8 -*-
"""
Sample script that generates datasets plotted in figure 2
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helper_funcs import *
import time
from pypet import Environment, cartesian_product, Trajectory
import logging
import os 
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from helper_exploration import *

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

generations = 300000   # Number of generations to simulate 
bins = 50 # bins x bins runs done acros sig_u x sig_s parameter space 

# STEP1 OF PYPET PIPELINE 
# Add parameters for no fluctuations scenario (Fig 2A)
def add_parameters_nofluc(traj):
    """Adds all parameters to `traj`"""
    print('Adding Parameters')

    # Following lines adds common paramaters
    traj.f_add_parameter('com.sig_e', np.sqrt(0.5).item(),
                         comment='Common phenotypic variance of both species')
    traj.f_add_parameter('com.sig_s', np.sqrt(300.0).item(),
                         comment='Strength of stabilising selection')
    traj.f_add_parameter('com.sig_u', np.sqrt(10.0).item(),
                         comment='Utilisation curve variance')
    traj.f_add_parameter('com.sig_eps', np.sqrt(10).item(),
                        comment='Strength of environmental fluctuations')
    traj.f_add_parameter('com.rho', 0.5,
                        comment='Autocorrelation between developmental'
                            'environment and selection environment')
    traj.f_add_parameter('com.tau1', np.float64(0.5),
                        comment='fraction of generation between'
                            'development and selection for species 1')
    traj.f_add_parameter('com.tau2', np.float64(0.5),
                        comment='fraction of generation between'
                            'development and selection for species 2')
    traj.f_add_parameter('com.r', 0.1,
                        comment='Growth rate')
    traj.f_add_parameter('com.seed', 0,
                        comment='Value of seed for choosing random values')
    traj.f_add_parameter('com.tot', generations,
                        comment='Number of generations ot run the simulation')
    
    
    # Following lines add species parameters
    traj.f_add_parameter('sp.A', np.array([5.0, 5.0]),
                         comment='Optimal genetic trait value')
    traj.f_add_parameter('sp.B', np.array([3.0, 3.0]),
                         comment='Optimal plasticity')
    traj.f_add_parameter('sp.a0', np.array([5.3, 4.7]),
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
    traj.f_add_parameter('sp.plast', np.array([-2, -2]),
                         comment='plasticity parameter')

# Add parameters for fluctuations scenario (Fig 2B)
def add_parameters_fluc(traj):
    """Adds all parameters to `traj`"""
    print('Adding Parameters')

    # Following lines adds common paramaters
    traj.f_add_parameter('com.sig_e', np.sqrt(0.5).item(),
                         comment='Common phenotypic variance of both species')
    traj.f_add_parameter('com.sig_s', np.sqrt(300.0).item(),
                         comment='Strength of stabilising selection')
    traj.f_add_parameter('com.sig_u', np.sqrt(10.0).item(),
                         comment='Utilisation curve variance')
    traj.f_add_parameter('com.sig_eps', np.sqrt(10).item(),
                        comment='Strength of environmental fluctuations')
    traj.f_add_parameter('com.rho', 0.5,
                        comment='Autocorrelation between developmental'
                            'environment and selection environment')
    traj.f_add_parameter('com.tau1', np.float64(0.5),
                        comment='fraction of generation between'
                            'development and selection for species 1')
    traj.f_add_parameter('com.tau2', np.float64(0.5),
                        comment='fraction of generation between'
                            'development and selection for species 2')
    traj.f_add_parameter('com.r', 0.1,
                        comment='Growth rate')
    traj.f_add_parameter('com.seed', 0,
                        comment='Value of seed for choosing random values')
    traj.f_add_parameter('com.tot', generations,
                        comment='Number of generations ot run the simulation')
    
    
    # Following lines add species parameters
    traj.f_add_parameter('sp.A', np.array([5.0, 5.0]),
                         comment='Optimal genetic trait value')
    traj.f_add_parameter('sp.B', np.array([3.0, 3.0]),
                         comment='Optimal plasticity')
    traj.f_add_parameter('sp.a0', np.array([5.3, 4.7]),
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
    traj.f_add_parameter('sp.plast', np.array([-1, -1]),
                         comment='plasticity parameter')

# Add parameters for constant plasticity scenario (from earlier exploration)
def add_parameters_constplast(traj):
    """Adds all parameters to `traj`"""
    print('Adding Parameters')

    # Following lines adds common paramaters
    traj.f_add_parameter('com.sig_e', np.sqrt(0.5).item(),
                         comment='Common phenotypic variance of both species')
    traj.f_add_parameter('com.sig_s', np.sqrt(300.0).item(),
                         comment='Strength of stabilising selection')
    traj.f_add_parameter('com.sig_u', np.sqrt(10.0).item(),
                         comment='Utilisation curve variance')
    traj.f_add_parameter('com.sig_eps', np.sqrt(10).item(),
                        comment='Strength of environmental fluctuations')
    traj.f_add_parameter('com.rho', 0.5,
                        comment='Autocorrelation between developmental'
                            'environment and selection environment')
    traj.f_add_parameter('com.tau1', np.float64(0.5),
                        comment='fraction of generation between'
                            'development and selection for species 1')
    traj.f_add_parameter('com.tau2', np.float64(0.5),
                        comment='fraction of generation between'
                            'development and selection for species 2')
    traj.f_add_parameter('com.r', 0.1,
                        comment='Growth rate')
    traj.f_add_parameter('com.seed', 0,
                        comment='Value of seed for choosing random values')
    traj.f_add_parameter('com.tot', generations,
                        comment='Number of generations ot run the simulation')
    
    
    # Following lines add species parameters
    traj.f_add_parameter('sp.A', np.array([5.0, 5.0]),
                         comment='Optimal genetic trait value')
    traj.f_add_parameter('sp.B', np.array([3.0, 3.0]),
                         comment='Optimal plasticity')
    traj.f_add_parameter('sp.a0', np.array([5.3, 4.7]),
                         comment='Initial genetic trait value')
    traj.f_add_parameter('sp.b0', np.array([1.5, 1.5]),
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
    traj.f_add_parameter('sp.plast', np.array([0, 0]),
                         comment='plasticity parameter')

# Add parameters for evolveable plasticity scenario (Fig 2CD)
def add_parameters_evolvplast(traj):
    """Adds all parameters to `traj`"""
    print('Adding Parameters')

    # Following lines adds common paramaters
    traj.f_add_parameter('com.sig_e', np.sqrt(0.5).item(),
                         comment='Common phenotypic variance of both species')
    traj.f_add_parameter('com.sig_s', np.sqrt(300.0).item(),
                         comment='Strength of stabilising selection')
    traj.f_add_parameter('com.sig_u', np.sqrt(10.0).item(),
                         comment='Utilisation curve variance')
    traj.f_add_parameter('com.sig_eps', np.sqrt(10).item(),
                        comment='Strength of environmental fluctuations')
    traj.f_add_parameter('com.rho', 0.5,
                        comment='Autocorrelation between developmental'
                            'environment and selection environment')
    traj.f_add_parameter('com.tau1', np.float64(0.5),
                        comment='fraction of generation between'
                            'development and selection for species 1')
    traj.f_add_parameter('com.tau2', np.float64(0.5),
                        comment='fraction of generation between'
                            'development and selection for species 2')
    traj.f_add_parameter('com.r', 0.1,
                        comment='Growth rate')
    traj.f_add_parameter('com.seed', 0,
                        comment='Value of seed for choosing random values')
    traj.f_add_parameter('com.tot', generations,
                        comment='Number of generations ot run the simulation')
    
    
    # Following lines add species parameters
    traj.f_add_parameter('sp.A', np.array([5.0, 5.0]),
                         comment='Optimal genetic trait value')
    traj.f_add_parameter('sp.B', np.array([3.0, 3.0]),
                         comment='Optimal plasticity')
    traj.f_add_parameter('sp.a0', np.array([5.3, 4.7]),
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
# Defines paramerter exploration on a linear scale
def explore_sigus(traj):
    """ Here is where you change all the kinda exploration you wanna do!"""
    print('Exploring across sig_s and sig_u')
    
    explore_dict = {'sig_u': [np.sqrt(i).item() for i in np.arange(1.0, 81.0, 80.0/bins)],
                    'sig_s': [np.sqrt(i).item() for i in np.arange(1.0, 1225.0, 1224.0/bins)]}
    
    explore_dict = cartesian_product(explore_dict, ('sig_s','sig_u'))
                    
    
    traj.f_explore(explore_dict)
    print('added exploration')

# STEP3 OF PYPET PIPELINE
# Define main run
def main_run(fn, fld, traje, i):
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

    # Add parameters
    if i == 0: add_parameters_nofluc(traj)
    elif i == 1: add_parameters_fluc(traj)
    elif i == 2: add_parameters_constplast(traj)
    elif i == 3: add_parameters_evolvplast(traj)
    # Let's explore
    explore_sigus(traj)

    # Run the experiment
    env.run(run_main_exp)

    # Finally disable logging and close all log-files
    env.disable_logging()
    
if __name__ == '__main__':    

    fn_ = ['NoFluc_main.hdf5','Fluc_main.hdf5', "ConstPlast_main.hdf5", 'EvolvingPlasticity_main.hdf5']
    fld = "figure2"
    # Which conditions to run
    plasts = [0, 1, 3]
    
    for i in plasts:
        traje = 'dummy'
        print(f'starting {i+1} OUT OF 4')
        fn = fn_[i]
        # STEP4 OF PYPET PIPELINE
        main_run(fn, fld, traje, i)
        post_proc(fn, fld, traje)
        print('finished')
