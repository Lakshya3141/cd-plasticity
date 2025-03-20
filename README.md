# Code, data and figures used in the manuscript

## Running a single instance of the model
helper_func.py file contains helper functions to simulate a single run of the model. An example of this can be seen with the file simulate_figure1.py, which runs 5 different scenarios of single runs, corresponding to Figure 1 and Figure 4 of the manuscript

## Parameter exploration across multiple variables
We used pypet (https://pypet.readthedocs.io/en/latest/) to simulate across a large parameter set with multiprocessing. helper_exploration.py contains helper functions to aid parameter exploration. Simulate_figure2.py, simulate_figure3abcd.py, and simulate_figure3ef.py are instances of such parameter exploration, and all each exploration outputs an hdf5 and a csv file. The output files are stored in the hdf5 folder. graphing_figure2.py, graphing_fig3abcd.py, and graphing_figure3ef.py use the data stored in hdf5 folders to plot the data analagous to their file names.
