paleochrono
===========

A statistical and physical model to optimize chronologies of paleoclimatic sites.


What this manual is and is not?
-------------------------------

This manual is a documentation on how to use the paleochrono software.  
It is _not_ a description of the paleochrono principles. Please read to the scientific articles
describing paleochrono for that purpose:\
Parrenin, F., Bouchet, M., Buizert, C., Capron, E., Corrick, E., Drysdale, R., Kawamura, K., Landais, A., Mulvaney, R., Oyabu, I., and Rasmussen, S. O.: 
The Paleochrono-1.1 probabilistic model to derive a common age model for several paleoclimatic sites using absolute and relative dating constraints, 
_Geoscientific Model Development_, 17, 8735–8750, https://doi.org/10.5194/gmd-17-8735-2024, 2024.
It is _not_ an operating system or python documentation.
Please use your operating system or python documentation instead.


Where can I get help on paleochrono?
------------------------------------

A mailing list has been set up on Google Groups:  
https://groups.google.com/forum/?hl=en#!forum/icechrono  
You just need a google account to access this mailing list.
You can also directly email to Frédéric Parrenin: frederic.parrenin@univ-grenoble-alpes.fr

How to download paleochrono?
----------------------------

Go here:  
https://github.com/parrenin/paleochrono/  
and click on the donwload button.  
In the downloaded folder, you will find the following files:
- `README.md`		: is the current documentation of paleochrono.
- `LICENCE`		: is the paleochrono licence file.
- `paleochrono.py`		: is the main paleochrono program that you will run.
- `pccfg.py`, pcmath.py, pcsite.py and pcsitepair.py	: are python modules used by paleochrono.py
- `Clean.py`		: is a python script to clean a dating experiment directory
- `convert_exp_header_in_txt.py` : is a python script to convert an old experiment to the new format with name columns for .txt files
- `AICC2023-Hulu`		: is an example experiment directory: it contains all the necessary
numerical settings, prior information and observations for the different ice cores in the AICC2023
dating experiment and for the MSD and MSL Hulu speleothem. It takes a few minutes to run an a recent
computer.
- `AICC2023-LowRes` : is the experiment directory used to create the AICC2023 ice core chronology.
It uses a lower resolution than the official AICC2023 experiment so that it runs faster.
- `Hulu_MSL` : is an experiment directory used in the Paleochrono-1.1 article for only the Hulu/MSL speleothem.
It is a simple example that you can follow when you date only one record from a simple archive.
- `EN18208` and `EN18218` : are two one-site experiments with C14 ages.

What do I need to run paleochrono?
----------------------------------

paleochrono is a scientific python3 software, therefore you need a scipy distribution.  
paleochrono is developed and tested using the anaconda distribution, therefore we recommend it.  
Anaconda can be downloaded here (use the python3 version):  
https://www.anaconda.com/download 
If you have an old computer or if you want to limit the installation size, you can opt for miniconda, which is a minimal version of Anaconda.
It installs faster, but then you have to install modules afterwards.

Paleochrono uses the following python modules:  
- sys
- os
- time
- math
- numpy
- matplotlib
- warnings
- scipy
- yaml
- gc
- multiprocessing
- warnings
- pickle
- pandas
- iosacal (optional, for C14 calibration)


How to run paleochrono?
---------------------

Assuming you use anaconda, you can go in the spyder IDE and type the following commands in the
ipython interpreter:

```
cd path-to-palechrono
run paleochrono.py exp_directory/
```

where `path-to-paleochrono` is the directory containing paleochrono and `exp_directory` is the name of
your experiment directory. 
A few experiment directories are provided for you convenience.
You can for example try the `Hulu_MSL` experiment, it takes a few seconds to run.

What are the outputs of a run:
------------------------------

If the run went correctly, it has created output files.

In the main directory, you have the following output file:
- `output.txt`: contains the size of the residuals and variables vectors, the initial and final
values of the cost function, the program execution times and the memory usage.
- `obs_residuals.pdf` : is a distribution diagram for the observations

In each site directory, you have the following output files:
- `output.txt`: is the main output file. It gives you the posterior estimates and
uncertainties of the three input variables (accumulation, LID and thinning) and of the output
 variables (ice age, air age, Δdepth, etc.). The header in the file tells you which column is which
 variable.
- `restart.txt`: is a restart file, which can be used to start an optimization experiment from the
result of a previous optimization experiment, for a faster convergence.
- `deposition.pdf`: is the deposition rate figure
- `deposition_log.pdf`: is the deposition rate figure with a log scale
- `age.pdf`: is the age for a non-ice-core archive
- `ice_age.pdf` and `air_age.pdf`: are the ice and air age figures for an ice core
- `delta_depth.pdf`: is the Δdepth figure for an ice core
- `ice_layer_thickness` and `air_layer_thickness.pdf`: are the ice and air layer thickness figures
for an ice core
- `lock_in_depth.pdf`: is the Lock-In Depth figure for an ice core
- `thinning.pdf`		: is the thinning figure for an ice core
- `obs_residuals.pdf` : is a distribution diagram for the observations related to this site

In each site-pair directory, you have the following output files:
- `synchro.pdf`: is the stratigraphic links figure for a pair of non-ice-cores archives
- `ice_synchro.pdf`: is the ice stratigraphic links figure when there is one ice core in the pair
- `air_synchro.pdf`: is the air stratigraphic links figure when there is one ice core in the pair
- `air_air_synchro.pdf`		: is the air-air stratigraphic links figure for a pair of ice cores
- `air_ice_synchro.pdf`		: is the air-ice stratigraphic links figure for a pair of ice cores
- `ice_air_synchro.pdf`		: is the ice-air stratigraphic links figure for a pair of ice cores
- `ice_ice_synchro.pdf`		: is the ice-ice stratigraphic links figure for a pair of ice cores
- `residuals.pdf` : is a distribution diagram for the observations of this site pair


How to clean an experiment directory after a run?
-------------------------------------------------

If your run was successful, it has produced output files and figure files.
To clean it from the results files, you can run the following command in ipython:

```
run Clean.py exp_directory/
```


What is the structure of an experiment directory?
-------------------------------------------------

You can have a look at the provided `Hulu_MSL` or `AICC20123-Hulu` directories.
`Hulu_MSL` is a simple experiment with one single site of a simple archive.
`AICC20123-Hulu` is a more complex experiment with several sites (simples and ice cores) combined together.
You need to specify your prior scenarios for deposition rate (in all cases) and LID and thinning
(for an ice core) and your age observations.

You have five general files:
- `parameters.yml`: contains general parameters for the
experiment
- `parameters_all_sites.yml`: defines site parameters that are the same
for all sites (there are overidded by site specific parameters).
- `parameters_covariance_observations_all_sites.py`: defines the covariance of the
observations that are the same for all sites  (there are overidded by site specific parameters).
- `parameters_covariance_observations_all_site_pairs.py`: defines the covariance for the
observations that are the same for all site pairs  (there are overidded by site pair specific
parameters).

Then you have one directory per site, which contains:
- `parameters.yml`: all the site specific parameters
- `parameters_covariance_observations.py`: this file allows to define the correlation of site
 specific observations
- `deposition.txt`: depth / deporate (/ opt., rel_unc)
- `age_horizons.txt`: depth / age / age_unc for dated horizons for a non-ice-core
- `age_horizons_C14.txt`: depth / age / age_unc for C14-dated horizons for a non-ice-core
- `age_intervals.txt`: depth_top / depth_bot / duration / dur_unc for intervals for a non-ice-core
- `ice_age_horizons.txt`: depth / age / age_unc for ice dated horizons for an ice core
- `air_age_horizons.txt`: depth / age / age_unc for air dated horizons for an ice core
- `ice_age_intervals.txt`: depth_top / depth_bot / duration / dur_unc for ice intervals for an
ice core
- `air_age_intervals.txt`: depth_top / depth_bot / duration / dur_unc for air intervals for an
ice core
- `density.txt`: depth / rel_dens for an ice core
- `lock_in_depth.txt`: depth / LID (/ opt., rel_unc) for an ice core
- `thinning.txt`: depth / thinning (/ opt., rel_unc) for an ice core
- `delta_depths.txt`: air_depth / Ddepth / Ddepth_unc for an ice core

Then you have one directory per site pair, which contains:
- `parameters_covariance_observations.py`: this file allows to define the correlation of site pair
specific observations
- `synchro_horizons.txt`: depth1 / depth2 / age_unc for stratigraphic links observations (two simple archives)
- `ice_synchro_horizons.txt`: depth1 / depth2 / age_unc for ice-ice stratigraphic links observations (one ice core, one simple archive)
- `air_synchro_horizons.txt`: depth1 / depth2 / age_unc for air-air stratigraphic links observations (one ice core, one simple archive)
- `iceice_synchro_horizons.txt`: depth1 / depth2 / age_unc for ice-ice stratigraphic links observations (two ice cores)
- `airair_synchro_horizons.txt`: depth1 / depth2 / age_unc for air-air stratigraphic links observations (two ice cores)
- `iceair_synchro_horizons.txt`: depth1 / depth2 / age_unc for ice-air stratigraphic links observations (two ice cores)
- `airice_synchro_horizons.txt`: depth1 / depth2 / age_unc for air-ice stratigraphic links observations (two ice cores)

A few things you need to know to use paleochrono:
1) You can use whatever units you want but they need to be consistent. For example, if you use meters for the depths and years for the dated horizons, you need to use meters per years for the accumulation rates. 
2) The site specific parameters override the general parameters for all sites. In the very same way, the site-pair specific parameters override the general parameters for all site-pairs.
3) Most of these files are optional. If there is no file for an certain type of observations, that means that there is no observation of this type. If a covariance matrix is not defined for an observation type, that means that the correlation matrix is supposed to be equal to identity and that the standard deviation is given in the observation file.
4) You can put comment at the end of `.yml` or `.txt` file after a # character. This is handy if you want to add a label to, e.g., a dated horizon.

New structure of data .txt files
---------------------------------
Starting April, 5th, 2025, we have a new structure for .txt files.
These files now have a header which contains the name of each column.
The names are specific and should be strictly equal to what Paleochrono expects.
This new structure has several advantages:
- the names in the header explains what each column is
- the columns can now be reordered, as long as the header corresponds (with the same separator)
- you can have your own descriptive columns, e.g., lab-id, method, author, etc.
- it is now possible to have several optional columns, which was not possible with the old format
The `convert_exp_header_in_txt.py` allows to convert an experiment to the new format.
It should work on most .txt files, but there might be corner cases which are not dealt with.
In this case, drop me an email with the file and I will adjust the script.

What is the structure of the general `parameters.yml` file?
--------------------------------------------------------

It contains the list of sites, the optimization method to be used and some settings for the figures.
It is where you define the names of your sites.
Have a look at the file `AICC2023-Hulu/parameters.yml`, it is commented.


What is the structure of a site `parameters.yml` file?
---------------------------------------------------------

It defines age at the top of the core, the unthinned depth at the top of the core, the age equation grid, the correction functions grids and the type of representation of the prior accu scenario (linear or staircase). You can also define other parameters that are used to defined the covariance matrices of the priors.
Have a look at the files `AICC2023-Hulu/EDC/parameters.yml`, it is commented.

How do I use the C14 calibration?
---------------------------------
Paleochrono can calibrate your C14 ages using the python module iosacal.
For now Paleochrono uses a least-squares optimization, 
so the distribution of calibrated ages is transformed into a Gaussian distribution by calculating its mean and its standard deviation.

First, you need to install iosacal, which is a python software for C14 calibration.
isoacal is included in conda-forge, so you have to add the conda-forge repo.
If you are using Anaconda, you can add conda-forge like this (in a terminal or Anaconda prompt):
```
conda config --append channels conda-forge
```
Then you can install iosacal like this:
```
conda install iosacal
```
To change the C14 calibration curve, you can set the `c14_cal` key in the site-specific parameters.yml file.
Please have a look at the iosacal doc to have a list of C14 calibration curves:

https://iosacal.readthedocs.io/en/latest/how.html#other-calibration-curves

The `dated_horizons_C14.txt` works with a header to describe the columns.\
The following columns should be present: `depth`, `age` and `age_unc`.\
Optionnally, you can specify `res_age` and `res_unc` for a reservoir age and its uncertainty.\
You can also optionnally specify a per-horizon synchronisation curve with the `calib` column. If you put `default`, the default site calibration curve will be used.\
You can also add other columns, for example `label` if you want to specify a label.\
The order of the columns does not matter, as long as the header is correct.\
Be careful that the C14 ages should always be expressed in BP, even though the rest is expressed in a different ref (e.g., CE).

How to set up the `parameters-CovarianceObservations.py` file?
--------------------------------------------------------------

You need to know a little bit of python to do that.
Feel free to send an email on the mailing list if you need assistance.

For site specific observations, you set up the correlation matrices in the file `parameters-CovarianceObservations.py` in the site directory.
- `self.icemarkers_correlation`     : for ice dated horizons
- `self.airmarkers_correlation`     : for air dated horizons
- `self.iceintervals_correlation`   : for ice dated intervals
- `self.airintervals_correlation`   : for air dated intervals
- `self.Ddepth_correlation`         : for Delta-depth observations

For site pair specific observations (stratigraphic links), you set up the correlation matrices in the file `parameters-CovarianceObservations.py` in the site pair directory.
- `self.iceicemarkers_correlation`  : for ice-ice stratigraphic links
- `self.airairmarkers_correlation`  : for air-air stratigraphic links
- `self.iceairmarkers_correlation`  : for ice-air stratigraphic links
- `self.airicemarkers_correlation`  : for air-ice stratigraphic links

Let us take a concrete example and assume we want a correlation matrix for ice dated horizons with ones in the diagonal and with a constant correlation factor k outside the diagonal, you can write:

```
self.icemarkers_correlation=k*np.ones((np.shape(self.icemarkers_correlation)))+(1-k)*np.diag(np.ones(np.shape(self.icemarkers_correlation)[0]))
```

Don't forget that if you find the use of python and the Paleochrono internal variables too difficult, you can define your correlation matrices outside paleochrono and import them here by using for example the `np.loadtxt` function.


What to do if something goes wrong?
-----------------------------------

Some errors can be eliminated by restarting the kernel in spyder (under "Console">"Restart kernel").
If the problem persist, please post an email to the author or on the mailing list with the error message appearing on the command line.

