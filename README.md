# x (UNFINISHED)

This repostory contains the source code to the paper "x". The paper compares different methods for the classification and parameter estimation of guitar effects in instrument mixes.

## Installation

Download the dataset from x

1. Install anaconda, if not already installed. Tested with anaconda 4.10.1. 
2. Create the necessary virtual environment by executing the env_install.bat script from an anaconda prompt.

## Get Started

### For the effect classification: 
1. Navigate to the Classification folder and execute the clf.bat script


### For the parameter estimation:
1. Navigate to the Parameter Estimation folder and execute the param_est.bat script

## Dataset Generation

The repository also contains the necessary scripts to generate the two new datasets via Reaper's ReaScript feature. The functionality of these scripts has only been tested on Windows 10.

### Installation

1. Install Reaper (Version 6.29 or higher).
2. Install all VST plugins from (((LIST)))).
3. Add the path to these plugins to the list of VST plugin paths in Reaper via Options > Preferences > Media > VST. Re-scan to make sure the plugins are available.
4. Open the presets folder revealed by clicking Options > Show Resource Path and move the 'vst-Ample Guitar LP' file included in this repository there. It contains the preset used for the Ample Guitar LP plugin.
5. Make sure Reaper outputs mono files with a sampling rate of 44.1 kHz and a bit depth of 16 bit via in the menu revealed by clicking File > Render. Uncheck the 'Tail' option.
6. Install Python 3.9.2 or higher and enable Python usage via Options > Preferences > Plug-Ins > ReaScript in Reaper.
7. Load the necessary scripts via Actions > Action list: Classification/dataset_classification, Classification/reaper_utility.py, Parameter Estimation/dataset_parameter_estimation and Parameter/Estimation/reaper_utility.py.
8. To generate the Classification dataset, run the dataset_classification.py script via the Actions list. For the Parameter Estimation dataset, run the dataset_parameter_estimation.py script.
