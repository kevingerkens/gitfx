# Audio Effect Extraction from Realistic Instrument Mixes

This repository contains the source code to the paper [1]. The paper compares different methods for the classification and parameter estimation of guitar effects in instrument mixes. The code has been tested with Windows 10.

## Installation

1. Install anaconda, if not already installed. Tested with anaconda 4.10.1. 
2. Create the necessary virtual environment by navigating to the Code directory and executing the env_install.bat script from an anaconda prompt.

## Get Started

### For the effect classification: 
1. Download the GEC-GIM dataset from https://seafile.cloud.uni-hannover.de/u/d/1754cd6cefe94c798e0f/, extract the .zip file and put the effects folders (Chorus, Distortion etc.) into the Datasets/GEC-GIM directory of this repository.
2. Download the IDMT dataset from https://www.idmt.fraunhofer.de/en/publications/datasets/audio_effects.html#:~:text=The%20IDMT%2DSMT%2DAudio%2D,30%20hours., extract the .zip file and move the effects folders contained in the 'Gitarre monophon/Samples' folder to the Datasets/IDMT directory of this repository.
3. Navigate to the Code/Classification directory of this repository.
4. Execute the clf.bat script.


### For the parameter estimation:
1. Download the Parameter Estimation and Parameter Estimation Pitch Changes datasets from https://seafile.cloud.uni-hannover.de/u/d/1754cd6cefe94c798e0f/, extract the zip files and move the effects folders (Distortion, Tremolo, etc.) to the Datasets/GEPE-GIM and datasets/GEPE-GIM Pitch Changes directories of this repository respectively.
2. Navigate to the Code/Parameter Estimation directory of this repository.
3. Execute the param_est.bat script.

## Dataset Generation

The repository also contains the necessary scripts to generate the two new datasets via Reaper's ReaScript feature. The functionality of these scripts has only been tested on Windows 10.

### Installation

1. Install Reaper (Version 6.29 or higher).
2. Install all VST plugins listed in plugins.txt into the same folder. Alternatively, you can download them as a bundle from https://seafile.cloud.uni-hannover.de/u/d/cde6f8b254cd4555a09b/.
3. Add the path to these plugins to the list of VST plugin paths in Reaper via Options > Preferences > Media > VST. Re-scan to make sure the plugins are available.
4. Open the presets folder revealed by clicking Options > Show Resource Path and move the 'vst-Ample Guitar LP' file included in this repository there. It contains the preset used for the Ample Guitar LP plugin.
5. Make sure Reaper outputs mono files with a sampling rate of 44.1 kHz and a bit depth of 16 bit via in the menu revealed by clicking File > Render. Uncheck the 'Tail' option.
6. Install Python 3.9.2 or higher and enable Python usage via Options > Preferences > Plug-Ins > ReaScript in Reaper.
7. Load the necessary scripts via Actions > Action list: Classification/dataset_classification, Classification/reaper_utility.py, Parameter Estimation/dataset_parameter_estimation and Parameter/Estimation/reaper_utility.py.
8. To generate the Classification dataset, run the dataset_classification.py script via the Actions list. For the Parameter Estimation dataset, run the dataset_parameter_estimation.py script.



[1] Gerkens, K., Hinrichs, R., Ostermann, J.: Proceedings of the 11th International Conference 2022, Held as Part of EvoStar 2022, April 20th to 22nd 2022, Seville, Spain. (submitted) 

