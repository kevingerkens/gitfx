"""ReaScript Documentation available at https://www.reaper.fm/sdk/reascript/reascripthelp.html.
This script generates the GEC-GIM dataset for guitar effect classification."""

from pathlib import Path
import pickle
import os
import random
from reaper_utility import *

script_directory = os.path.dirname(__file__)
file_directory = os.path.join(script_directory, '../..', 'Datasets/GEC-GIM')

RPR_RENDER_PATH = file_directory
DIST_FX_SLOT = 1
MOD_FX_SLOT = 2
DLY_FX_SLOT = 3
PITCH = 40
PARAM_STEPS = 21 

git_plugin_dict = {
    'ag': 'Ample Guitar LP.dll',
    'sv': 'BrightElectricGuitar_x64.dll'
}  
    
    
def render_file(mix, fx, index, new_vals, plugin):
    pickle_data = [mix, fx] 
    for val in new_vals:
        pickle_data.append(val)
    pickle_data.append(PITCH)
    #prepare render info
    file_name = plugin + '_' + str(mix) + '_' +  str(fx) + str(PITCH) + '_' + str(index) + '_' 
    RPR_GetSetProjectInfo_String(0, "RENDER_FILE", str(RPR_RENDER_PATH) + '\\' + str(fx), 1)
    RPR_GetSetProjectInfo_String(0, "RENDER_PATTERN",  str(file_name), 1)
    data_file = str(RPR_RENDER_PATH) + '\\' + str(fx) + '\\' + str(file_name) + '.pickle'
    with open(data_file, 'wb') as handle:
        pickle.dump(pickle_data, handle)
    RPR_Main_OnCommand(40108, 0) #Normalize
    RPR_Main_OnCommand(41824, 0) #Render
    RPR_SetEditCurPos(0.0, True, False)

def get_parameter_name(fx, index):
    names = {'Chorus' : ['Rate', 'Depth'],
             'Distortion' : ['Gain', 'Tone'],
             'FeedbackDelay' : ['Time', 'Feedback', 'Mix'],
             'Flanger' : ['Rate', 'Depth', 'Feedback'],
             'NoFX' : ['Low', 'Mid', 'High'],
             'Overdrive' : ['Gain', 'Tone'],
             'Phaser' : ['Rate', 'Depth'],
             'Reverb' : ['Room Size'],
             'SlapbackDelay' : ['Time', 'Mix'],
             'Tremolo' : ['Depth', 'Frequency'],
             'Vibrato' : ['Rate', 'Depth'],
             }
    
    par_name = names[fx][index]
    return par_name

def get_fx_default_values(fx):
    vals =      {'Chorus' : [0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0],                                #standard fx parameters per fx type
                 'Distortion' : [0.0, 0.0, 0.5, 0.8, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 0.0], 
                 'FeedbackDelay' : [0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.7, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5],
                 'Flanger' : [1.0, 0.5, 0.5, 0.7, 0.5, 0.8, 0.0, 0.0],
                 'NoFX' : [1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
                 'Overdrive' : [0.7, 0.5, 0.5],
                 'Phaser' : [0.0, 1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.8],
                 'Reverb' : [0.7, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5],
                 'SlapbackDelay' : [0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5],
                 'Tremolo' : [0.5, 0.7, 0.1, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
                 'Vibrato' : [0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.0]
                 }
    fx_vals = vals[fx]
    return fx_vals

def get_variable_parameters(fx):
    parameters =    {'Chorus': [2, 3],               #chorus depth, rate
                    'Distortion': [3, 9],        #dist edge, gain, tone
                    'FeedbackDelay': [6, 7, 12],    #fb delay time, feedback, mix
                    'Flanger': [1, 2, 3],           #flanger depth, rate, feedback
                    'NoFX': [2, 3, 4],              #no fx bass, mid, treble
                    'Overdrive': [0, 1],            #od gain, tone
                    'Phaser': [2, 4],               #phaser depth, rate
                    'Reverb': [0],                  #reverb size
                    'SlapbackDelay': [6, 12],       #sb delay time, mix
                    'Tremolo': [1, 2],              #tremolo depth, rate
                    'Vibrato': [2, 3]               #vibrato depth, rate
                    }
    variable_parameters = parameters[fx]
    return variable_parameters

def get_folder_name(fx):
    folders = ['Chorus', 'Distortion', 'FeedbackDelay', 'Flanger', 'NoFX', 'Overdrive', 'Phaser', 'Reverb', 'SlapbackDelay', 'Tremolo', 'Vibrato']
    folder = folders[fx]
    return folder
    
def process_files():
    RPR_PreventUIRefresh(1)
    mix_dict = {
        'G' : git_solo,
        'G+B' : git_bass,
        'G+K' : git_keys,
        'G+HHC' : git_hh_closed,
        'G+HHO' : git_hh_open,
        'G+SD' : git_snare,
        'G+KD' : git_kick,
        'G+C' : git_cymbal,
        'G+B+D' : git_bass_drums,
        'G+K+D' : git_keys_drums,
        'G+K+B' : git_bass_keys,
        'G+K+B+D' : git_bass_keys_drums
     }
    mix_names = ('G', 'G+B', 'G+K', 'G+HHC', 'G+KD', 'G+HHO''G+KD', 'G+C', 'G+SD', 'G+B+D', 'G+K+D', 'G+K+B', 'G+K+B+D')  
    fx_list = ["Classic Chorus.dll", "OctBUZ.dll", "Classic Delay.dll", "Classic Flanger.dll", "Rednef Twin.dll", 
                "Tube Screamer.dll", "Classic Phaser.dll", "Classic Reverb.dll", "Classic Delay.dll", "PechenegTremolo.dll", "Classic Chorus.dll"]

    for plugin in list(git_plugin_dict.keys()):                
        for mix_id, mix in enumerate(mix_names):
            mix_dict[mix](git_plugin_dict[plugin])
            set_vol()
            for fx in range(len(fx_list)):
                folder = get_folder_name(fx)
                GitTr = RPR_GetTrack(0,0)
                insert_fx(GitTr, fx_list[fx], multi_toggle=False)
                set_default_parameter_values(GitTr, get_fx_default_values(folder))
                for i in range(60):
                    new_values = []
                    for index in get_variable_parameters(folder):
                        _, new_parameter_value = change_parameter_value(GitTr, index, folder)            
                        new_values.append(new_parameter_value)               
                    render_file(mix_names[mix_id], folder, i, new_values, plugin)
            delete_all_tracks()
        RPR_PreventUIRefresh(-1)
        RPR_ShowConsoleMsg('\nDone\n')
                
process_files()


    


