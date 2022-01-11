"""ReaScript Documentation available at https://www.reaper.fm/sdk/reascript/reascripthelp.html.
This script generates the GEPE_GIM dataset for effect parameter estimation"""

from pathlib import Path
import math
import pickle
import os
import random
from reaper_utility import *

script_directory = os.path.dirname(os.path.abspath(__file__))
file_directory = os.path.join(script_directory, '../..', 'Datasets/GEPE-GIM')

RPR_RENDER_PATH = Path(file_directory)
PARAM_STEPS = 20
DIST_FX_SLOT = 1
TREM_FX_SLOT = 2
DLY_FX_SLOT = 3
THREE_DB_FACTOR = 2**0.5
OG_VOL = 0.25/(THREE_DB_FACTOR)

git_plugin_dict = {
    'ag': 'Ample Guitar LP.dll',
    'sv': 'BrightElectricGuitar_x64.dll'
    }
    

def render_file(mix, fx, setting, vol, pitch, bp_oct, plugin):  
    set_str = ''
    for par in setting:
        par_str = str(par)
        set_str += (par_str + '_')
    str_bp = 'bp' + str(bp_oct)
    str_vol = str(-3+vol*3)
    file_name = plugin + '_' + str(mix) + '_' +  str(fx)  + '_' + set_str + str(pitch) + '_' + str_bp + '_' + str_vol
    RPR_GetSetProjectInfo_String(0, "RENDER_FILE", str(RPR_RENDER_PATH) +  '\\' + str(fx), 1)
    RPR_GetSetProjectInfo_String(0, "RENDER_PATTERN",  str(file_name), 1)
    #data_file = str(RPR_RENDER_PATH) + '\\' + str(fx) + '\\' + str(file_name) + '.pickle'
    #with open(data_file, 'wb') as handle:
    #    pickle.dump(data, handle)
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
             'Reverb' : ['Room Size', 'Mix'],
             'SlapbackDelay' : ['Time', 'Mix'],
             'Tremolo' : ['Depth', 'Frequency'],
             'Vibrato' : ['Rate', 'Depth'],
             }
    
    par_name = names[fx][index]
    return par_name

def get_default_fx_values(fx):
    vals =      {'Chorus' : [0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0],                                #standard fx parameters per fx type
                 'Distortion' : [0.0, 0.0, 0.0, 0.8, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 0.0], 
                 'FeedbackDelay' : [0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.7, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5],
                 'Flanger' : [1.0, 0.5, 0.5, 0.7, 0.5, 0.8, 0.0, 0.0],
                 'NoFX' : [1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
                 'Overdrive' : [0.7, 0.5, 0.5],
                 'Phaser' : [0.0, 1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.8],
                 'Reverb' : [0.7, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5],
                 'Tremolo' : [0.5, 0.7, 0.1, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
                 'SlapbackDelay' : [0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5],
                 'Vibrato' : [0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.0]
                 }
    fx_vals = vals[fx]
    return fx_vals

def get_variable_parameters(fx):
    params =       {'Chorus': [2, 3],               #chorus depth, rate
                    'Distortion': [3, 9],        #dist edge, gain, tone
                    'FeedbackDelay': [6, 7, 12],    #fb delay time, feedback, mix
                    'Flanger': [1, 2, 3],           #flanger depth, rate, feedback
                    'NoFX': [2, 3, 4],              #no fx bass, mid, treble
                    'Overdrive': [0, 1],            #od gain, tone
                    'Phaser': [2, 4],               #phaser depth, rate
                    'Reverb': [0, 6],                  #reverb size
                    'Tremolo': [1, 2], 
                    'SlapbackDelay': [6, 12],         #sb delay time, mix
                    'Vibrato': [2, 3]               #vibrato depth, rate
                    }
    var_params = params[fx]
    return var_params

def get_folder_name(fx):
    folders = ['Distortion', 'SlapbackDelay', 'Tremolo']
    folder = folders[fx]
    return folder

def create_2parameter_grid():
    param0 = [*range(1, PARAM_STEPS+1, 1)]
    param1 = param0
    settings = []
    for i in param0:
        for j in param1: 
            settings.append([i/PARAM_STEPS, j/PARAM_STEPS])
    return settings

def create_3parameter_grid():
    param0 = [*range(1, PARAM_STEPS+1, 1)]
    param1 = param0
    param2 = param0
    settings = []
    for i in param0:
        for j in param1:
            for k in param2:
                settings.append([i/PARAM_STEPS, j/PARAM_STEPS, k/PARAM_STEPS])
    return settings

def get_settings(var_count):
    if var_count == 1:
        settings = [*range(1, PARAM_STEPS+1, 1)]
    if var_count == 2:
        settings = create_2parameter_grid()
    if var_count == 3:
        settings = create_3parameter_grid()
    return settings

def toggle_fx(GitTr, fx_onoff):
    RPR_TrackFX_SetEnabled(GitTr, DIST_FX_SLOT, fx_onoff[0])
    RPR_TrackFX_SetEnabled(GitTr, TREM_FX_SLOT, fx_onoff[1])
    RPR_TrackFX_SetEnabled(GitTr, DLY_FX_SLOT, fx_onoff[2])

def get_fx_slot(fx):
    slots = {'Distortion': DIST_FX_SLOT,
             'Tremolo': TREM_FX_SLOT,
             'SlapbackDelay': DLY_FX_SLOT
    }
    return slots[fx]

def single_fx(fx_list, vol_list, mix_name, pitch, plugin):
    for fx in range(len(fx_list)):
        folder = get_folder_name(fx)
        GitTr = RPR_GetTrack(0,0)
        insert_fx(GitTr, fx_list[fx], mult_toggle = False)
        var_par = get_variable_parameters(folder)
        var_count = len(var_par)
        set_params(GitTr, get_default_fx_values(folder), DIST_FX_SLOT)
        for setting in get_settings(var_count):
            for index, par_val in enumerate(setting):                       
                change_param(GitTr, var_par[index], par_val, folder, 1)
            for vol_exp in vol_list:
                vol = OG_VOL*(THREE_DB_FACTOR**vol_exp)
                set_vol(vol)
                render_file(mix_name, folder, setting, vol_exp, pitch, 0, plugin)
                if mix_name != 'G':
                    bass_piano_pitch(12)
                    render_file(mix_name, folder, setting, vol_exp, pitch, 12, plugin)
                    bass_piano_pitch(-12)

def set_number_of_samples(combo):
    if combo == [True, True, True]:
        n_samples = 1200
    else:
        n_samples = 800
    return n_samples

def delete_all_track_fx(track):
    fx_count = RPR_TrackFX_GetCount(track)
    for fx in range(1, fx_count):
        RPR_TrackFX_Delete(track, fx)

def multi_fx(fx_list, vol_list, mix_name, pitch, plugin):
    fx_combos = [[True, True, False], [True, False, True], [False, True, True], [True, True, True]]  
    GitTr = RPR_GetTrack(0,0)  
    for slot, fx in enumerate(fx_list):
        add_fx(GitTr, fx, mult_toggle=True)
        CurFx = get_folder(slot)
        set_params(GitTr, get_fx_vals(CurFx), slot+1) 
    for combo in fx_combos:
        fx_active = []
        for fx_id,i in enumerate(combo):
            if i==True:
                CurFx = get_folder(fx_id)
                fx_active.append(CurFx) 
        folder = ''.join(fx_active)
        toggle_fx(GitTr, combo)
        n_samples = set_number_of_samples(combo)
        for _ in range(n_samples):
            setting = []
            for cur_fx in fx_active:
                var_par = get_var_params(cur_fx)
                vals = []
                for par in var_par:
                    val = round(random.random(), 2)
                    change_param(GitTr, par, val, cur_fx, get_fx_slot(cur_fx))
                    vals.append(val)
                setting.extend(vals)
            for vol_exp in vol_list:
                vol = OG_VOL*(THREE_DB_FACTOR**vol_exp)
                set_vol(vol)
                render_file(mix_name, folder, setting, vol_exp, pitch, 0, plugin)
                if mix_name != 'G':
                    bass_piano_pitch(12)
                    render_file(mix_name, folder, setting, vol_exp, pitch, 12, plugin)
                    bass_piano_pitch(-12)
    delete_all_track_fx(GitTr)
    
def process_files():
    RPR_PreventUIRefresh(1)
    mix_dict = {
    'G+K+B+D' : [git_bass_keys_drums, [-11, -7, -3, -1, 0, 1, 2]] 
    }
    mix_list = ['G+K+B+D']
    fx_list =  ["OctBUZ.dll", "PechenegTremolo.dll", "Classic Delay.dll"] #"OctBUZ.dll",, "Classic Flanger.dll", "Rednef Twin.dll", "Tube Screamer.dll", "Classic Phaser.dll", "Classic Reverb.dll", "Classic Delay.dll", "PechenegTremolo.dll", "Classic Chorus.dll"]
    pitch_list = [40, 52]       

    for plugin in list(git_plugin_dict.keys()):
        for mix_id, mix in enumerate(mix_list):
            mix_dict[mix][0](pitch_list[0], git_plugin_dict[plugin])
            vol_list = mix_dict[mix][1]
            mix_name = mix_list[mix_id]
            GitTr = RPR_GetTrack(0, 0)
            single_fx(fx_list, vol_list, mix_name, pitch_list[0], plugin)
            transpose_note(GitTr, pitch_list[1]-pitch_list[0])
            single_fx(fx_list, vol_list, mix_name, pitch_list[1])
            RPR_TrackFX_Delete(GitTr, 1)
            multi_fx(fx_list, vol_list, mix_name, pitch_list[0], plugin)
            transpose_note(GitTr, pitch_list[1]-pitch_list[0])
            multi_fx(fx_list, vol_list, mix_name, pitch_list[1])
            track_count = RPR_CountTracks(0)
            for _ in range(track_count):
                CurTr = RPR_GetTrack(0, 0)
                RPR_DeleteTrack(CurTr)
        RPR_PreventUIRefresh(-1)
        RPR_ShowConsoleMsg('\nDone\n')
                
process_files()



    


