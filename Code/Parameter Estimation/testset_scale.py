"""ReaScript Documentation available at https://www.reaper.fm/sdk/reascript/reascripthelp.html.
This script generates the test dataset with bass and piano pitch changes for parameter estimation."""

from pathlib import Path
import math
import pickle
import os
import random
from reaper_utility import *

RPR_RENDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'Datasets/GEPE-GIM Pitch Changes')
PARAM_STEPS = 20
DIST_FX_SLOT = 1
THREE_DB_FACTOR = 2**0.5
OG_VOL = 0.25/(THREE_DB_FACTOR)


def render_file(mix, fx, setting, vol, pitch, bp_oct):  
    set_str = ''
    for par in setting:
        par_str = str(par)
        set_str += (par_str + '_')
    str_bp = 'bp' + str(bp_oct)
    str_vol = str(-3+vol*3)
    #prepare render info
    file_name = 'ag' + '_' + str(mix) + '_' +  str(fx)  + '_' + set_str + str(pitch) + '_' + str_bp + '_' + str_vol
    RPR_GetSetProjectInfo_String(0, "RENDER_FILE", str(RPR_RENDER_PATH) + '\\' + str(mix) + '\\' + str(fx), 1)
    RPR_GetSetProjectInfo_String(0, "RENDER_PATTERN",  str(file_name), 1)
    RPR_Main_OnCommand(40108, 0) #Normalize
    RPR_Main_OnCommand(41824, 0) #Render
    RPR_SetEditCurPos(0.0, True, False)

def get_param_name(fx, index):
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

def get_fx_vals(fx):
    vals =      {#'Chorus' : [0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0],                                #standard fx parameters per fx type
                 'Distortion' : [0.0, 0.0, 0.0, 0.8, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 0.0], 
                 #'FeedbackDelay' : [0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.7, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5],
                 #'Flanger' : [1.0, 0.5, 0.5, 0.7, 0.5, 0.8, 0.0, 0.0],
                 #'NoFX' : [1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
                 #'Overdrive' : [0.7, 0.5, 0.5],
                 #'Phaser' : [0.0, 1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.8],
                 #'Reverb' : [0.7, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5],
                 'Tremolo' : [0.5, 0.7, 0.1, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
                 'SlapbackDelay' : [0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5]
                 #'Vibrato' : [0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.0]
                 }
    fx_vals = vals[fx]
    return fx_vals

def get_var_params(fx):
    params =       {#'Chorus': [2, 3],               #chorus depth, rate
                    'Distortion': [3, 9],        #dist edge, gain, tone
                    #'FeedbackDelay': [6, 7, 12],    #fb delay time, feedback, mix
                    #'Flanger': [1, 2, 3],           #flanger depth, rate, feedback
                    #'NoFX': [2, 3, 4],              #no fx bass, mid, treble
                    #'Overdrive': [0, 1],            #od gain, tone
                    #'Phaser': [2, 4],               #phaser depth, rate
                    #'Reverb': [0],                  #reverb size
                    'Tremolo': [1, 2], 
                    'SlapbackDelay': [6, 12]         #sb delay time, mix
                    #tremolo depth, rate
                    #'Vibrato': [2, 3]               #vibrato depth, rate
                    }
    var_params = params[fx]
    return var_params

def get_folder(fx):
    folders = ['Distortion', 'Tremolo', 'SlapbackDelay']
    folder = folders[fx]
    return folder

def set_2params():
    param0 = [*range(1, PARAM_STEPS+1, 1)]
    param1 = param0
    settings = []
    for i in param0:
        for j in param1: 
            settings.append([i/PARAM_STEPS, j/PARAM_STEPS])
    return settings

def set_3params():
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
        settings = set_2params()
    if var_count == 3:
        settings = set_3params()
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

def single_fx(fx_list, vol_list, mix_name, pitch):
    for fx in range(len(fx_list)):
        folder = get_folder(fx)
        GitTr = RPR_GetTrack(0,0)
        add_fx(GitTr, fx_list[fx], mult_toggle = False)
        var_par = get_var_params(folder)
        var_count = len(var_par)
        set_params(GitTr, get_fx_vals(folder), DIST_FX_SLOT)
        for setting in get_settings(var_count):
            for index, par_val in enumerate(setting):                       
                change_param(GitTr, var_par[index], par_val, folder, 1)
            for vol_exp in vol_list:
                vol = OG_VOL*(THREE_DB_FACTOR**vol_exp)
                set_vol(vol)
                render_file(mix_name, folder, setting, vol_exp, pitch, 0)
                for bp in [*range(1, 12)]:
                    bass_piano_pitch(1)
                    render_file(mix_name, folder, setting, vol_exp, pitch, bp)
                bass_piano_pitch(-11)
                s

def delete_all_tracks():
    track_count = RPR_CountTracks(0)
    for _ in range(track_count):
        CurTr = RPR_GetTrack(0, 0)
        RPR_DeleteTrack(CurTr)

    
def process_files():
    RPR_PreventUIRefresh(1)
    mix_dict = {
    'G+K+B+D' : [git_bass_keys_drums, [-3, 0, 1]] #-11, -7, -3, -1, 0, 1, 
    }

    mix_list = ['G+K+B+D']
    fx_list = ["OctBUZ.dll", "PechenegTremolo.dll", "Classic Delay.dll"]
    pitch_list = [40]              
    
    for mix_id, mix in enumerate(mix_list):
        mix_dict[mix][0](pitch_list[0])
        vol_list = mix_dict[mix][1]
        mix_name = mix_list[mix_id]
        GitTr = RPR_GetTrack(0, 0)
        single_fx(fx_list, vol_list, mix_name, pitch_list[0])
        track_count = RPR_CountTracks(0)
        delete_all_tracks()
    RPR_PreventUIRefresh(-1)
    RPR_ShowConsoleMsg('\nDone\n')
                
process_files()



    


