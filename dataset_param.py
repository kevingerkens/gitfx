from pathlib import Path
import math
import pickle
import os
import random

RPR_RENDER_PATH = Path('D:/NewDataset')
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

def add_fx(CurTr, fx_name, mult_toggle):
    if mult_toggle == False:
        RPR_TrackFX_Delete(CurTr, 1)        #clear fx slot, add plugin
    RPR_TrackFX_AddByName(CurTr, fx_name, False, -1)
    
def set_params(CurTr, param_vals, fx_slot):
    for index, param in enumerate(param_vals):
        RPR_TrackFX_SetParam(CurTr, fx_slot, index, param_vals[index])
        
def change_param(CurTr, param, val, fx, fx_slot):
    if (fx == 'SlapbackDelay' or fx == 'FeedbackDelay') and (param == 6 or param == 12):
        val = val * 0.5
    RPR_TrackFX_SetParam(CurTr, fx_slot, param, val)
    
def git_pitch(CurIt, pitch):
    CurTk = RPR_GetMediaItemTake(CurIt, 0)
    RPR_MIDI_SetNote(CurTk, 0, False, False, -1, -1, -1, pitch, -1, True)

def transpose_note(CurTr, interval):
    CurIt = RPR_GetTrackMediaItem(CurTr, 0)
    CurTk = RPR_GetMediaItemTake(CurIt, 0)
    CurPitch = RPR_MIDI_GetNote(CurTk, 0, 0, 0, 0, 0, 0, 0, 0)[8]
    RPR_MIDI_SetNote(CurTk, 0, False, False, -1, -1, -1, CurPitch+interval, -1, True)

def bass_piano_pitch(interval):
    for trackid in [1, 2]:
        CurTr = RPR_GetTrack(0, trackid)
        transpose_note(CurTr, interval)
    
def create_guitar(track_nr, plugin, pitch):
    RPR_InsertTrackAtIndex(track_nr, False)
    CurTr = RPR_GetTrack(0, track_nr)
    RPR_TrackFX_AddByName(CurTr, plugin, False, -1)
    #RPR_SetMediaTrackInfo_Value(CurTr, "D_VOL", 0.7) 
    CurIt = RPR_CreateNewMIDIItemInProj(CurTr, 0, 4, True)
    CurIt = CurIt[0]
    CurTk = RPR_GetMediaItemTake(CurIt, 0)
    RPR_MIDI_InsertNote(CurTk, False, False, 0, 3840, 1, pitch, 110, False)[0]
    
def create_bass(track_nr, pitch):
    RPR_InsertTrackAtIndex(track_nr, False)
    CurTr = RPR_GetTrack(0, track_nr)
    RPR_TrackFX_AddByName(CurTr, "ABPL2.dll", False, -1)
    CurIt = RPR_CreateNewMIDIItemInProj(CurTr, 0, 4, True)
    CurIt = CurIt[0]
    CurTk = RPR_GetMediaItemTake(CurIt, 0)
    RPR_MIDI_InsertNote(CurTk, False, False, 0, 3840, 1, pitch, 100, False)[0]
    
def create_keys(track_nr, pitch):
    RPR_InsertTrackAtIndex(track_nr, False)
    CurTr = RPR_GetTrack(0, track_nr)
    RPR_TrackFX_AddByName(CurTr, "Keyzone Classic.dll", False, -1)
    CurIt = RPR_CreateNewMIDIItemInProj(CurTr, 0, 4, True)
    CurIt = CurIt[0]
    CurTk = RPR_GetMediaItemTake(CurIt, 0)
    RPR_MIDI_InsertNote(CurTk, True, False, 0, 3840, 0, pitch, 127, False)[0]
    
def kick_drum(CurTk):
    for i in range(2):
        RPR_MIDI_InsertNote(CurTk, False, False, i*1920, i*1920+10, 1, 36, 100, False)
        
def snare(CurTk):
    for i in range(2):
        RPR_MIDI_InsertNote(CurTk, False, False, i*1920+960, i*1920+970, 1, 38, 100, False)
        
def hh_closed(CurTk):
    for i in range(8):
        RPR_MIDI_InsertNote(CurTk, False, False, i*480, i*480+10, 1, 42, 100, False)
        
def hh_open(CurTk):
    for i in range(8):
        RPR_MIDI_InsertNote(CurTk, False, False, i*480, i*480+10, 1, 46, 100, False)
        
def cymbal(CurTk):
    RPR_MIDI_InsertNote(CurTk, False, False, 0, 10, 1, 49, 100, False)
    
def create_drums(track_nr):
    RPR_InsertTrackAtIndex(track_nr, False)
    CurTr = RPR_GetTrack(0, track_nr)
    RPR_TrackFX_AddByName(CurTr, "MT-PowerDrumKit.dll", False, -1)
    CurIt = RPR_CreateNewMIDIItemInProj(CurTr, 0, 4, True)
    CurIt = CurIt[0]
    CurTk = RPR_GetMediaItemTake(CurIt, 0)
    return CurTk
    
def git_bass_keys_drums(pitch, plugin):
    create_guitar(0, plugin, pitch)
    create_bass(1, pitch)
    create_keys(2, pitch)
    CurTk = create_drums(3)
    kick_drum(CurTk)
    snare(CurTk)
    hh_closed(CurTk)    
    
def set_vol(vol):  # lower volume of other instruments
        track_count = RPR_CountTracks(0)
        for track in range(track_count-1):
          CurTr = RPR_GetTrack(0, (track+1))
          RPR_SetMediaTrackInfo_Value(CurTr, "D_VOL", vol)    

def render_file(mix, fx, setting, vol, pitch, bp_oct, plugin):  
    set_str = ''
    for par in setting:
        par_str = str(par)
        set_str += (par_str + '_')
    str_bp = 'bp' + str(bp_oct)
    str_vol = str(-3+vol*3)
    file_name = plugin + '_' + str(mix) + '_' +  str(fx)  + '_' + set_str + str(pitch) + '_' + str_bp + '_' + str_vol
    RPR_GetSetProjectInfo_String(0, "RENDER_FILE", str(RPR_RENDER_PATH) + '\\' + str(mix) + '\\' + str(fx), 1)
    RPR_GetSetProjectInfo_String(0, "RENDER_PATTERN",  str(file_name), 1)
    #data_file = str(RPR_RENDER_PATH) + '\\' + str(fx) + '\\' + str(file_name) + '.pickle'
    #with open(data_file, 'wb') as handle:
    #    pickle.dump(data, handle)
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
             'Reverb' : ['Room Size', 'Mix'],
             'SlapbackDelay' : ['Time', 'Mix'],
             'Tremolo' : ['Depth', 'Frequency'],
             'Vibrato' : ['Rate', 'Depth'],
             }
    
    par_name = names[fx][index]
    return par_name

def get_fx_vals(fx):
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

def get_var_params(fx):
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

def get_folder(fx):
    folders = ['Distortion', 'SlapbackDelay', 'Tremolo']
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

def fx_one(fx_list, vol_list, mix_name, pitch, plugin):
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
                render_file(mix_name, folder, setting, vol_exp, pitch, 0, plugin)
                if mix_name != 'G':
                    bass_piano_pitch(12)
                    render_file(mix_name, folder, setting, vol_exp, pitch, 12, plugin)
                    bass_piano_pitch(-12)

def fx_mult(fx_list, vol_list, mix_name, pitch, plugin):
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
        if combo == [True, True, True]:
            n_samples = 1200
        else:
            n_samples = 800
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
    for i in range(1, 4):
        RPR_TrackFX_Delete(GitTr, i)
    
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
            fx_one(fx_list, vol_list, mix_name, pitch_list[0], plugin)
            transpose_note(GitTr, pitch_list[1]-pitch_list[0])
            fx_one(fx_list, vol_list, mix_name, pitch_list[1])
            RPR_TrackFX_Delete(GitTr, 1)
            fx_mult(fx_list, vol_list, mix_name, pitch_list[0], plugin)
            transpose_note(GitTr, pitch_list[1]-pitch_list[0])
            fx_mult(fx_list, vol_list, mix_name, pitch_list[1])
            track_count = RPR_CountTracks(0)
            for _ in range(track_count):
                CurTr = RPR_GetTrack(0, 0)
                RPR_DeleteTrack(CurTr)
        RPR_PreventUIRefresh(-1)
        RPR_ShowConsoleMsg('\nDone\n')
                
process_files()



    


