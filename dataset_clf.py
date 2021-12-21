from pathlib import Path
import pickle
import os
import random

RPR_RENDER_PATH = Path('C:/Desktop/Uni/Studienarbeit/Audio/MixRand/Gitarre monophon')
DIST_FX_SLOT = 1
MOD_FX_SLOT = 2
DLY_FX_SLOT = 3
PITCH = 40
PARAM_STEPS = 21 

git_plugin_dict = {
    'ag': 'Ample Guitar LP.dll',
    'sv': 'BrightElectricGuitar_x64.dll'
}

def add_fx(CurTr, fx_name):
    RPR_TrackFX_Delete(CurTr, 1)        #clear fx slot, add plugin
    RPR_TrackFX_AddByName(CurTr, fx_name, False, -1)
    
def set_params(CurTr, param_vals):
    for index, param in enumerate(param_vals):
        RPR_TrackFX_SetParam(CurTr, DIST_FX_SLOT, index, param_vals[index])
        
def change_param(CurTr, param, fx):
    val = random.uniform(0.0, 1.0)
    og_val = val
    if (fx == 2 or fx == 8) and (param == 6 or param == 12):
        val = val * 0.5
    if fx == 3 and param == 3:
        val = (val + 1) * 0.5    
    RPR_TrackFX_SetParam(CurTr, DIST_FX_SLOT, param, val)
    return val, og_val
    
def git_pitch(CurIt, pitch):
    CurTk = RPR_GetMediaItemTake(CurIt, 0)
    
    RPR_MIDI_SetNote(CurTk, 0, False, False, -1, -1, -1, pitch, -1, True)
    
def create_guitar(track_nr, plugin):
    RPR_InsertTrackAtIndex(track_nr, False)
    CurTr = RPR_GetTrack(0, track_nr)
    RPR_TrackFX_AddByName(CurTr, plugin, False, -1)
    CurTr = RPR_GetTrack(0, 0)
    CurIt = RPR_CreateNewMIDIItemInProj(CurTr, 0, 4, True)
    CurIt = CurIt[0]
    CurTk = RPR_GetMediaItemTake(CurIt, 0)
    RPR_MIDI_InsertNote(CurTk, False, False, 0, 3840, 1, PITCH+12, 110, False)[0]
    
def create_bass(track_nr):
    RPR_InsertTrackAtIndex(track_nr, False)
    CurTr = RPR_GetTrack(0, track_nr)
    RPR_TrackFX_AddByName(CurTr, "ABPL2.dll", False, -1)
    CurIt = RPR_CreateNewMIDIItemInProj(CurTr, 0, 4, True)
    CurIt = CurIt[0]
    CurTk = RPR_GetMediaItemTake(CurIt, 0)
    RPR_MIDI_InsertNote(CurTk, False, False, 0, 3840, 1, PITCH, 100, False)[0]
    
def create_keys(track_nr):
    RPR_InsertTrackAtIndex(track_nr, False)
    CurTr = RPR_GetTrack(0, track_nr)
    RPR_TrackFX_AddByName(CurTr, "Keyzone Classic.dll", False, -1)
    CurIt = RPR_CreateNewMIDIItemInProj(CurTr, 0, 4, True)
    CurIt = CurIt[0]
    CurTk = RPR_GetMediaItemTake(CurIt, 0)
    RPR_MIDI_InsertNote(CurTk, True, False, 0, 3840, 0, PITCH, 127, False)[0]
    
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
    
def git_solo(plugin):
    create_guitar(0, plugin)
    
def git_bass(plugin):
    create_guitar(0, plugin)
    create_bass(1)
    #set_vol(1, 0.125)
    
    
def git_keys(plugin):
    create_guitar(0, plugin)
    create_keys(1)
    #set_vol(1, 0.25)
    
def git_kick(plugin):
    create_guitar(0, plugin)
    CurTk = create_drums(1)
    kick_drum(CurTk)
    #set_vol(1, 0.125)

def git_snare(plugin):
    create_guitar(0, plugin)
    CurTk = create_drums(1)
    snare(CurTk)
    #set_vol(1, 0.125)
    
def git_hh_closed(plugin):
    create_guitar(0, plugin)
    CurTk = create_drums(1)
    hh_closed(CurTk)
    #set_vol(1, 0.125)
    
def git_hh_open(plugin):
    create_guitar(0, plugin)
    CurTk = create_drums(1)
    hh_open(CurTk)
    #set_vol(1, 0.125)
    
def git_cymbal(plugin):
    create_guitar(0, plugin)
    CurTk = create_drums(1)
    cymbal(CurTk)
    #set_vol(1, 0.125)
    
def git_bass_drums(plugin):
    create_guitar(0, plugin)
    create_bass(1)
    #set_vol(1, 0.125)
    CurTk = create_drums(2)
    #set_vol(2, 0.125)
    kick_drum(CurTk)
    snare(CurTk)
    hh_closed(CurTk)
    
def git_keys_drums(plugin):
    create_guitar(0, plugin)
    create_keys(1)
    #set_vol(1, 0.125)
    CurTk = create_drums(2)
    #set_vol(2, 0.125)
    kick_drum(CurTk)
    snare(CurTk)
    hh_closed(CurTk)
    
def git_bass_keys(plugin):
    create_guitar(0, plugin)
    create_bass(1)
    #set_vol(1, 0.125)
    create_keys(2)
    #set_vol(2, 0.25)
    
def git_bass_keys_drums(plugin):
    create_guitar(0, plugin)
    create_bass(1)
    #set_vol(1, 0.125)
    create_keys(2)
    #set_vol(2, 0.25)
    CurTk = create_drums(3)
    #set_vol(3, 0.125)
    kick_drum(CurTk)
    snare(CurTk)
    hh_closed(CurTk)    
    
def set_vol():  # lower volume of other instruments
        track_count = RPR_CountTracks(0)
        for track in range(track_count-1):
          CurTr = RPR_GetTrack(0, (track+1))
          RPR_SetMediaTrackInfo_Value(CurTr, "D_VOL", 0.15)    
    
    
def render_file(mix, fx, index, new_vals, plugin):
    data = [mix, fx] # new_params, par_name, PITCH]
    for val in new_vals:
        data.append(val)
    data.append(PITCH)
    #prepare render info
    file_name = plugin + '_' + str(mix) + '_' +  str(fx) + str(PITCH) + '_' + str(index) + '_' 
    RPR_GetSetProjectInfo_String(0, "RENDER_FILE", str(RPR_RENDER_PATH) + '\\' + str(fx), 1)
    RPR_GetSetProjectInfo_String(0, "RENDER_PATTERN",  str(file_name), 1)
    data_file = str(RPR_RENDER_PATH) + '\\' + str(fx) + '\\' + str(file_name) + '.pickle'
    with open(data_file, 'wb') as handle:
        pickle.dump(data, handle)
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

def get_var_params(fx):
    params =       {'Chorus': [2, 3],               #chorus depth, rate
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
    var_params = params[fx]
    return var_params

def get_folder(fx):
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
    mix_list = ('G', 'G+B', 'G+K', 'G+HHC', 'G+KD', 'G+HHO''G+KD', 'G+C', 'G+SD', 'G+B+D', 'G+K+D', 'G+K+B', 'G+K+B+D') #list of isntrument combinations   
    fx_list = ["Classic Chorus.dll", "OctBUZ.dll", "Classic Delay.dll", "Classic Flanger.dll", "Rednef Twin.dll", 
                "Tube Screamer.dll", "Classic Phaser.dll", "Classic Reverb.dll", "Classic Delay.dll", "PechenegTremolo.dll", "Classic Chorus.dll"]

    for plugin in list(git_plugin_dict.keys()):                
        for mix_id, mix in enumerate(mix_list):
            mix_dict[mix](git_plugin_dict[plugin])
            set_vol()
            for fx in range(len(fx_list)):
                folder = get_folder(fx)
                GitTr = RPR_GetTrack(0,0)
                add_fx(GitTr, fx_list[fx])
                var_count = len(get_var_params(folder))
                set_params(GitTr, get_fx_vals(folder))
                for i in range(60):
                    new_vals = []
                    for index in get_var_params(folder):
                        _, new_param_val = change_param(GitTr, index, folder)                     #sweep through param while keeping other params constant)    
                        new_vals.append(new_param_val)               
                    render_file(mix_list[mix_id], folder, i, new_vals, plugin)
            track_count = RPR_CountTracks(0)
            for _  in range(track_count):
                CurTr = RPR_GetTrack(0, 0)
                RPR_DeleteTrack(CurTr)
        RPR_PreventUIRefresh(-1)
        RPR_ShowConsoleMsg('\nDone\n')
                
process_files()


    


