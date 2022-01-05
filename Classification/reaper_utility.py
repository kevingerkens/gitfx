"""ReaScript Documentation available at https://www.reaper.fm/sdk/reascript/reascripthelp.html.
This script containts Reaper utility functions used in both scripts for dataset generation"""

from reaper_python import *
import os
import random
import pickle
from pathlib import Path

DIST_FX_SLOT = 1
MOD_FX_SLOT = 2
DLY_FX_SLOT = 3
PITCH = 40


def delete_all_tracks():
    track_count = RPR_CountTracks(0)
    for _  in range(track_count):
        CurTr = RPR_GetTrack(0, 0)
        RPR_DeleteTrack(CurTr)

def insert_fx(CurTr, fx_name, multi_toggle):
    if multi_toggle == False:
        RPR_TrackFX_Delete(CurTr, 1)        #clear fx slot, add plugin
    RPR_TrackFX_AddByName(CurTr, fx_name, False, -1)
    
def set_default_parameter_values(CurTr, param_vals):
    for index, param in enumerate(param_vals):
        RPR_TrackFX_SetParam(CurTr, DIST_FX_SLOT, index, param_vals[index])
        
def change_parameter_value(CurTr, param, fx):
    val = random.uniform(0.0, 1.0)
    og_val = val
    if (fx == 'SlapbackDelay' or fx == 'FeedbackDelay') and (param == 6 or param == 12):
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
    if plugin == 'Ample Guitar LP.dll':
        RPR_TrackFX_SetPreset(CurTr, 0, 'Clean')
    
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
    
    
def git_keys(plugin):
    create_guitar(0, plugin)
    create_keys(1)

    
def git_kick(plugin):
    create_guitar(0, plugin)
    CurTk = create_drums(1)
    kick_drum(CurTk)


def git_snare(plugin):
    create_guitar(0, plugin)
    CurTk = create_drums(1)
    snare(CurTk)

    
def git_hh_closed(plugin):
    create_guitar(0, plugin)
    CurTk = create_drums(1)
    hh_closed(CurTk)

    
def git_hh_open(plugin):
    create_guitar(0, plugin)
    CurTk = create_drums(1)
    hh_open(CurTk)

    
def git_cymbal(plugin):
    create_guitar(0, plugin)
    CurTk = create_drums(1)
    cymbal(CurTk)

    
def git_bass_drums(plugin):
    create_guitar(0, plugin)
    create_bass(1)
    CurTk = create_drums(2)
    kick_drum(CurTk)
    snare(CurTk)
    hh_closed(CurTk)

    
def git_keys_drums(plugin):
    create_guitar(0, plugin)
    create_keys(1)
    CurTk = create_drums(2)
    kick_drum(CurTk)
    snare(CurTk)
    hh_closed(CurTk)

    
def git_bass_keys(plugin):
    create_guitar(0, plugin)
    create_bass(1)
    create_keys(2)

    
def git_bass_keys_drums(pitch, plugin):
    create_guitar(0, plugin, pitch)
    create_bass(1)
    create_keys(2)
    CurTk = create_drums(3)
    kick_drum(CurTk)
    snare(CurTk)
    hh_closed(CurTk) 

    
def set_vol():  # lower volume of other instruments
        track_count = RPR_CountTracks(0)
        for track in range(track_count-1):
          CurTr = RPR_GetTrack(0, (track+1))
          RPR_SetMediaTrackInfo_Value(CurTr, "D_VOL", 0.15)