import os
import pandas as pd
import pretty_midi
import numpy as np
import music21
import glob
import torch
import torch.nn as nn
from torch.nn.functional import softplus
from _dataloader import MidiDataset
from _bar_transform import BarTransform
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns

def clear():
    if os.name == 'nt':  # windows machine
        _ = os.system('cls')
    else:  # for mac and linux(here, os.name is 'posix')
        _ = os.system('clear')


def encode_dummies(instrument, sampling_freq, note_columns_subset):
    """ MIDI to piano-roll with pretty_midi object instrument. Adds a lot of blank space.
    Transform to type uint8 (128 notes, uint8 goes to 255, saves memory). """
    pr = instrument.get_piano_roll(fs=sampling_freq).astype('uint8').T
    piano_roll_dataframe = pd.DataFrame(pr, columns=note_columns_subset)
    # empty dataframe with columns = { piano_roll_name, timestep, C3, C#0, ..., B7 }
    return piano_roll_dataframe


def trim_blanks(df):
    """ :return: first index of activity for this instrument to remove the first period
    of no activity for this instrument """
    nonzero = df.apply(lambda s: s != 0)
    nonzeroes = df[nonzero].apply(pd.Series.first_valid_index)
    first_nonzero = nonzeroes.min()
    if first_nonzero is pd.np.nan:
        return None
    return df.iloc[int(first_nonzero):]


def chopster(dframe):
    """ :return: the dframe with removed upper and lower bounds of zeros """
    dframe.drop(labels=[pretty_midi.note_number_to_name(n) for n in range(108, 128)], axis=1, inplace=True)
    dframe.drop(labels=[pretty_midi.note_number_to_name(n) for n in range(0, 48)], axis=1, inplace=True)
    return dframe


def minister(dframe):  # Non-zero values changed to 1's
    return dframe.where(dframe < 1, 1)


# Removes chords and turns them into melody
def arpster(dframe):
    """ :return dframe with chord removed and changed into melodies """
    note_amount = np.asarray(dframe.astype(bool).sum(axis=1))  # count amount of notes being played at once.
    i = 0
    while i < dframe.shape[0]:  # slide through whole MIDI
        if note_amount[i] == 1:  # check if note is single
            i += 1
            continue
        elif note_amount[i] > 1:  # if not, calculates the amount of notes being played
            hits = 0
            hit_index = []
            for j in range(dframe.shape[1]):
                if dframe.iloc[i, j] == 1:
                    hit_index.append(j)
                    hits += 1
                    if hits == note_amount[i]:
                        break
            length = 0

            while True in (dframe.iloc[i + length, hit_index] == 1).values:
                # removes all notes such that chords are turned into arpeggios.
                for k in range(len(hit_index)):
                    if k != (length % hits):
                        dframe.iloc[i + length, hit_index[k]] = 0
                length += 1
                if len(note_amount) <= i + length or note_amount[i + length - 1] != note_amount[i + length]:
                    # Ensures that all values in hit_index are the same as ones in dframe row.
                    break
            i += length  # skip ahead to next note
        elif note_amount[i] == 0:  # maybe a case where we count i-the amount of silent steps going ahead
            i += 1
            continue
    return dframe


def cutster(dframe, frame_size, undesired_silence):
    """ Cut into desired window size, check if frame size is greater than MIDI length
    :param frame_size: amount of measures per input :param undesired_silence:
    :return: chops up into desired window size (and maybe saves them to csv in this step?) """
    if frame_size > dframe.shape[0] / 16:
        return dframe
    else:
        note_amount = np.asarray(dframe.astype(bool).sum(axis=1))
        zero_amount = 0
        i = 0
        while i < len(note_amount):
            # cuts out silent measures if greater than undesired_silence
            if note_amount[i] == 0:  # count sequential zeros
                zero_amount += 1
                i += 1
            elif zero_amount / 16 > undesired_silence and note_amount[i] != 0:
                drop_amount = [j for j in range(i - zero_amount, i)]
                dframe.drop(drop_amount, inplace=True)
                note_amount = np.delete(note_amount, drop_amount)
                i -= zero_amount - 1
                zero_amount = 0
            elif note_amount[i] != 0:
                if zero_amount != 0:
                    zero_amount = 0
                i += 1
        return dframe


def padster(dframe):
    """ Add desired amount of padding to all MIDI files such that they contain the same amount of dimensions. """
    return dframe.fillna(0)


# Transpose MIDI to same key (C major or A minor)
def transposer(midi_file):
    """ transpose midi files to the same key"""
    # midi_array = midi_file.split('/')
    # Major conversion in C
    majors = dict([("A-", 4), ("G#", 4), ("A", 3), ("A#", 2), ("B-", 2), ("B", 1), ("C", 0), ("C#", -1), ("D-", -1),
                   ("D", -2), ("D#", -3), ("E-", -3), ("E", -4), ("F", -5), ("F#", 6), ("G-", 6), ("G", 5)])
    # Minor conversion in A
    minors = dict([("G#", 1), ("A-", 1), ("A", 0), ("A#", -1), ("B-", -1), ("B", -2), ("C", -3), ("C#", -4),
                   ("D-", -4), ("D", -5), ("D#", 6), ("E-", 6), ("E", 5), ("F", 4), ("F#", 3), ("G-", 3), ("G", 2)])
    score = music21.converter.parse(midi_file)
    key = score.analyze('key')
    if key.mode == "major":
        halfSteps = majors[key.tonic.name]
    elif key.mode == "minor":
        halfSteps = minors[key.tonic.name]
    return halfSteps
