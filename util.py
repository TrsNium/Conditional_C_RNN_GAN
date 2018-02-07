import pretty_midi
import os
import numpy as np
import re
import random

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

def read_midi_as_piano_roll(fn, fs):
    print(fn)
    p_r = p_m.get_piano_roll(fs)
    return np.array(p_r).transpose((1,0))

def check_file(fn):
    try:
        p_m = pretty_midi.PrettyMIDI(fn)
        return True
    except:
        return False

def pre_processing(data_path, fs, step_num, max_time_step, range_):
    roll = read_midi_as_piano_roll(data_path, fs) 
    data_len = step_num*max_time_step
    empty = np.empty(shape=[0, range_])
    
    #print(roll.shape)
    while True:
        if not empty.shape[0] < data_len:
            break

        empty = np.append(empty, roll, axis=0)
    return empty[:data_len,:]
        
def mk_index(data_dir="./data", save_path="./data/index.txt"):
    dir_ = [dir_ for dir_ in os.listdir(data_dir) if re.match("genre-", dir_)]
    dir_ = [d for d in dir_ if not len(os.listdir(os.path.join(data_dir, d))) == 0]
    content = "\n".join(dir_)
    with open(save_path, "w") as fs:
        fs.write(content)

def read_index(read_path="./data/index.txt"):
    with open(read_path, "r") as fs:
        lines = fs.readlines()
    return [line.split("\n")[0] for line in lines]

def mk_label(label_size, c):
    content = np.zeros(label_size)
    content[c] = 1
    return content

def mk_train_func(batch_size, step_num, max_time_step, fs, range_, data_dir="./data", index_path="./data/index.txt"):
    if not os.path.exists(index_path): mk_index(data_dir, index_path)
    indexes = read_index(index_path)
    label_size = len(indexes)

    data_group_by_label = {index: [pre_processing(os.path.join(data_dir, index, fn), fs, step_num, max_time_step, range_) for fn in os.listdir(os.path.join(data_dir, index)) if check_file(os.path.join(data_dir, index, fn))] for index in indexes}
    
    def train_func():
        while True:
            labels = []
            data = []
            choiced_indexes = random.sample(range(len(indexes)), k=batch_size)
            [labels.append(mk_label(label_size, c)) for c in choiced_indexes]

            for choiced_index in choiced_indexes:
                choiced_data_index = random.sample(range(len(data_group_by_label[indexes[choiced_index]])), k=1)
                data.append(data_group_by_label[indexes[choiced_index]][choiced_data_index[0]])

            yield np.array(data), np.array(labels)
    return train_func
