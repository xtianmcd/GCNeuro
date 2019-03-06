import sys
import os
import subprocess
import nibabel as nib
import numpy as np


def bash_command(string_cmd):
    process = subprocess.Popen(string_cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output

def read_trkfile(run):

    trkfile = nib.streamlines.load()

def get_trks(subject):
    streamlines={}
    if subject.split('-')[0]=='sub':
        sub = maindir+subject
        trkdir = sub+'/dwi/tracks'
        for run in os.listdir(trkdir):
            # streamlines[f'{run}']={}
            run_algos={}
            trkrun = trkdir+run
            for algo in os.listdir(trkrun):
                # algo=trkrun+algo_file
                tracks = nib.streamlines.load(f'{trkrun+algo}')
                run_algos[f'{algo}'] = np.asarray(tracks.tractogram.streamlines)
            streamlines[f'{run}'] = run_algos
    return streamlines


if __name__=="__main__":

    main_dir = '/Volumes/ElementsExternal/test2/'

    tracks={}

    for subdir in os.listdir(main_dir):
        tracks[f'{subdir}'] = get_trks(subdir)
