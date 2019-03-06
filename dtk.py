import os
import json
import subprocess

def bash_cmd(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return

def dtk_setup(dtk_path):
    bash_cmd('DSI_PATH={}/matrices/'.format(dtk_path))
    return

def create_gm(root_dir):
    with open('{}siemens_64.txt'.format(root_dir),'w') as gm:
        gm.write('1.000000,   0.000000,   0.000000\n')
        gm.write('0.000000,   1.000000,   0.000000\n')
        gm.write('-0.026007,   0.649170,   0.760199\n')
        gm.write('0.591136,   -0.766176,   0.252058\n')
        gm.write('-0.236071,   -0.524158,   0.818247\n')
        gm.write('-0.893021,   -0.259006,   0.368008\n')
        gm.write('0.796184,   0.129030,   0.591137\n')
        gm.write('0.233964,   0.929855,   0.283956\n')
        gm.write('0.935686,   0.139953,   0.323891\n')
        gm.write('0.505827,   -0.844710,   -0.174940\n')
        gm.write('0.346220,   -0.847539,   -0.402256\n')
        gm.write('0.456968,   -0.630956,   -0.626956\n')
        gm.write('-0.486997,   -0.388997,   0.781995\n')
        gm.write('-0.617845,   0.672831,   0.406898\n')
        gm.write('-0.576984,   -0.104997,   -0.809978\n')
        gm.write('-0.826695,   -0.520808,   0.212921\n')
        gm.write('0.893712,   -0.039987,   -0.446856\n')
        gm.write('0.290101,   -0.541189,   -0.789276\n')
        gm.write('0.115951,   -0.962591,   -0.244896\n')
        gm.write('-0.800182,   0.403092,   -0.444101\n')
        gm.write('0.513981,   0.839970,   0.173994\n')
        gm.write('-0.788548,   0.152912,   -0.595659\n')
        gm.write('0.949280,   -0.233069,   0.211062\n')
        gm.write('0.232964,   0.782880,   0.576911\n')
        gm.write('-0.020999,   -0.187990,   -0.981946\n')
        gm.write('0.216932,   -0.955701,   0.198938\n')
        gm.write('0.774003,   -0.604002,   0.190001\n')
        gm.write('-0.160928,   0.355840,   0.920587\n')
        gm.write('-0.147035,   0.731173,   -0.666158\n')
        gm.write('0.888141,   0.417066,   0.193031\n')
        gm.write('-0.561971,   0.231988,   -0.793959\n')
        gm.write('-0.380809,   0.142928,   0.913541\n')
        gm.write('-0.306000,   -0.199000,   -0.931001\n')
        gm.write('-0.332086,   -0.130034,   0.934243\n')
        gm.write('-0.963226,   -0.265062,   0.044010\n')
        gm.write('-0.959501,   0.205107,   0.193101\n')
        gm.write('0.452965,   -0.888932,   0.067995\n')
        gm.write('-0.773133,   0.628108,   0.088015\n')
        gm.write('0.709082,   0.408047,   0.575066\n')
        gm.write('-0.692769,   0.023992,   0.720760\n')
        gm.write('0.681659,   0.528735,   -0.505747\n')
        gm.write('-0.141995,   -0.724976,   0.673978\n')
        gm.write('-0.740168,   0.388088,   0.549125\n')
        gm.write('-0.103006,   0.822044,   0.560030\n')
        gm.write('0.584037,   -0.596038,   0.551035\n')
        gm.write('-0.088008,   -0.335031,   0.938088\n')
        gm.write('-0.552263,   -0.792377,   0.259123\n')
        gm.write('0.838158,   -0.458086,   -0.296056\n')
        gm.write('0.362995,   -0.560993,   0.743990\n')
        gm.write('-0.184062,   0.392133,   -0.901306\n')
        gm.write('-0.720938,   -0.692941,   0.008999\n')
        gm.write('0.433101,   0.682159,   -0.589137\n')
        gm.write('0.502114,   0.690157,   0.521119\n')
        gm.write('-0.170944,   -0.508833,   -0.843722\n')
        gm.write('0.462968,   0.422971,   0.778946\n')
        gm.write('0.385030,   -0.809064,   0.444035\n')
        gm.write('-0.713102,   -0.247035,   0.656094\n')
        gm.write('0.259923,   0.884737,   -0.386885\n')
        gm.write('0.001000,   0.077002,   -0.997030\n')
        gm.write('0.037002,   -0.902057,   0.430027\n')
        gm.write('0.570320,   -0.303170,   -0.763428\n')
        gm.write('-0.282105,   0.145054,   -0.948354\n')
        gm.write('0.721098,   0.608082,   0.332045\n')
        gm.write('0.266985,   0.959945,   -0.084995')
    return

def det_algos(main_path, sub_path, dwi_runs, dtk_path, subj, num_b0s=1):
    bash_cmd(f'mkdir {sub_path}/dwi/dtk/')
    bash_cmd(f'mkdir {sub_path}/dwi/tracks')
    for run in dwi_runs:
        bash_cmd(f'mkdir {sub_path}/dwi/tracks/{run}')
        bash_cmd(f'{dtk_path}dti_recon {sub_path}/anat/brainsuite/{run}/{run}_T1w_brain.dwi.RAS.correct.nii.gz {sub_path}/dwi/dtk/{run} -gm {main_path}/siemens_64.txt -b0 {num_b0s}')
        bash_cmd(f'{dtk_path}dti_tracker {sub_path}/dwi/dtk/{run} {sub_path}/dwi/tracks/{run}/{run}_fact.nii -it 'nii' -fact -m {sub_path}/dwi/dtk/{run}_dwi.nii')
        bash_cmd(f'{dtk_path}dti_tracker {sub_path}/dwi/dtk/{run} {sub_path}/dwi/tracks/{run}/{run}_rk2.nii -it 'nii' -rk2 -m {sub_path}/dwi/dtk/{run}_dwi.nii')
        bash_cmd(f'{dtk_path}dti_tracker {sub_path}/dwi/dtk/{run} {sub_path}/dwi/tracks/{run}/{run}_tl.nii -it 'nii' -tl -m {sub_path}/dwi/dtk/{run}_dwi.nii')
        bash_cmd(f'{dtk_path}dti_tracker {sub_path}/dwi/dtk/{run} {sub_path}/dwi/tracks/{run}/{run}_sl.nii -it 'nii' -sl -m {sub_path}/dwi/dtk/{run}_dwi.nii')
    return

# bash_cmd('dti_tracker {} -it {} [-fact -rk2 -tl -sl] -m {}'.format(INPUT_DATA_PREFIX OUTPUT_FILE,'nii.gz'], [mask --> <filename>] ))
bash_cmd('odf_tracker {} -it {} [-rk2, default is non-interpolate streamline] -m {}'.format(INPUT_DATA_PREFIX OUTPUT_FILE,[input and output file type --> nii or nii.gz],[mask --> <filename>]  ))
bash_cmd('spline_filter {} {}'.format(INPUT_TRACK_FILE STEP_LENGTH [in unit of min. voxel size],OUTPUT_TRACK_FILE )) #--> smooth/clean up orig. track file
bash_cmd('track_transform {} -src {} -ref {}'.format(INPUT_TRACK_FILE OUTPUT_TRACK_FILE,[source vol. file - dwi or b0, nifti], [reference volume tracks are registered to, nifti] )) #--> transform a track file using given registration matrix
bash_cmd('track_merge {} {} [...] {} --> merge multiple track files into one'.format(INPUT_TRACK_FILE_1,INPUT_TRACK_FILE_2, ... , OUTPUT_FILE))

def run_tractography(subject, maindir, dtkdir, init_setup=False):

    if subject.split('-')[0]=='sub':

        print(f'Preprocessing data for {subject}')
        sub = maindir+subject

        if init_setup: dtk_setup(dtkdir)
        create_gm(maindir)

    return


if __name__ == "__main__":

    main_dir='/Volumes/ElementsExternal/test2/'
    dtk_home='/Applications/Diffusion\ Toolkit.app/Contents/MacOS/'

    for subdir in os.listdir(main_dir):
        run_tractography(subdir, main_dir, dtk_home)
