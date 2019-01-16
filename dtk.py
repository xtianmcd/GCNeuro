import os
import json
import subprocess

def bash_cmd(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return

bash_cmd('dti_tracker {} -it {} [-fact -rk2 -tl -sl] -m {}'.format(INPUT_DATA_PREFIX OUTPUT_FILE,[input and output file type --> nii or nii.gz], [mask --> <filename>] ))
bash_cmd('odf_tracker {} -it {} [-rk2, default is non-interpolate streamline] -m {}'.format(INPUT_DATA_PREFIX OUTPUT_FILE,[input and output file type --> nii or nii.gz],[mask --> <filename>]  ))
bash_cmd('spline_filter {} {}'.format(INPUT_TRACK_FILE STEP_LENGTH [in unit of min. voxel size],OUTPUT_TRACK_FILE )) #--> smooth/clean up orig. track file
bash_cmd('track_transform {} -src {} -ref {}'.format(INPUT_TRACK_FILE OUTPUT_TRACK_FILE,[source vol. file - dwi or b0, nifti], [reference volume tracks are registered to, nifti] )) #--> transform a track file using given registration matrix
bash_cmd('track_merge {} {} [...] {} --> merge multiple track files into one'.format(INPUT_TRACK_FILE_1,INPUT_TRACK_FILE_2, ... , OUTPUT_FILE))
