import sys
import os
import subprocess


def bash_command(string_cmd):
    process = subprocess.Popen(string_cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output

def get_rois(subject):
    if subject.split('-')[0]=='sub':
        sub = maindir+subject
        sub_fs = sub+'/anat/freesurfer/'
        for run in os.listdir(sub_fs):
            if 'sub' in run.split('_')[0]:
                subrun_fs = sub_fs+run
                for hemi in ['rh','lh']:
                    annot_cmd = f"mri_annotation2label --subject {subrun_fs} --hemi {hemi} --outdir annot")
                    bash_command(annot_cmd)

                for coord_file in os.listdir(subrun_fs+'/annot'):
                    with open(coord_file, 'r') as cf:
                        voxel_info = cf.read()
                    print(voxel_info)

    return


if __name__=="__main__":

    main_dur='/Volumes/ElementsExternal/test2'

    for subdir in os.listdir(main_dir):
        get_rois(subdir)
