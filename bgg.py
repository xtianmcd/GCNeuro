import sys
import os
import subprocess
import pandas as pd
import numpy as np


def bash_command(string_cmd):
    process = subprocess.Popen(string_cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output

def set_fs_var(path):
    # os.putenv('SUBJECTS_DIR', f'{path}')
    # os.system('export SUBJECTS_DIR')
    # os.environ['SUBJECTS_DIR'] = f'{path}'
    with open(f'{path}/fs_env.sh', 'a') as fs_env:
        fs_env.write(f'export SUBJECTS_DIR={path}')
    bash_command(f'source {path}/fs_env.sh')
    return

def nearest_voxel(center, roi):
    nearest=[]
    min_dist = 10000
    for vxl in roi:
        dist = sum(abs(np.subtract(vxl,center)))/3
        if dist < min_dist:
            min_dist=dist
            nearest=vxl
        elif dist==min_dist:
            nearest.append(vxl)
    return nearest


def get_rois(sub, maindir):
    if sub.split('-')[0]=='sub':
        # set_fs_var(maindir+sub)
        sub_fs = sub+'/anat/freesurfer/'
        for run in os.listdir(maindir+sub_fs):
            if 'sub' in run.split('_')[0]:
                subrun_fs = sub_fs+run
                for hemi in ['rh','lh']:
                    annot_cmd = f"mri_annotation2label --subject {subrun_fs} --hemi {hemi} --outdir {maindir+subrun_fs}/annot --sd {maindir}"
                    bash_command(annot_cmd)

                for coord_file in os.listdir(maindir+subrun_fs+'/annot'):
                    #with open(maindir+subrun_fs+'/annot/'+coord_file, 'r') as cf:
                    #    voxel_info = cf.read()
                    voxel_info = pd.read_csv(f'{maindir+subrun_fs}/annot/{coord_file}',sep='  ',skiprows=2,usecols=[1,2,3],names=['x','y','z'])#names=['row','x','y','z','unknown'])#,usecols=['x','y','z'])
                    voxel_info[['z','unknown']] = voxel_info['z'].str.split(' ',expand=True)
                    voxel_info= voxel_info.drop('unknown',axis=1)
                    voxel_info['z']=voxel_info['z'].astype(float)
                    # print(voxel_info.head())
                    cntr_vox = voxel_info.mean(axis=0).values
                    # print(cntr_vox)
                    if cntr_vox in voxel_info.values: print('yes')
                    else: cntr_vox=nearest_voxel(cntr_vox,voxel_info.values)
                    # print(cntr_vox)
                    # print()

    return


if __name__=="__main__":

    main_dir='/Volumes/ElementsExternal/test2/'

    for subdir in os.listdir(main_dir):
        get_rois(subdir, main_dir)
