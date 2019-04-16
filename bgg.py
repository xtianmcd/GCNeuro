import sys
import os
import subprocess
import pandas as pd
import numpy as np
import random
import json
import pickle


def bash_command(string_cmd):
    process = subprocess.Popen(string_cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output


def nearest_voxel(center, roi):
    """ Finds the voxel nearest to the XYZ mean of an ROI that is also part of the ROI """
    nearest=[]
    min_dist = 10000
    for vxl in roi:
        dist = sum(abs(np.subtract(vxl,center)))/3
        if dist < min_dist:
            min_dist=dist
            nearest=[vxl]
        elif dist==min_dist:
            nearest.append(vxl)
    return nearest[random.randint(0,len(nearest)-1)]


def get_rois(sub, maindir):
    voxs = {}
    if sub.split('-')[0]=='sub':
        sub_fs = os.path.join(sub,'anat/freesurfer/')
        for run in os.listdir(os.path.join(maindir,sub_fs)):
            if 'sub' in run.split('_')[0]:
                subrun_fs = os.path.join(sub_fs,run)
                for hemi in ['rh','lh']:
                    annot_cmd = f"mri_annotation2label --subject {subrun_fs} --hemi {hemi} --outdir {os.path.join(maindir,subrun_fs,'annot')} --sd {maindir}"
                    bash_command(annot_cmd)
                for coord_file in os.listdir(os.path.join(maindir,subrun_fs,'annot')):
                    with open(f'{os.path.join(main_dir,subrun_fs,"annot",coord_file)}','r') as f:
                        f.readline()
                        voxel_num = f.readline()
                    voxel_info = pd.read_csv(f'{os.path.join(maindir,subrun_fs,"annot",coord_file)}',sep='  ',skiprows=2,usecols=[1,2,3],names=['x','y','z'])#names=['row','x','y','z','unknown'])#,usecols=['x','y','z'])
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
                    voxs[voxel_num.strip()]=cntr_vox
    s_voxs={}
    srtd_v=[]
    # print(voxs.keys())
    kEz=sorted([int(k) for k in voxs.keys()])
    # print(kEz)
    for k in kEz:
        s_voxs[str(k)] = list(voxs[str(k)])
        srtd_v.append(voxs[str(k)])
    return s_voxs, np.array(srtd_v)


if __name__=="__main__":

    main_dir='/Volumes/ElementsExternal/test2'
    rois_d={}
    rois_l=[]
    for subdir in os.listdir(main_dir):
        if os.path.isdir(os.path.join(main_dir,subdir)):
            sub_d,sub_l = get_rois(subdir, main_dir)
            rois_d[subdir]=sub_d
            rois_l.append(sub_l)
    with open(f'{os.path.join(main_dir,subdir,"anat","freesurfer","rois.json")}','w') as rois:
        json.dump(rois_d,rois)
    with open(f'{os.path.join(main_dir,subdir,"anat","freesurfer","rois.pkl")}','w') as rois:
        pickle.dump(np.arrays(rois_l),rois)
