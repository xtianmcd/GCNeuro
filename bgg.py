import sys
import os
import subprocess
import pandas as pd
import numpy as np
import random
import json
import pickle
import nibabel as nib


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
    # print(nearest)
    return nearest[random.randint(0,len(nearest)-1)]

def gen_cntrs(main_dir,fs_run):
    parcseg = os.path.join(main_dir,fs_run,'mri','aparc+aseg.mgz')
    anat = nib.load(parcseg)
    anat_img = anat.get_fdata().astype(int)
    # anat_aff = anat.affine
    centers={}
    for label in np.unique(anat_img):
        mask = np.ma.array(anat_img, mask = anat_img != label, fill_value=0).filled()
        # print(np.unique(mask))
        idx = np.array([[a,b,c] for a in range(mask.shape[0]) for b in range(mask.shape[1]) for c in range(mask.shape[2]) if mask[a,b,c]!=0])
        # print(idx)
        if len(idx)>0:
            cntr_vox = np.mean(idx,axis=0)
            # print(cntr_vox)
            if cntr_vox not in idx: cntr_vox=nearest_voxel(cntr_vox,idx)
            centers[str(label)]=cntr_vox
    print(centers.keys())
    # cntrs={}
    # ks = sorted([int(k) for k in centers.keys()])
    # for kE in ks:
    #     cntrs[str(kE)]=centers[str(kE)]
    return centers

def get_rois(sub, maindir):
    center_vxls={}
    if sub.split('-')[0]=='sub':
        sub_fs = os.path.join(sub,'anat/freesurfer/')
        for run in os.listdir(os.path.join(maindir,sub_fs)):
            if 'sub' in run.split('_')[0]:
                subrun_fs = os.path.join(sub_fs,run)
                if os.path.isdir(os.path.join(maindir,subrun_fs)):
                    center_vxls[run] = gen_cntrs(maindir,subrun_fs)

    return center_vxls


if __name__=="__main__":

    main_dir='/Volumes/ElementsExternal/mridti_test2'

    subj_centers={}
    for subdir in os.listdir(main_dir):
        if os.path.isdir(os.path.join(main_dir,subdir)):
            subj_centers[subdir] = get_rois(subdir, main_dir)

    with open(f'{os.path.join(main_dir,"center_vxls.json")}','w') as rois:
        json.dump(subj_centers,rois)
