import sys
import os
import subprocess
import nibabel as nib
import numpy as np
import json
# from dipy.tracking.streamline import transform_streamlines
from dipy.tracking import utils
from scipy.io import loadmat


def bash_command(string_cmd):
    process = subprocess.Popen(string_cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output

def gen_hshtbl(trk):
    hshtbl = []
    for sl in trk:
        hshtbl.append(sum([256**2*c[0]+256*c[1]+c[2] for c in sl]))
    if len(trk) != len(np.unique(np.array(hshtbl))): print("Streamline hashing generated equivalent hashes")
    return hshtbl


def gen_masks(fs_dir,subrun):
    parcseg = os.path.join(fs_dir,subrun,'mri','aparc+aseg.mgz')
    anat = nib.load(parcseg)
    anat_img = anat.get_fdata().astype(int)
    # anat_aff = anat.affine
    mask_dir = os.path.join(fs_dir,subrun,'seg_masks')
    bash_command(f'mkdir {mask_dir}')
    for label in np.unique(anat_img):
        mask = np.ma.array(anat_img, mask = anat_img != label, fill_value=0).filled()
        np.save(os.path.join(mask_dir,f'{label}_msk'), mask)
    return mask_dir,parcseg

def read_trkfile(run):
    extra_calc=False
    trkfile = nib.streamlines.load(run)
    trk_img = trkfile.tractogram.streamlines
    trk_aff = trkfile.affine
    ht = gen_hshtbl(trk_img)
    return trk_img,trk_aff,ht

def get_trks(subject, maindir,dtk_path):
    if subject.split('-')[0]=='sub':
        sub = maindir+subject
        trkdir = os.path.join(sub,'dwi','tracks')
        fsdir = os.path.join(sub,'anat','freesurfer')
        for run in os.listdir(trkdir):
            trkrun = os.path.join(trkdir,run)
            # run_algos={}
            if os.path.isdir(trkrun) and run in os.listdir(fsdir):
                print(run)
                maskdir,mgz = gen_masks(fsdir,run)
                nii = mgz.split('.')[0]+'.nii'
                bash_command(f'mri_convert --in_type mgz --out_type nii {mgz} {nii}')
                bash_command(f'flirt -in {os.path.join(sub,"dwi","dtk",run)}_dwi.nii -ref {nii} -omat {os.path.join(sub,"dwi",run)}_reg.mtx -out {os.path.join(sub,"dwi",run)}_reg')
                algo_vxl_sl={}
                for algo in os.listdir(trkrun):
                    if algo.endswith('.trk') and 'fltr' in algo and 'reg' not in algo:
                        print(algo)
                        bash_command(f'{dtk_path}track_transform {os.path.join(trkrun,algo)} {os.path.join(trkrun,algo.split(".")[0])}_reg.trk -src {os.path.join(sub,"dwi","dtk",run)}_dwi.nii -ref {nii} -reg {os.path.join(sub,"dwi",run)}_reg.mtx -reg_type flirt')
                        algo_trks,algo_aff,hshtbl = read_trkfile(f'{os.path.join(trkrun,algo.split(".")[0])}_reg.trk')
                        # with open(f'{os.path.join(trkrun,algo.split(".")[0])}_sl.txt','w') as ht:
                        #     ht.write(hshtbl)
                        np.savetxt(f'{os.path.join(trkrun,algo.split(".")[0])}_sl.txt',np.array(hshtbl))
                        vxl_sl={}
                        for maskf in os.listdir(maskdir):
                            if maskf.endswith('npy'):
                                print(maskf)
                                mask = np.load(os.path.join(maskdir,maskf))
                                vxl_sl[maskf.split("_")[0]]=gen_hshtbl(list(utils.target(algo_trks,mask,algo_aff)))

                        sl_vxl={}
                        ks = sorted([int(k) for k in vxl_sl.keys()])
                        for kE in ks:
                            sl_vxl[str(kE)]=vxl_sl[str(kE)]
                        algo_vxl_sl[algo.split('.')[0]]=sl_vxl
                        # run_algos[f'{("_").join(algo.split(".")[0].split("_")[2:])}'] = [vxl.tolist() for vxl in algo_trks]
                with open(os.path.join(trkrun,"algo_vol-trk_map.json"),'w') as t:
                    json.dump(algo_vxl_sl,t)
    return


if __name__=="__main__":

    main_dir = '/Volumes/ElementsExternal/mridti_test2/'
    dtk_home='/Applications/Diffusion_Toolkit.app/Contents/MacOS/'

    # tracks={}

    for subdir in os.listdir(main_dir):
        if os.path.isdir(main_dir+subdir):
            # tracks[f'{subdir}'] = get_trks(subdir,main_dir)
            get_trks(subdir,main_dir,dtk_home)

    # for k in tracks.keys():
    #     print(k)
    # np.save(main_dir+'tracks.npy',tracks)
    # print(tracks['sub-01']['sub-01_run-01']['sub-01_run-01_fact.trk'])
