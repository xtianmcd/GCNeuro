from os.path import exists as exists
from os.path import join as pathjoin
from os import mkdir as mkdir
import os
import sys

def restructure(main_dir):
    outdir = f'./{main_dir}_outputs'
    if not exists(outdir): mkdir(outdir)
    for sub in os.listdir(main_dir):
        if 'sub-' in sub:
            if not exists(pathjoin(outdir,sub)):
                mkdir(pathjoin(outdir,sub))
            if not exists(pathjoin(outdir,sub,'anat')):
                mkdir(pathjoin(outdir,sub,'anat'))
            if not exists(pathjoin(outdir,sub,'anat','brainsuite')):
                mkdir(pathjoin(outdir,sub,'anat','brainsuite'))
            if not exists(pathjoin(outdir,sub,'anat','freesurfer')):
                mkdir(pathjoin(outdir,sub,'anat','freesurfer'))
            if exists(pathjoin(main_dir,sub,'anat','brainsuite')):
                for r in os.listdir(pathjoin(main_dir,sub,'anat','brainsuite')):
                    if os.path.isdir(pathjoin(main_dir,sub,'anat','brainsuite',r)):
                        if not exists(pathjoin(outdir,sub,'anat','brainsuite',r)):
                            mkdir(pathjoin(outdir,sub,'anat','brainsuite',r))
                        if exists(pathjoin(main_dir,sub,'anat','brainsuite',r,\
                                f'{r}_T1w_brain.dwi.RAS.correct.nii.gz')):
                            os.rename(pathjoin(main_dir,sub,'anat','brainsuite',r,\
                                f'{r}_T1w_brain.dwi.RAS.correct.nii.gz'),
                                os.path.join(outdir,sub,'anat','brainsuite',r,
                                f'{r}_T1w_brain.dwi.RAS.correct.nii.gz'))
            if exists(pathjoin(main_dir,sub,'anat','freesurfer')):
                for r in os.listdir(pathjoin(main_dir,sub,'anat','freesurfer')):
                    if os.path.isdir(pathjoin(main_dir,sub,'anat','freesurfer',r)) and sub in r:
                        if not exists(pathjoin(outdir,sub,'anat','freesurfer',r)):
                            mkdir(pathjoin(outdir,sub,'anat','freesurfer',r))
                        if not exists(pathjoin(outdir,sub,'anat','freesurfer',r,'mri')):
                            mkdir(pathjoin(outdir,sub,'anat','freesurfer',r,'mri'))
                        if exists(pathjoin(main_dir,sub,'anat','freesurfer',r,\
                                'mri','aparc+aseg.mgz')):
                            os.rename(pathjoin(main_dir,sub,'anat','freesurfer',r,\
                                'mri','aparc+aseg.mgz'),
                                pathjoin(outdir,sub,'anat','freesurfer',r,
                                'mri','aparc+aseg.mgz'))
            if exists(pathjoin(main_dir,sub,'dwi','dtk')):
                if not exists(pathjoin(outdir,sub,'dwi')):
                    mkdir(pathjoin(outdir,sub,'dwi'))
                if not exists(pathjoin(outdir,sub,'dwi','dtk')):
                    mkdir(pathjoin(outdir,sub,'dwi','dtk'))
                if not exists(pathjoin(outdir,sub,'dwi','tracks')):
                    mkdir(pathjoin(outdir,sub,'dwi','tracks'))
                for r in os.listdir(pathjoin(main_dir,sub,'dwi','dtk')):
                    if 'dwi' in r and r.endswith('nii'):
                        os.rename(pathjoin(main_dir,sub,'dwi','dtk',r),
                            pathjoin(outdir,sub,'dwi','dtk',r))
            if exists(pathjoin(main_dir,sub,'dwi','tracks')):
                for r in os.listdir(pathjoin(main_dir,sub,'dwi','tracks')):
                    if os.path.isdir(pathjoin(main_dir,sub,'dwi','tracks',r)):
                        if not exists(pathjoin(outdir,sub,'dwi','tracks',r)):
                            mkdir(pathjoin(outdir,sub,'dwi','tracks',r))
                        for t in os.listdir(pathjoin(main_dir,sub,'dwi','tracks',r)):
                            if 'fltr' in t and t.endswith('trk'):
                                os.rename(pathjoin(main_dir,sub,'dwi','tracks',r,t),
                                pathjoin(outdir,sub,'dwi','tracks',r,t))
    return

if __name__=="__main__":

    maindir = sys.argv[1]

    restructure(maindir)
