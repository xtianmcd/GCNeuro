import os
import json
import subprocess
from joblib import Parallel, delayed
import time


def bash_cmd(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return

def fsl_fs_setup:
    """ FSL Setup """
    bash_cmd('FSLDIR=/usr/local/fsl')
    bash_cmd('PATH=${FSLDIR}/bin:${PATH}')
    bash_cmd('export FSLDIR PATH')
    bash_cmd('. ${FSLDIR}/etc/fslconf/fsl.sh')

    """ Freesurfer Setup """
    bash_cmd('export FREESURFER_HOME=/Applications/freesurfer')
    bash_cmd('source $FREESURFER_HOME/SetUpFreeSurfer.sh')
    return


def pick_dwi_runs(subject):
    print(" Connectivity (DWI)")
    print("+==================+")
    jsons = list([os.path.join(root,i) for root,d,f in os.walk(subject)\
                                        for i in f \
                                        if root.split('/')[-1]=='dwi' \
                                        and i.endswith('json')])
    stats_for_topup={}
    for jsonf in jsons:
        """ read through the json header files for each run """
        with open(jsonf,'r') as header:
            hdr_dict = json.load(header)
            stats_for_topup[jsonf.split('/')[-1].split('.')[0]] = \
                            {'readout':hdr_dict["TotalReadoutTime"],\
                            'ped':hdr_dict["PhaseEncodingDirection"]}
    runs_for_topup = set([k for k,v in stats_for_topup.items() \
                            for k1,v1 in stats_for_topup.items() \
                                if k!=k1 and \
                                    (v['readout']!=v1['readout'] \
                                        or v['ped']!=v1['ped'])])
                                            # runs need different stats
    if not runs_for_topup:
        print("Topup stats are all the same for {}\n... Topup will not be used; distortions will be corrected during co-registration via BrainSuite"\
        .format(subject))
        topup_qual = False
        return stats_for_topup, [], topup_qual
    else: print("Imaging parameters allow for Topup to be used prior to eddy")
    print("\tRuns found for {}: {}".format(subject,list(stats_for_topup.keys())))
    print()
    topup_qual = True
    return stats_for_topup,runs_for_topup, topup_qual

def merge_b0s(subject,runs,topup_req):
    print("\tMerging b0's")
    merged = ''
    for run in runs:
        nii = subject+'/dwi/'+run+'.nii'
        bash_cmd("fslroi {} {}_b0 0 1".format(nii,subject+'/dwi/'+'_'\
        .join(run.split('_')[:2]))) # create the b0's
        merged += subject+'/dwi/'+'_'.join(run.split('_')[:2])+'_b0 '
    bash_cmd("fslmerge -t {}/dwi/b0merge {}".format(subject,merged)) #merge b0's
    print("\t\tcreated {}/dwi/b0merge from {}".format(subject,merged))
    print()
    return

def create_acq(hdr_stats, subj):
    print("\t\t...Creating acquisition file")
    acq_str = ''
    for k,v in hdr_stats.items():
        """ create acquisition file for topup """
        if len(v['ped'])==1:
            if v['ped']=='i':
                acq_str+='1 0 0'
                v['bdp']='x'
            elif v['ped']=='j':
                acq_str+='0 1 0'
                v['bdp']='y'
            elif v['ped']=='k':
                acq_str+='0 0 1'
                v['bdp']='z'
        else:
             if v['ped']=='i-':
                 acq_str+='-1 0 0'
                 v['bdp']='x-'
             elif v['ped']=='j-':
                 acq_str+='0 -1 0'
                 v['bdp']='y-'
             elif v['ped']=='k-':
                 acq_str+='0 0 -1'
                 v['bdp']='z-'
        if not acq_str: print("error reading PhaseEncodingDirection of {}"\
                                    .format(k))
        acq_str+=' '+str(v['readout'])+'\n'
    # print('printf \"{}\" > {}acq.txt'.format(acq_str, subj+'/dwi/'))
    # bash_cmd('printf "{}" > {}/dwi/acq.txt'.format(acq_str, subj)) #create acquisit. file
    with open(subj+'/dwi/acq.txt', 'w') as acq_txt:
        acq_txt.write(acq_str)
    print(hdr_stats)
    return hdr_stats

def create_cnf(subj):
    print("\t\t...Creating config file")
    with open(subj+'/dwi/b02b0.cnf', 'w') as cnf:
        cnf.write('# Resolution (knot-spacing) of warps in mm\n')
        cnf.write('--warpres=20,16,14,12,10,6,4,4,4\n')
        cnf.write('# Subsampling level (a value of 2 indicates that a 2x2x2 neighbourhood is collapsed to 1 voxel)\n')
        cnf.write('--subsamp=2,2,2,2,2,1,1,1,1\n')
        cnf.write('# FWHM of gaussian smoothing\n')
        cnf.write('--fwhm=8,6,4,3,3,2,1,0,0\n')
        cnf.write('# Maximum number of iterations\n')
        cnf.write('--miter=5,5,5,5,5,10,10,20,20\n')
        cnf.write('# Relative weight of regularisation\n')
        cnf.write('--lambda=0.005,0.001,0.0001,0.000015,0.000005,0.0000005,0.00000005,0.0000000005,0.00000000001\n')
        cnf.write('# If set to 1 lambda is multiplied by the current average squared difference\n')
        cnf.write('--ssqlambda=1\n')
        cnf.write('# Regularisation model\n')
        cnf.write('--regmod=bending_energy\n')
        cnf.write('# If set to 1 movements are estimated along with the field\n')
        cnf.write('--estmov=1,1,1,1,1,0,0,0,0\n')
        cnf.write('# 0=Levenberg-Marquardt, 1=Scaled Conjugate Gradient\n')
        cnf.write('--minmet=0,0,0,0,0,1,1,1,1\n')
        cnf.write('# Quadratic or cubic splines\n')
        cnf.write('--splineorder=3\n')
        cnf.write('# Precision for calculation and storage of Hessian\n')
        cnf.write('--numprec=double\n')
        cnf.write('# Linear or spline interpolation\n')
        cnf.write('--interp=spline\n')
        cnf.write('# If set to 1 the images are individually scaled to a common mean intensity\n')
        cnf.write('--scale=1\n')
    return

def run_topup(hdr_info, main_dir, subject):
    print("\tPrepping for topup...")
    updated_stats = create_acq(hdr_info, subject)
    create_cnf(subject)
    print("\tRunning topup...")
    bash_cmd(\
        'topup --imain={}/dwi/b0merge --datain={}/dwi/acq.txt --config={}/dwi/b02b0.cnf --out={}/dwi/{}_topup --iout={}/dwi/{}_topup_b0'\
            .format(subject,subject,subject,subject,subject.split('/')[-1],\
            subject,subject.split('/')[-1]))
    print()
    return updated_stats

def skull_strip_dwi(subject, nii_name):
    print("\tSkull strip...")
    bash_cmd('fslmaths {}/dwi/{} -Tmean {}/dwi/{}_mean'\
    .format(subject,nii_name, subject,nii_name))
    bash_cmd('bet {}/dwi/{}_mean {}/dwi/{}_brain -m'\
    .format(subject,nii_name, subject,nii_name)) #run BET for skull stripping
    print()
    return

def run_eddy(hdr_info, subject, topup_ran):
    print("\tRun eddy for each run")
    ind=0
    if topup_ran:
        topup_flag = '--topup={}/dwi/{}_topup'\
                        .format(subject,subject.split('/')[-1])
        mask_prefix = subject.split('/')[-1]
    else: topup_flag = ''
    for run in list(hdr_info.keys()):
        """ run eddy on each run """
        print("\t\tRun {}".format(run))
        if not topup_ran:
            mask_prefix = run
        ind +=1
        # bash_cmd('indx = ""')
        indx = ''
        # bash_cmd('for ((i=1;i<=65;i+=1));do indx="$indx {}"; done'.format(ind))
        indx = ' '.join(list([str(ind) for _ in range(1,66)]))
        # bash_cmd('echo $indx > index.txt')
        with open (subject+'/dwi/index{}.txt'.format(ind), 'w') as index_txt:
            index_txt.write(indx)
            index_txt.write(' ')
        bash_cmd(\
            'eddy --imain={}/dwi/{}.nii --mask={}/dwi/{}_brain_mask --acqp={}/dwi/acq.txt --index={}/dwi/index{}.txt --bvecs={}/dwi/{}.bvec --bvals={}/dwi/{}.bval {} --out={}/dwi/{}_eddy_corr'\
                .format(subject,run, subject,mask_prefix,\
                 subject, subject, ind, subject,run, subject,run,\
                 topup_flag, subject,run))
    print()
    return

def run_brainsuite(subject, hdr_info, bs_home, topup_ran):
    print("\tRunning BrainSuite...")
    if topup_ran: dwi_mask = '{}/dwi/{}_topup_b0_brain_mask'\
        .format(subject,subject.split('/')[-1])
    bash_cmd('mkdir {}/anat/brainsuite'.format(subject))
    for run in list(hdr_info.keys()):
        print("\tPerforming dwi-mri co-registration for run {}"\
            .format(run))
        print("\t\t\t... skull strip re: anatomical image")
        bash_cmd('bet {}/anat/{}_T1w {}/anat/brainsuite/{}_T1w_bdp_brain -m'\
            .format(subject,'_'.join(run.split('_')[:2]),\
                    subject,'_'.join(run.split('_')[:2])))
        print("\t\t\t... Bias Field Correction on anatomical image")
        bash_cmd('{}/bin/bfc -i {}/anat/brainsuite/{}_T1w_bdp_brain -o {}/anat/brainsuite/{}_T1w_brain.bfc'\
            .format(bs_home, subject, '_'.join(run.split('_')[:2]),\
                             subject, '_'.join(run.split('_')[:2])))
        print("\t\t\t... running brainsuite diffusion pipeline (BDP)")
        if not topup_ran: dwi_mask = '{}/dwi/{}_brain_mask'.format(subject,run)
        with open('{}/dwi/{}_run_bdp.sh'.format(subject,run),'w') as bdp_sh:
            bdp_sh.write('{}/bdp/bdp.sh {}/anat/brainsuite/{}_T1w_brain.bfc.nii.gz --output-diffusion-coordinate --output-subdir {} --dir=\"{}\" --t1-mask {}/anat/brainsuite/{}_T1w_bdp_brain_mask.nii.gz --dwi-mask {}.nii.gz --nii {}/dwi/{}_eddy_corr.nii.gz -g {}/dwi/{}_eddy_corr.eddy_rotated_bvecs -b {}/dwi/{}.bval'\
                .format(bs_home, subject,'_'.join(run.split('_')[:2]), \
                           '_'.join(run.split('_')[:2]), hdr_info[run]['bdp'], \
                           subject,'_'.join(run.split('_')[:2]), dwi_mask \
                           subject,run, subject,run, subject,run))

        bash_cmd('sh {}/dwi/{}_run_bdp.sh'.format(subject,run))
    return

def run_freesurfer(main_dir, subject, sub_dir):
    print(" Anatomy (MRI)")
    print("+=============+")
    print("\tPerforming Freesurfer for each run")
    anatdir = subject+'/anat'
    bash_cmd('mkdir {}'.format(anatdir+'/freesurfer'))

    for anatfile in os.listdir(anatdir):
        if anatfile.endswith('.nii'):
            print("\t\trun {}".format(anatfile))
            bash_cmd('recon-all -sd {} -s {} -i {} -all'\
            .format(anatdir+'/freesurfer', sub_dir+'_'+anatfile.split('_')[1], \
            anatdir+'/'+anatfile))
    print("\n")
    return

def preprocess_subject(subject, maindir, brainsuitedir, init_setup=False):
    if init_setup: fsl_fs_setup()

    if subject.split('-')[0]=='sub':

        print(f'Preprocessing data for {subject}')

        sub = maindir+subject

        topup_stats,topup_runs,use_topup = pick_dwi_runs(sub)

        if use_topup:
            merge_b0s(sub,topup_runs,use_topup)
            stats_topup = run_topup(topup_stats, maindir, sub)
            skull_strip_dwi(sub, sub.split('/')[-1]+'_topup_b0')
        else:
            merge_b0s(sub,list(topup_stats.keys()),use_topup)
            stats_topup = create_acq(topup_stats, sub)
            for run in list(stats_topup.keys()):
                skull_strip_dwi(sub, run)

        # skull_strip(sub, list(topup_stats.keys()), use_topup)

        run_eddy(stats_topup, sub, use_topup)

        run_brainsuite(sub, stats_topup, brainsuitedir, use_topup)

        run_freesurfer(maindir,sub,subject)

        print(f"+===========+\nDone with {subject}\n+===========+")
    return

if __name__ == "__main__":

    main_dir = '/Volumes/ElementsExternal/test2/'
    brainsuite_home = '/Applications/BrainSuite18a'

    njobs=[1,-1]

    for n, jobs in enumerate(njobs):
        start=time.time()
        Parallel(n_jobs=jobs,verbose=50)(delayed(preprocess_subject)(subdir, main_dir, brainsuite_home) for subdir in os.listdir(maindir))
        times[n]= time.time() - start
    np.save("times.txt",times)
