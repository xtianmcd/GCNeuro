import os
import sys
import json
import subprocess
import pandas as pd
import datetime

"""
SECOND FILE IN PIPELINE (following neuro_format.py)

Converts raw DICOM image collections to NIfTI file format;
changes file structure from to be compatible with BIDS format;
image sources other than PPMI will probably need changes;
BIDS is the expected format for many libraries and is intuitively structured

To run, requires argument:

    `python make_bids.py <name_of_main_directory_with_images>`

Next file to run is `automate_preproc.py`
"""

def re_run(files):
    """ Checks for duplicate timestamps """
    for f in range(len(files)):
        i=0
        for ff in range(len(files)):
            if files[f].split('/')[2]==files[ff].split('/')[2] and \
                    files[f].split('/')[4]==files[ff].split('/')[4] and \
                    files[f].split('/')[3]!=files[ff].split('/')[3]:
                i+=1
                newf = files[ff].split('/')
                nf4_s = int(newf[4].split('_')[-1].split('.')[0])+1
                nf4_new = '_'.join(newf[4].split('_')[:-1])+str(nf4_s)+'.0'
                newf[4]=f'{newf[4]}'
                os.mkdir('/'.join(newf[:5]))
                newf='/'.join(newf)
                os.rename(files[ff],os.path.join(newf))
                files[ff]=newf
    return files

def bidsify(files,runsdict,modality_fn,modality_dn,modality_src,srcdata,derivs,sub,maindir,procdiffs=None):
    if not os.path.exists(os.path.join(srcdata,sub)):
        os.mkdir(os.path.join(srcdata,sub))
    if not os.path.exists(os.path.join(srcdata,sub,modality_src)):
        os.mkdir(os.path.join(srcdata,sub,modality_src))
    for f in files:
        run = [k+1 for k in runsdict.keys() if runsdict[k]==f.split('/')[4].split('.')[0]][0]
        if int(run)<=9:
            run=f'0{run}'
        else:
            run=str(run)
        newfn = os.path.join(sub,f'_run-{run}_{modality_fn}.nii')
        newdir = os.path.join(maindir,sub,modality_dn)
        cmd = "dcm2niix -b y -o {} -f {} {}".format(newdir,newfn,f)
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.rename(f,os.path.join(srcdata,sub,modality_src,f'{sub}_run-{run}/'))
    if modality_fn=='dwi':
        if not os.path.exists(os.path.join(derivs,sub)):
            os.mkdir(os.path.join(derivs,sub))
            os.mkdir(os.path.join(derivs,sub,'dwi'))
        for pd in procdiffs:
            os.rename(pd,os.path.join(derivs,sub,'dwi',f'{sub}_run-{run}_processed_dwi.nii'))
    if modality_dn=='func': return newdir
    else: return

def re_format(maindir,metadata):
    main_dir = os.path.join('.',maindir)
    subj_dirs = os.listdir(main_dir)
    metadata=os.path.join(main_dir,metadata)

    jsondict = {
        "Name":maindir,
        "BIDSVersion": "1.0.0rc1"
    }

    metadf = pd.read_csv(metadata)
    metadf["Subject_Num"]=""

    with open(os.path.join(main_dir,'dataset_description.json'), 'w') as outfile:
        json.dump(jsondict, outfile)

    srcdata = os.path.join(main_dir,'sourcedata/')
    if not os.path.exists(srcdata):
        os.mkdir(srcdata)
    derivs = os.path.join(main_dir,'derivatives')
    if not os.path.exists(derivs):
        os.mkdir(derivs)

    sub=0
    for subj in subj_dirs:
        print(f"File structure for {subj}")
        subject = os.path.join(main_dir,subj)
        if os.path.isdir(subject):
            run=0
            sub+=1
            if sub<=9: sub_name = "sub-0{}".format(sub)
            else: sub_name = "sub-{}".format(sub)
            print("\tOld ID: {} --> NewID: {}\n".format(subj, sub_name))
            metadf.loc[metadf.Subject.astype(int) == int(subj), 'Subject_Num'] = sub_name

            if not os.path.exists(os.path.join(main_dir,sub_name)):
                os.mkdir(os.path.join(main_dir,sub_name))
                os.mkdir(os.path.join(main_dir,sub_name,'anat'))
                os.mkdir(os.path.join(main_dir,sub_name,'dwi'))
                os.mkdir(os.path.join(main_dir,sub_name,'func'))


            # non_nii = {
            #     "anat": list([os.path.join(roota,filenamea) for roota,dirnamesa,filenamesa in os.walk(subject) if len(roota.split('/'))==6 and 'T1' in roota.split('/')[3] for filenamea in filenamesa if not filenamea.endswith('.nii')]),
            #     "fa"  : list(os.path.join(rootf,filenamef) for rootf,dirnamesf,filenamesf in os.walk(subject) if len(rootf.split('/'))==6 and 'FA' in rootf.split('/')[3] for filenamef in filenamesf if not filenamef.endswith('.nii'))
            # }
            # if non_nii["anat"]: print("WARNING: some T1 file(s) not .nii format in {} - {}".format(subj, non_nii['anat']))
            # if non_nii["fa"]: print("WARNING: some FA file(s) not .nii format in {} - {}".format(subj, non_nii['fa']))

            # Compile lists of all filepaths to images of a given modality
            # anats = list([os.path.join(roota,filenamea) \
            #           for roota,dirnamesa,filenamesa in os.walk(subject) \
            #           if len(roota.split('/'))==6 and 'T1' in \
            #           roota.split('/')[3] for filenamea in filenamesa \
            #           if filenamea.endswith('.nii')])
            rawanats = re_run(list(roota for roota,dirnamesa,filenamesa in \
                        os.walk(subject) if len(roota.split('/'))==6 and 'MPRAGE' in \
                        roota.split('/')[3].upper()))
            rawdiffs = re_run(list(rootr for rootr,dirnamesr,filenamesr in \
                        os.walk(subject) if len(rootr.split('/'))==6 and 'DTI'    in \
                        rootr.split('/')[3].upper()))
            # fadiffs = list(os.path.join(rootf,filenamef) for rootf,dirnamesf,filenamesf in os.walk(subject) if len(rootf.split('/'))==6 and 'FA' in rootf.split('/')[3] for filenamef in filenamesf if filenamef.endswith('.nii'))
            # diffs = list([os.path.join(roota,filenamea) for roota,dirnamesa,filenamesa in os.walk(subject) if len(roota.split('/'))==6 and 'DWI' in roota.split('/')[3] for filenamea in filenamesa if filenamea.endswith('.nii')])
            rawfuncs = re_run(list(rootf for rootf,dirnamesf,filenamesf in \
                        os.walk(subject) if len(rootf.split('/'))==6 and 'EP2D'   in \
                        rootf.split('/')[3].upper()))
            procdiffs = re_run(list(os.path.join(rootr,filenamer) \
                        for rootr,dirnamesr,filenamesr in \
                        os.walk(subject) if len(rootr.split('/'))==6 and 'DWI'
                        in rootr.split('/')[3].upper() for filenamer in filenamesr if filenamer.endswith('.nii')))

            # Genreate mapping from image timestamps to integer 'run' number
            dates=set([d for modality in os.listdir(subject) \
                if os.path.isdir(os.path.join(subject,modality)) \
                for d in os.listdir(os.path.join(subject,modality)) \
                if os.path.isdir(os.path.join(subject,modality,d))])
            dates = [ts.split('.')[0] for ts in dates]
            dates = [datetime.datetime.strptime(ts, "%Y-%m-%d_%H_%M_%S") for ts in dates]
            dates.sort()
            sorteddates = [datetime.datetime.strftime(ts, "%Y-%m-%d_%H_%M_%S") for ts in dates]
            # print(sorteddates)

            runs = dict(enumerate(sorteddates))
            # print(runs)

            # For each modality,
            # Convert images to NIfTI and rearrange file structure
            if len(rawanats) > 0: bidsify(rawanats,runs,'T1w',           'anat','mprage',srcdata,derivs,sub_name,main_dir)
            if len(rawdiffs) > 0: bidsify(rawdiffs,runs,'dwi',           'dwi' ,'dti'   ,srcdata,derivs,sub_name,main_dir,procdiffs)
            if len(rawfuncs) > 0:
                funcdir      =    bidsify(rawfuncs,runs,'task-rest_bold','func','ep2d'  ,srcdata,derivs,sub_name,main_dir)

                for jsn in os.listdir(funcdir):
                    if jsn.endswith('json'):
                        with open(os.path.join(funcdir,jsn),'r') as jr:
                            jdict = json.load(jr)
                        jdict['TaskName']='rest'
                        with open(os.path.join(funcdir,jsn),'w') as jw:
                            json.dump(jdict,jw)

            # run=0
            # for fadiff in fadiffs:
            #     run+=1
            #     if not os.path.exists(derivs+sub_name):
            #         os.mkdir(derivs+sub_name)
            #         os.mkdir(derivs+sub_name+'/fa/')
            #     newfn = sub_name+'_run-0{}_dwi.nii'.format(run)
            #     faroot = '/'.join(fadiff.split('/')[:-1])
            #     os.rename(fadiff, os.path.join(faroot,newfn))
            #     os.rename(os.path.join(faroot, newfn), os.path.join(derivs,sub_name,'fa',newfn))

    # print("... removing unneeded files ... \n")
    # while len(os.listdir(maindir)) > sub+4:
    #     for root, dirnames, filenames in os.walk(main_dir):
    #         if os.path.isfile(os.path.join(root,'.DS_Store')):
    #             os.remove(os.path.join(root,'.DS_Store'))
    #         if os.path.isdir(root):
    #             if not os.listdir(root):
    #                 os.rmdir(root)

    metadf.to_csv(os.path.join(main_dir,'participants.tsv'),sep='\t')
    return

if __name__=='__main__':

    re_format(sys.argv[1],'participants.csv')
