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
    conflicting=True
    while conflicting:
        conflicting=False
        for f in range(len(files)):
            i=0
            for ff in range(len(files)):
                if files[f].split('/')[2]==files[ff].split('/')[2] and \
                        files[f].split('/')[4]==files[ff].split('/')[4] and \
                        files[f].split('/')[3]!=files[ff].split('/')[3]:
                    conflicting=True
                    i+=1
                    print(i)
                    newf = files[ff].split('/')
                    nf4_s = int(newf[4].split('_')[-1].split('.')[0])+i # if causes problems, change i back to 1
                    if nf4_s>59: nf4_s=nf4_s-59-1
                    if nf4_s<=9: nf4_new = '_'.join(newf[4].split('_')[:-1])+'_0'+str(nf4_s)+'.0'
                    else: nf4_new = '_'.join(newf[4].split('_')[:-1])+'_'+str(nf4_s)+'.0'
                    newf[4]=nf4_new
                    print(newf)
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
    ses=0
    for f in files:
        conflicting=False
        print(f.split('/')[4].split('.')[0])
        run = [k+1 for k in runsdict.keys() if runsdict[k]==f.split('/')[4].split('.')[0]][0]
        if int(run)<=9:
            run=f'0{run}'
        else:
            run=str(run)
        newfn = f'{sub}_run-{run}_{modality_fn}'
        newdir = os.path.join(maindir,sub,modality_dn)
        for ff in files:
            if f.split('/')[:5]==ff.split('/')[:5] and f.split('/')[:6]!=ff.split('/')[:6]:
                ses+=1
                newfn=newfn.split('.')[0]+f'_ses-{ses}'
                conflicting=True
        cmd = "dcm2niix -b y -o {} -f {} {}".format(newdir,newfn,f)
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if conflicting: os.rename(f,os.path.join(srcdata,sub,modality_src,f'{sub}_run-{run}_ses-{ses}/'))
        else: os.rename(f,os.path.join(srcdata,sub,modality_src,f'{sub}_run-{run}/'))
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
    metadf["Run_Num"]=""

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
        subject = os.path.join(main_dir,subj)
        if os.path.isdir(subject):
            print(f"File structure for {subj}")
            run=0
            sub+=1
            if sub<=9: sub_name = "sub-0{}".format(sub)
            else: sub_name = "sub-{}".format(sub)
            # print("\tOld ID: {} --> NewID: {}\n".format(subj, sub_name))
            metadf.loc[metadf.Subject.astype(int) == int(subj), 'Subject_Num'] = sub_name

            if not os.path.exists(os.path.join(main_dir,sub_name)):
                os.mkdir(os.path.join(main_dir,sub_name))
                os.mkdir(os.path.join(main_dir,sub_name,'anat'))
                os.mkdir(os.path.join(main_dir,sub_name,'dwi'))
                os.mkdir(os.path.join(main_dir,sub_name,'func'))


            # Compile lists of all filepaths to images of a given modality
            rawanats = re_run(list(roota for roota,dirnamesa,filenamesa in \
                        os.walk(subject) if len(roota.split('/'))==6 and 'MPRAGE' in \
                        roota.split('/')[3].upper()))
            rawdiffs = re_run(list(rootr for rootr,dirnamesr,filenamesr in \
                        os.walk(subject) if len(rootr.split('/'))==6 and 'DTI'    in \
                        rootr.split('/')[3].upper()))
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
            
            runs = dict(enumerate(sorteddates))

            im_ids={}
            for k,v in runs.items():
                im_ids[k+1]=[f[0].split('_I')[-1].split('.')[0] for r,d,f in os.walk(subject) if len(r.split('/'))==6 and r.split('/')[4].split('.')[0]==v]

            for k,v in im_ids.items():
                print(k,v)
                for id_str in v:
                    if k<=9:
                        metadf.loc[metadf['Image Data ID'].astype(str) == id_str, 'Run_Num'] = f'0{k}'
                    else:
                        metadf.loc[metadf['Image Data ID'].astype(str) == id_str, 'Run_Num'] = str(k)
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


    # print("... removing unneeded files ... \n")
    # while len(os.listdir(maindir)) > sub+4:
    #     for root, dirnames, filenames in os.walk(main_dir):
    #         if os.path.isfile(os.path.join(root,'.DS_Store')):
    #             os.remove(os.path.join(root,'.DS_Store'))
    #         if os.path.isdir(root):
    #             if not os.listdir(root):
    #                 os.rmdir(root)

    metadf.to_csv(os.path.join('/Volumes/ElementsExternal','participants.tsv'),sep='\t')
    return

if __name__=='__main__':

    re_format(sys.argv[1],'participants.csv')
