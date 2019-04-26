import os
import sys
import json
import subprocess

"""
SECOND FILE IN PIPELINE (following neuro_format.py)

Changes file structure from to be compatible with BIDS format;
image sources other than PPMI will probably need changes;
BIDS is the expected format for libraries such as dcm2niix and is quite readable

To run, requires argument:

    `python make_bids.py <name_of_main_directory_with_images>`

Next file to run is `automate_preproc.py`
"""

def re_format(maindir):
    main_dir = "./" + maindir + "/"
    subj_dirs = os.listdir(main_dir)

    jsondict = {
        "Name":maindir,
        "BIDSVersion": "1.0.0rc1"
    }

    with open(main_dir+'dataset_description.json', 'w') as outfile:
        json.dump(jsondict, outfile)

    srcdata = main_dir+'sourcedata/'
    if not os.path.exists(srcdata):
        os.mkdir(srcdata)
    derivs = main_dir+'derivatives/'
    if not os.path.exists(derivs):
        os.mkdir(derivs)

    sub=0
    for subj in subj_dirs:
        print(f"File structure for {subj}")
        subject = main_dir+subj
        if os.path.isdir(subject):
            run=0
            sub+=1
            if sub<=9: sub_name = "sub-0{}".format(sub)
            else: sub_name = "sub-{}".format(sub)
            print("\tOld ID: {} --> NewID: {}\n".format(subj, sub_name))
            with open(os.path.join(main_dir,'ID_mapping.txt')) as ids:
                ids.write("Old ID: {} --> NewID: {}\n".format(subj, sub_name))
            if not os.path.exists(main_dir+sub_name):
                os.mkdir(main_dir+sub_name)
                os.mkdir(main_dir+sub_name+'/anat')
                os.mkdir(main_dir+sub_name+'/dwi')

            non_nii = {
                "anat": list([os.path.join(roota,filenamea) for roota,dirnamesa,filenamesa in os.walk(subject) if len(roota.split('/'))==6 and 'T1' in roota.split('/')[3] for filenamea in filenamesa if not filenamea.endswith('.nii')]),
                "fa"  : list(os.path.join(rootf,filenamef) for rootf,dirnamesf,filenamesf in os.walk(subject) if len(rootf.split('/'))==6 and 'FA' in rootf.split('/')[3] for filenamef in filenamesf if not filenamef.endswith('.nii'))
            }
            if non_nii["anat"]: print("WARNING: some T1 file(s) not .nii format in {} - {}".format(subj, non_nii['anat']))
            if non_nii["fa"]: print("WARNING: some FA file(s) not .nii format in {} - {}".format(subj, non_nii['fa']))

            anats = list([os.path.join(roota,filenamea) for roota,dirnamesa,filenamesa in os.walk(subject) if len(roota.split('/'))==6 and 'T1' in roota.split('/')[3] for filenamea in filenamesa if filenamea.endswith('.nii')])
            rawdiffs = list(rootr for rootr,dirnamesr,filenamesr in os.walk(subject) if len(rootr.split('/'))==6 and 'DTI' in rootr.split('/')[3])
            fadiffs = list(os.path.join(rootf,filenamef) for rootf,dirnamesf,filenamesf in os.walk(subject) if len(rootf.split('/'))==6 and 'FA' in rootf.split('/')[3] for filenamef in filenamesf if filenamef.endswith('.nii'))
            for anat in anats:
                run +=1
                newfn = sub_name+'_run-0{}_T1w.nii'.format(run)
                anatroot = '/'.join(anat.split('/')[:-1])
                os.rename(anat, os.path.join(anatroot,newfn))
                os.rename(os.path.join(anatroot, newfn), os.path.join(main_dir,sub_name,'anat',newfn))
            run=0
            for rawdiff in rawdiffs:
                run+=1
                newfn = sub_name+'_run-0{}_dwi'.format(run)
                newdir = os.path.join(main_dir,sub_name,'dwi/')
                cmd = "dcm2niix -b y -o {} -f {} {}".format(newdir,newfn,rawdiff)
                process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                if not os.path.exists(srcdata+sub_name):
                    os.mkdir(srcdata+sub_name)
                os.rename(rawdiff,srcdata+sub_name+'/rawdcm_run-0{}/'.format(run))
            run=0
            for fadiff in fadiffs:
                run+=1
                if not os.path.exists(derivs+sub_name):
                    os.mkdir(derivs+sub_name)
                    os.mkdir(derivs+sub_name+'/fa/')
                newfn = sub_name+'_run-0{}_dwi.nii'.format(run)
                faroot = '/'.join(fadiff.split('/')[:-1])
                os.rename(fadiff, os.path.join(faroot,newfn))
                os.rename(os.path.join(faroot, newfn), os.path.join(derivs,sub_name,'fa',newfn))

    print("... removing unneeded files ... \n")
    while len(os.listdir(main_dir)) > sub+4:
        for root, dirnames, filenames in os.walk(main_dir):
            if os.path.isfile(os.path.join(root,'.DS_Store')):
                os.remove(os.path.join(root,'.DS_Store'))
            if os.path.isdir(root):
                if not os.listdir(root):
                    os.rmdir(root)
    return

if __name__=='__main__':

    reformat(sys.argv[1])
