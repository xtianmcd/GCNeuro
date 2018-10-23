import os
import sys
import pydcm2niix.pydcm2niix.conversion as niix

main_dir = "./" + sys.argv[1] + "/"
sub=1
run=1
subj_dirs = os.listdir(main_dir)
for subj in subj_dirs:
    sub_name = "sub-0{}".format(sub)
    if not os.path.exists(main_dir+sub_name):
        os.mkdir(main_dir+sub_name)
        os.mkdir(main_dir+sub_name+'/anat')
        os.mkdir(main_dir+sub_name+'/dwi')
    subject = main_dir+subj
    if os.path.isdir(subject):
        for root, dirnames, filenames in os.walk(subject):
            if len(root.split('/'))==6 and 'MPRAGE' in root.split('/')[3]:
                for filename in filenames:
                    if filename.endswith('.nii'):
                        newfn = sub_name+'_run-0{}_T1w.nii'.format(run)
                        os.rename(os.path.join(root,filename), os.path.join(root,newfn))
                        os.rename(os.path.join(root, newfn), os.path.join(main_dir,sub_name,'anat',newfn))
                run+=1
            elif len(root.split('/'))==6 and 'DTI' in root.split('/')[3]:
                newfn = os.path.join(main_dir,sub_name,sub_name+'_run-0{}_dwi.nii'.format(run))
                niix.dicom_to_nifti(root, newfn)
                run+=1

        sub+=1
    # print('\n')
    run = 1

while len(os.listdir(main_dir)) > sub+2:
    for root, dirnames, filenames in os.walk(main_dir):
        if os.path.isfile(os.path.join(root,'.DS_Store')):
            os.remove(os.path.join(root,'.DS_Store'))
        if os.path.isdir(root):
            if not os.listdir(root):
                os.rmdir(root)
