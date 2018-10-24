import os
import sys
from builtins import any as b_any

"""
Combine multiple zip files into a single directory.
Before running: navigate to external HD (in terminal) via
    cd /Volumes/<name of HD>
To run:
    enter
        python neuro_format.py <name_of_destination_directory> in terminal.
"""
destination = './'+sys.argv[1]+'/' # path to destination folder
if not os.path.exists(destination):
    os.mkdir(destination)

main_dir = "./JustDownloaded/" # location of multi-zip downloaded images
# transferred = os.listdir(destination) # initial list of transferred ID names
# run = list(range(0,int(sys.argv[3])))
# expected_num = int(sys.argv[2])
# zipped_num = 0

fileroots = list([os.path.join(root,filename) for root,dirnames,filenames \
                    in os.walk(main_dir) if len(root.split('/'))==7 \
                    for filename in filenames])
for filepath in fileroots:
    # print(filepath)
    filedest = list([rootd for rootd,dirnamesd,filenamesd \
                    in os.walk(destination)])
    for step in range(3,len(filepath.split('/')[:-1])):
        # print('step={}'.format(step))
        # print(filepath.split('/')[3:(step+1)])
        # print(filedest)
        # print('/'.join(filepath.split('/')[3:step+1]))
        # print(list(['/'.join(dest.split('/')[2:step]) \
        #             for dest in filedest if len(dest.split('/'))>=step]))
        # print(b_any('/'.join(filepath.split('/')[3:step+1])=='/'.join(dest.split('/')[2:step]) \
                    # for dest in filedest if len(dest.split('/'))>=step))
        if not b_any('/'.join(filepath.split('/')[3:step+1]).upper()==\
                    '/'.join(dest.split('/')[2:step]).upper() \
                    for dest in filedest if len(dest.split('/'))>=step):
            os.mkdir(os.path.join(destination,\
                    '/'.join(filepath.split('/')[3:step+1])))
    os.rename(filepath, destination+'/'.join(filepath.split('/')[3:]))

#     sub_dir = main_dir+ppmi # temp var for the individual zipped folder
#     if os.path.isdir(sub_dir):
#         for sub in os.listdir(sub_dir): # iterate through ID's
#             subject = sub_dir+'/'+sub
#             if os.path.isdir(subject):
#                 if sub in transferred:
#                     run=1
#                     for mod in subject:
#                         modality = subject+'/'+mod
#                         if os.path.isdir(modality):
#
#                         for r in modality:
#                             for root, di
#                 if sd in transferred:
#                     for r in run:
#                         if sd+"_"+str(r) not in os.listdir(destination):
#                             os.rename(subDir+"/"+sd, destination+"/"+sd+"_"+str(suff))
#                             break
#
#                 else:
#                     transferred.append(sd)
#                     os.rename(subDir+"/"+sd, destination+"/"+sd)
#                 zipped_num+=1
#
# if os.path.isfile(os.path.join(destination, '.DS_Store')):
#     transferred_num = len(os.listdir(destination))-3
# else: transferred_num = len(os.listdir(destination))-2
#
# if expected_num==zipped_num: print("Transferred the expected number of image sets")
# else: print("ERROR: not the expected number of image sets in zipped dirs")
# if expected_num==transferred_num: print("Consolidated the expected number of image sets")
# else: print("ERROR: not the expected number of image sets in final directory")
# if zipped_num==transferred_num: print("Consolidated all transferred image sets")
# else: print("ERROR: not all image sets consolidated")

while len(os.listdir(main_dir)) > 2:
    for root, dirnames, filenames in os.walk(main_dir):
        if os.path.isfile(os.path.join(root,'.DS_Store')):
            os.remove(os.path.join(root,'.DS_Store'))
        if os.path.isdir(root):
            if not os.listdir(root):
                os.rmdir(root)
