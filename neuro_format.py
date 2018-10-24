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

main_dir = "./JustDownloaded/" 

fileroots = list([os.path.join(root,filename) for root,dirnames,filenames \
                    in os.walk(main_dir) if len(root.split('/'))==7 \
                    for filename in filenames])
for filepath in fileroots:
    filedest = list([rootd for rootd,dirnamesd,filenamesd \
                    in os.walk(destination)])
    for step in range(3,len(filepath.split('/')[:-1])):
        if not b_any('/'.join(filepath.split('/')[3:step+1]).upper()==\
                    '/'.join(dest.split('/')[2:step]).upper() \
                    for dest in filedest if len(dest.split('/'))>=step):
            os.mkdir(os.path.join(destination,\
                    '/'.join(filepath.split('/')[3:step+1])))
    os.rename(filepath, destination+'/'.join(filepath.split('/')[3:]))

while len(os.listdir(main_dir)) > 2:
    for root, dirnames, filenames in os.walk(main_dir):
        if os.path.isfile(os.path.join(root,'.DS_Store')):
            os.remove(os.path.join(root,'.DS_Store'))
        if os.path.isdir(root):
            if not os.listdir(root):
                os.rmdir(root)
