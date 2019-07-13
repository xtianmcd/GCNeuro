import os
import sys
from builtins import any as b_any

"""
FIRST FILE OF PIPELINE
(Following multi-zip download from PPMI / other source into dir JustDownloaded/)

Combines multiple zip files into a single directory.
Before running: navigate to external HD (or location of files) in terminal via
    `cd /Volumes/<name of HD>`
To run (requires argument):
    `python neuro_format.py <name_of_destination_directory>`

Next file to run will be   `make_bids.py`
"""

def move_files(destdir,maindir):
    """
    Consolidates files from multi-zip download into a single directory for
    each subject.
    """

    print(f"Moving files from {maindir} and consolidating into {destdir}\n")
    if not os.path.exists(destdir):
        os.mkdir(destdir)

    # Generate list of full filepaths for each image file
    fileroots = list([os.path.join(root,filename) for root,dirnames,filenames \
                        in os.walk(maindir) if len(root.split('/'))==7 \
                        for filename in filenames])

    # Check if a directory already exists along the filepath to each image,
    # so that images from the same directory are consolidated together.
    # Otherwise, create the directory for the image in the destination dir.
    for filepath in fileroots:
        filedest = list([rootd for rootd,dirnamesd,filenamesd \
                        in os.walk(destination)])
        for step in range(3,len(filepath.split('/')[:-1])):
            if not b_any('/'.join(filepath.split('/')[3:step+1]).upper()==\
                        '/'.join(dest.split('/')[2:step]).upper() \
                        for dest in filedest if len(dest.split('/'))>=step):
                os.mkdir(os.path.join(destination,\
                        '/'.join(filepath.split('/')[3:step+1])))
        os.rename(filepath, os.path.join(destination,'/'.join(filepath.split('/')[3:])))
    print(f'... empyting {maindir} ...\n')

    # Empty the Downloads folder
    while  len(list([dwnld_dir for dwnld_dir in os.listdir(maindir) if os.path.isdir(os.path.join(maindir,dwnld_dir))]))>0: #len(os.listdir(maindir)) > 2:
        for root, dirnames, filenames in os.walk(maindir):
            if os.path.isfile(os.path.join(root,'.DS_Store')):
                os.remove(os.path.join(root,'.DS_Store'))
            if os.path.isdir(root):
                if not os.listdir(root):
                    os.rmdir(root)
    print(f"All files moved from {maindir} and consolidated into {destdir}\n\n")
    return

if __name__=='__main__':

    destination = f'{sys.argv[2]}/' # path to destination folder

    main_dir = f'./{sys.argv[1]}/' # path to download folder

    move_files(destination,main_dir)
