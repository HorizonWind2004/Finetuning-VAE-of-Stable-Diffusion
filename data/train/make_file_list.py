import os
import glob
with open('filelist.txt', 'w') as f:
    file_name = glob.glob('data/0/*.png')
    for _ in file_name:
        f.write(str(_.split('data/')[-1]) + '\n')