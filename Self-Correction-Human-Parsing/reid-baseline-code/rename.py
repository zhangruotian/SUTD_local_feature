import os
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description="rename files")
parser.add_argument('--dir',default='../market_lower_mask/pytorch', type=str, help='directory')
parser.add_argument("--rename_upper", action='store_true', default=False, help="rename the file xxx.png to xxx_upper.png")
parser.add_argument("--rename_lower", action='store_true', default=False, help="rename the file xxx.png to xxx_lower.png")
opt = parser.parse_args()

def rename(path):
    '''
    modify the file names excluding directories

    '''
    FileList = os.listdir(path)
    for files in tqdm(FileList):
        oldDirPath = os.path.join(path, files)
        if os.path.isdir(oldDirPath):
            rename(oldDirPath)
            continue
        fileName = os.path.splitext(files)[0]
        fileType = os.path.splitext(files)[1]
        if opt.rename_upper:
            newDirPath = os.path.join(path, fileName+'_upper' + fileType)
        if opt.rename_lower:
            newDirPath = os.path.join(path, fileName + '_lower' + fileType)
        os.rename(oldDirPath, newDirPath)

if __name__ == '__main__':
    rename(opt.dir)