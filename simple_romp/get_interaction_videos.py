from glob import glob
import os
import argparse
import json

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path", type=str, help='NTU RGB+D root path'
    )
    return parser.parse_args()

args = get_args()
file_list = glob(os.path.join(args.root_path, '*/*.avi'))
interaction_classes = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]

file_list = [one for one in file_list if int(one.split('A')[1][:3]) in interaction_classes]
file_num = len(file_list)
with open('file1.json', 'w') as f:
    json.dump(file_list[:file_num//4], f)
with open('file2.json', 'w') as f:
    json.dump(file_list[file_num//4:file_num//4*2], f)
with open('file3.json', 'w') as f:
    json.dump(file_list[file_num//4*2:file_num//4*3], f)
with open('file4.json', 'w') as f:
    json.dump(file_list[file_num//4*3:], f)