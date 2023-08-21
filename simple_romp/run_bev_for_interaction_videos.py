import argparse
import json
import os
import shutil
import subprocess

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id", type=str, help="Number of samples to generate", default=1
    )
    parser.add_argument(
        "--dataset_path", type=str, help='NTU RGB+D root path'
    )
    return parser.parse_args()

args = get_args()

with open('file{}.json'.format(args.id)) as f:
    data = json.load(f)
fin_save_dir = os.path.join(args.dataset_path, 'output', 'dir'+args.id)
if not os.path.exists(fin_save_dir):
    os.makedirs(fin_save_dir)
save_path = 'res{}/'.format(args.id)
for one_file in data:
    subprocess.run(['bev', '-m', 'video', '-o', save_path, '--render_mesh', '-i', one_file])
    shutil.move(os.path.join(save_path, 'video_results.npz'), os.path.join(fin_save_dir, one_file.split('/')[-1].replace('.avi','.npz')))
    shutil.rmtree(save_path)