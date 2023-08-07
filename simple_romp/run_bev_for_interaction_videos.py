import argparse
import json
import subprocess

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id", type=int, help="Number of samples to generate", default=1
    )
    parser.add_argument(
        "--output_path", type=str, help='None'
    )
    return parser.parse_args()

args = get_args()

subprocess.run(['bev', '-m', 'video', '-o', 'res{}/'.format(args.id), '--render_mesh', '--json_file', 'file{}.json'.format(args.id), '--output_path', args.output_path])