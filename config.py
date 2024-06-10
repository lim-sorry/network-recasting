import argparse


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epoch', default=10, type=int)

    return parser.parse_args()