import argparse

def get_args():
    parser = argparse.ArgumentParser(description="influence score based weighting")
    parser.add_argument('--dataset', required=True, default='', choices=['adult', 'compas', 'bank'])
    parser.add_argument('--constraint', required=True, default='', choices=['dp', 'eo', 'eopp'])
    parser.add_argument('--method', required=True, default='', choices=['naive', 'influence', 'reweighting'])
    parser.add_argument('--epoch', required=True, default=0, type=int)
    parser.add_argument('--iteration', required=True, default=0, type=int)

    args = parser.parse_args()
    return args
