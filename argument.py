import argparse

def get_args():
    parser = argparse.ArgumentParser(description="influence score based weighting")
    parser.add_argument('--dataset', required=True, default='', choices=['adult', 'compas', 'bank'])
    parser.add_argument('--constraint', required=True, default='', choices=['dp', 'eo', 'eopp'])
    parser.add_argument('--method', required=True, default='', choices=['naive', 'influence', 'reweighting'])
    parser.add_argument('--epoch', required=True, default=0, type=int)
    parser.add_argument('--iteration', required=True, default=0, type=int)
    parser.add_argument('--scaler', default=None, type=float)
    parser.add_argument('--eta', default=None, type=float)

    args = parser.parse_args()
    if args.method == 'influence' and args.scaler is None:
        parser.error('influence requires --scaler')
    if args.method == 'reweighting' and args.eta is None:
        parser.error('reweighting requires --eta')
    return args
