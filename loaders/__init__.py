from loaders.make_dataset import DotsDataset, DensityDataset, CountDataset


def get_loader(args):
    """get_loader

    :param name:
    """
    return {
        'dot': DotsDataset,
        'density': DensityDataset,
        'count': CountDataset
    }[args.dataset]
