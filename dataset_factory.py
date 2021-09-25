from utkface_dataloader import UTKFaceDataset

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, transform=None, split='Train', target='Attractive', seed=0, skew_ratio=1., labelwise=False):

        if name == "utkface":
            root = './data/utkface_aligned_cropped/UTKFace_preprocessed'
            return UTKFaceDataset(root=root, split=split, transform=transform,
                                  labelwise=labelwise)