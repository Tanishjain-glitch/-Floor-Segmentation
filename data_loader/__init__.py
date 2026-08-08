from .cityscapes import CitySegmentation
from .floor import FloorDataset

datasets = {
    'citys': CitySegmentation,
    'floor': FloorDataset
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)