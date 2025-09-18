import argparse
from data_utils import datasets
from data_utils.dataset_base import H5DatasetMapBase


def list_datasets():
    classes = datasets.get_subclasses(H5DatasetMapBase)
    return {c.__name__: c for c in classes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, nargs="?")
    parser.add_argument("-l", "--list", action="store_true", default=False,
                        help="list the available datasets")
    opt = parser.parse_args()

    if opt.list:
        classes = list_datasets()
        print("[INFO] Available datasets:")
        for i, c in enumerate(classes):
            print(" - [{}] {}".format(i, c))
        quit()
    
    dataset_name = opt.dataset
    if dataset_name is None:
        classes = list_datasets()
        if len(classes) == 0:
            print("[INFO] No datasets available.")
            quit()
        
        print("[INFO] Available datasets:")
        for i, c in enumerate(classes):
            print(" - [{}] {}".format(i, c))
        
        print()
        idx = input("[INFO] Please specify the dataset by its index: ")
        idx = int(idx)
        dataset_name = list(classes.keys())[idx]

    dataset_cls: type[H5DatasetMapBase] = getattr(datasets, dataset_name)
    dataset = dataset_cls.inst()
    dataset.visualize()

