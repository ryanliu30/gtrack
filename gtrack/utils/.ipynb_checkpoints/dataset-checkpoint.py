import glob
from weaver.utils.dataset import SimpleIterDataset
from torch.utils.data import IterableDataset
from typing import Optional
import yaml

class ConcatDatasets(IterableDataset):
    def __init__(
        self, 
        dataset1: IterableDataset,
        dataset2: IterableDataset
    ) -> IterableDataset:
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        
    def __iter__(self):
        return zip(self.dataset1, self.dataset2)

def load_files(flist):
    """
    a helper function to list all root files
    """
    file_dict = {}
    for f in flist:
        if ':' in f:
            name, fp = f.split(':')
        else:
            name, fp = '_', f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)
    return file_dict

def get_dataset(
    stage: str,
    data_config_file: Optional[str] = "/global/homes/r/ryanliu/JetRep/configs/dataset_config.yaml",
    loader_config_file: Optional[str] = "/global/homes/r/ryanliu/JetRep/configs/loader_config.yaml",
) -> SimpleIterDataset:
    """
    a helper function to build dataset
    arguments:
        stage: a string specifies which dataset to load
        data_config_file: path to the data pre-processing configuration
        loader_config_file: path to the data loading/splitting configuration
    returns:
        a `SimpleIterDataset`
    """
    if not stage in ["train", "train_downstream", "val", "test"]:
        raise ValueError("Undefined dataset type (train, train_downstream, val, test)")
    with open(loader_config_file) as f:
        flist = yaml.load(f, Loader=yaml.FullLoader)[f"data_{stage}"]
    file_dict = load_files(flist)
    
    dataset = SimpleIterDataset(
        file_dict, 
        data_config_file,
        for_training=(stage != "test"),
        remake_weights=True,
        fetch_step=0.001,
        infinity_mode=False,
        name=stage
    )
    
    return dataset