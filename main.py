import sys 
import os

#sys.path.append('../')
#sys.path.append('../../')

from utils.logging_utils import * 
from datasets import dataset_factory 
from datasets.dataset_utils import * 
from utils.common_utils import * 
from core.passive_learning import * 

# configuration
config_file ='./conf/basic-ml/MNIST_lenet.yaml'

def main():
    
    conf = load_yaml_config_std(config_file)
    conf['base_dir'] = '.'
    conf = OmegaConf.create(conf)

    logger = get_logger(conf.output.log_file_path,stdout_redirect=True,level=logging.DEBUG)

    set_seed(conf.random_seed)
    # get data
    ds = dataset_factory.load_dataset(conf)
    ds.build_dataset()
    train_set,val_set_std = randomly_split_dataset(ds,1- conf.data_conf.val_fraction)

    test_ds = ds.get_test_datasets()

    pas_learn = PassiveLearning(conf, train_set, ds_val=val_set_std,ds_test=test_ds,logger=logger)
    out = pas_learn.run()
    
if __name__ == '__main__':


    

    main()