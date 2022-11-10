from datasets.mnist import MNISTData
from datasets.mnist_sklearn import MNISTData_sklearn
from datasets.cub_birds import CUB_BirdsData
from datasets.synthetic_general import *
from datasets.synthetic_moon import *
from datasets.uniform_unit_ball import *
from datasets.xor_balls import *
from datasets.cifar10 import *


def load_dataset(conf):
    data_conf = conf['data_conf']
    dataset_name = data_conf['dataset']
    
    if(dataset_name == 'general_synthetic'):
        
        return GeneralSynthetic(data_conf)
    elif(dataset_name == 'synth_moon'):
        return SyntheticMoon(data_conf)
    
    elif(dataset_name == 'mnist'):
        return MNISTData(data_conf)

    elif(dataset_name == 'mnist_sklearn'):
        return MNISTData_sklearn(data_conf)

    elif(dataset_name == 'cifar10'):
        return Cifar10Data(data_conf)
    
    elif(dataset_name == 'cub_birds'):
        return CUB_BirdsData(data_conf)
    
    elif(dataset_name == 'unif_unit_ball'):

        return UniformUnitBallDataset(data_conf)
    
    elif(dataset_name == 'xor_balls'):

        return XORBallsDataset(data_conf)
    
    else:
        print('Datset {} Not Defined'.format(dataset_name))
        return None

