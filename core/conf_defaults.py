def set_defaults(conf):

    '''
    TODO: describe parameters and options
    Add new params with default values so that existing stuff works.
    '''
    conf.setdefault('random_seed',42)
    conf.setdefault("device","cpu")
    
    conf.setdefault("inference_conf",{"device":"cpu","shuffle":False})

    

    train_conf = conf['training_conf']
    train_conf.setdefault('device',conf['device'])
    
    train_conf.setdefault('ckpt_load_path',None)
    train_conf.setdefault('ckpt_save_path',None)
    train_conf.setdefault('embedding_save_path',None)
    
    
    if(conf['inference_conf'] is None):
        conf['inference_conf'] = {}

    conf['inference_conf'].setdefault('device',conf['device']) 


    
 