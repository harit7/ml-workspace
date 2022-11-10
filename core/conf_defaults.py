
def set_defaults(conf):

    '''
    TODO: describe parameters and options
    Add new params with default values so that existing stuff works.
    '''
    conf.setdefault('random_seed',42)
    conf.setdefault("device","cpu")
    
    conf.setdefault("inference_conf",{"device":"cpu","shuffle":False})

    conf.setdefault("store_model_weights_in_mem",False)
    conf.setdefault("dump_result",False)

    train_conf = conf['training_conf']
    train_conf.setdefault('device',conf['device'])
    train_conf.setdefault('store_embedding',False)
    train_conf.setdefault('ckpt_load_path',None)
    train_conf.setdefault('save_ckpt',False)
    
    if(conf['inference_conf'] is None):
        conf['inference_conf'] = {}

    conf['inference_conf'].setdefault('device',conf['device'])



