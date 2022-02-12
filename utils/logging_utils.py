import logging
import sys

def get_logger(log_file,stdout_redirect=False,level=logging.INFO):
    #log_file_name = os.path.basename(args.log_file).split(".")[0]+".log"
    logging.basicConfig(filename=log_file,filemode='w')
    logger = logging.getLogger()
    logger.setLevel(level)
    if(stdout_redirect):
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
    return logger 