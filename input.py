import yaml
import numpy as np

##########################
# simple function for opening the file
def read_dict_from_yaml(yaml_file):
    assert(yaml_file != None)
    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    return config
##########################

##########################
"""
class that contains map info (and associated data), specifications, etc., and handles input
"""
class Info(object):
    def __init__(self, input_file):
        self.input_file = input_file
        p = read_dict_from_yaml(self.input_file)

        self.ellmax = p['ellmax']
        assert type(self.ellmax) is int and self.ellmax>=0, "ellmax"
        self.nside = p['nside']
        assert type(self.nside) is int and (self.nside & (self.nside-1) == 0) and self.nside != 0, "nside"
        assert self.ellmax <= 3*self.nside-1, "ellmax > 3*nside-1"
        
        self.map_file = p['map_file']
        assert type(self.map_file) is str
        self.mask_file = p['mask_file']
        assert type(self.mask_file) is str
        self.wigner_file = p['wigner_file']
        assert type(self.wigner_file) is str

        self.output_dir = p['output_dir']
        assert type(self.output_dir) is str
        self.save_files = p['save_files']
        self.plot = p['plot']

