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
        self.dl = p['dl']
        assert type(self.dl) is int and self.dl>=0, "dl"
        self.comp = p['comp']
        assert type(self.comp) is str and self.comp in ['ISW', 'CMB'], "comp"
        self.cut = p['cut']
        self.aposcale = p['aposcale']
        
        self.map_file = p['map_file']
        assert type(self.map_file) is str
        self.wigner_file = p['wigner_file']
        assert type(self.wigner_file) is str

