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
    def __init__(self, input_file, mask_provided=True):
        self.input_file = input_file
        p = read_dict_from_yaml(self.input_file)

        #If mask needs to be created
        if not mask_provided:
            self.nsims = p['nsims']
            assert type(self.nsims) is int and self.nsims>=0, "nsims"
            self.ellmax = p['ellmax']
            assert type(self.ellmax) is int and self.ellmax>=0, "ellmax"
            self.ell_sum_max = p['ell_sum_max']
            assert type(self.ell_sum_max) is int and self.ell_sum_max>=self.ellmax, "ell_sum_max"
            self.remove_high_ell_power = p['remove_high_ell_power']
            self.nside = p['nside']
            assert type(self.nside) is int and (self.nside & (self.nside-1) == 0) and self.nside != 0, "nside"
            assert self.ellmax <= 3*self.nside-1, "ellmax > 3*nside-1"
            self.nside_for_masking = p['nside_for_masking']
            assert type(self.nside_for_masking) is int and (self.nside_for_masking & (self.nside_for_masking-1) == 0) and self.nside_for_masking != 0, "nside_for_masking"
            self.comp = p['comp']
            assert type(self.comp) is str and self.comp in ['ISW', 'CMB'], "comp"
            self.cut = p['cut']
            self.aposcale = p['aposcale']
            self.cut_high = p['cut_high']
            self.cut_low = p['cut_low']
            
            self.map_file = p['map_file']
            assert type(self.map_file) is str
            self.wigner_file = p['wigner_file']
            assert type(self.wigner_file) is str

            self.output_dir = p['output_dir']
            assert type(self.output_dir) is str
            self.save_files = p['save_files']
            self.plot = p['plot']
        
        #If doing single realization with specified map and mask
        else:
            self.ellmax = p['ellmax']
            assert type(self.ellmax) is int and self.ellmax>=0, "ellmax"
            self.ell_sum_max = p['ell_sum_max']
            assert type(self.ell_sum_max) is int and self.ell_sum_max>=self.ellmax, "ell_sum_max"
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



