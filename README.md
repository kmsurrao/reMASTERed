# reMASTERed  

Calculates contributions to pseudo-Cl for maps with correlated masks. In particular, it extends the **M**onte Carlo **A**podised **S**pherical **T**ransform **E**stimato**R** (MASTER) approach (https://arxiv.org/abs/astro-ph/0105302) to situations where the map and mask exhibit correlations, as is the case for point source masks used in CMB analyses. If you use the code in a publication, please cite https://arxiv.org/abs/2302.05436.  


## Running

For a user-inputted map and mask:  
Modify or create yaml file similar to [moto.yaml](moto.yaml)     
Run: ```python main.py --config=moto.yaml```  



For the ensemble-averaged threshold mask operation:    
Modify or create yaml file similar to [threshold_moto.yaml](threshold_moto.yaml)     
Run: ```python ensemble_threshold.py --config=threshold_moto.yaml```  


## Dependencies 

healpy    
pywigxjpf   
NaMaster   


## Outputs

The main output is remastered_curves.p, computed in [test_remastered.py](test_remastered.py). It is a list of seven 1D numpy arrays:  
- aa_ww_term: 1D numpy array of length ellmax+1, $\langle aa \rangle \langle ww \rangle$ term  
- aw_aw_term: 1D numpy array of length ellmax+1, $\langle aw \rangle \langle aw \rangle$ term  
- w_aaw_term: 1D numpy array of length ellmax+1, $\langle w \rangle \langle aaw \rangle$ term  
- a_waw_term: 1D numpy array of length ellmax+1, $\langle a \rangle \langle waw \rangle$ term  
- aaww_term: 1D numpy array of length ellmax+1, $\langle aaww \rangle$ term  
- directly_computed: 1D numpy array of length ellmax+1, directly computed power spectrum of masked map  
- remastered: 1D numpy array of length ellmax+1, reMASTERed result for power spectrum of masked map   


## Acknowledgments 

The bispectrum and trispectrum estimation code is adapted from https://github.com/oliverphilcox/PolyBin.  
