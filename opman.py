# -*- coding: utf-8 -*-
"""
Created on Mon May 29 23:07:28 2023

@author: s.paramonov
"""
   
import os 
import json
import numpy as np
import oc_processing


dirnames = ['sst', 'sss', 'wind','ssha','chla']

     
LM = np.load(os.path.join('landmask.npy'))    
    
with open((os.path.join("dconfig.json")), "r") as dataset_config_file:
    config = json.load(dataset_config_file)    
        
    
d1 = len(config['lat'])
d2 = len(config['lon'])
d0 = len(config['dynamic_vars'])
    
day_dataset = np.zeros((d0,d1,d2))
            
    
totalcheck = 1
    
for dd in dirnames:
        
    dirfiles = []        
    files = os.listdir('data')        
    for fl in files:
        dirfiles.append(os.path.join('daydata', fl))
                               
    if (dd=='chla'):
            
        chladata, check_ = oc_processing.prepare_chla(dirfiles[0], LM, config) 
        day_dataset[5,:,:] = chladata
        totalcheck = totalcheck*check_
                         
    if (dd=='wind'):
        windspeeddata, winddirdata, check_ = oc_processing.prepare_wind_west(dirfiles, LM, config)
        day_dataset[2,:,:] = windspeeddata
        day_dataset[3,:,:] = winddirdata
        totalcheck = totalcheck*check_    
            
        
    if (dd=='sss'):
        sssdata, check_ = oc_processing.prepare_sss(dirfiles, LM, config)
        day_dataset[1,:,:] = sssdata
        totalcheck = totalcheck*check_ 
            
    
                    
    if (dd=='ssha'):
        sshadata, check_ = oc_processing.prepare_ssha_nrt_west(dirfiles, LM, config)
        day_dataset[4,:,:] = sshadata
        totalcheck = totalcheck*check_ 
            
        
    if (dd=='sst'):
        sstdata, check_ = oc_processing.prepare_sst_west(dirfiles[-1], LM, config)
        day_dataset[0,:,:] = sstdata
        totalcheck = totalcheck*check_ 
        
            
ds_name = os.path.join('daydata.npy')
np.save(ds_name, day_dataset)
        
exit(0)



    

    