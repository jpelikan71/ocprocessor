# -*- coding: utf-8 -*-
"""
Created on Wed May 31 21:05:35 2023

@author: s.paramonov
"""

import os
import xarray as xa
import numpy as np
import json

from scipy.ndimage import interpolation
from scipy import interpolate 

#vkoef = 0.4
#hkoef = 0.4

def quadrant(bbox):
    latmax_ = bbox[1][0]
    latmin_ = bbox[0][0]
    lonmax_ = bbox[1][1]
    lonmin_ = bbox[0][1]
    
    return latmax_,latmin_,lonmax_,lonmin_




def prepare_chla(chlafile, lm, config_):
    latmax,latmin,lonmax,lonmin = quadrant(config_['bbox']) 
       
    Lats_ = config_['lat']
    Lons_ = config_['lon']

    Lats_ = np.asarray(Lats_)
    Lons_ = np.asarray(Lons_) 
    
    try:
    
        with xa.open_dataset(chlafile) as data_0:
        
            sslice = data_0.sel(lat=slice(latmax,latmin), lon=slice(lonmin,lonmax))
            lats1 = sslice['lat'].values
            lons1 = sslice['lon'].values
            data0 =  sslice['chlor_a'].values 
    except ValueError: 
        
        return None, 0
    
    
    # ADJUST MASK
    vkoef = data0.shape[0] / lm.shape[0] 
    hkoef =  data0.shape[1]/ lm.shape[1] 
           
    lmsqueezed = interpolation.zoom(lm, (vkoef,hkoef), order = 0)
        
    landwhere = np.where(lmsqueezed==0)
    
    chla_markered_lm = np.copy(data0)
    chla_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(data0))
    hnan = np.where(np.isnan(data0))
    
    lmarked_nan = np.where(np.isnan(chla_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, data0.shape[1]),
                                 np.arange(0, data0.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_chla = data0[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    #sst_interp = np.copy(data2)
    # generate interpolated array of z-values
    chla_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_chla.ravel(),
                                     P, method='linear')  
    
    #print(sst_interp_data.shape, P.shape, (nnan[0].shape), data2.shape)
    data_it1 = np.copy(data0)
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    data_it1[lmarked_nan] = chla_interp_iter1
 
    
    chla_markered_lm = np.copy(data_it1)
    chla_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(data_it1))
    hnan = np.where(np.isnan(data_it1))
    
    lmarked_nan = np.where(np.isnan(chla_markered_lm))
    nonlmarked_nan = np.where(~np.isnan(chla_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, data_it1.shape[1]),
                                 np.arange(0, data_it1.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_chla = data_it1[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    #sst_interp = np.copy(data2)
    # generate interpolated array of z-values
    chla_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_chla.ravel(),
                                     P, method='nearest')  
    
    test = chla_markered_lm[nonlmarked_nan]

    #print(sst_interp_data.shape, P.shape, (nnan[0].shape), data2.shape)
    data_it2 = np.copy(data_it1)
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    data_it2[lmarked_nan] = chla_interp_iter2
    chla_markered_lm[lmarked_nan] = chla_interp_iter2
    
    vkoef = Lats_.shape[0] /  chla_markered_lm.shape[0]
    hkoef = Lons_.shape[0] /  chla_markered_lm.shape[1]

    CHLA = interpolation.zoom(chla_markered_lm, (vkoef,hkoef), order = 0)
    
    checknan = np.where(~np.isnan(CHLA))[0].shape[0]
    #print('sstnan', checknan)
       
    return CHLA, checknan   


def prepare_chla_splitted(chlafile, lm, config_):
    latmax,latmin,lonmax,lonmin = quadrant(config_['bbox'])

    Lats_ = config_['lat']
    Lons_ = config_['lon']

    Lats_ = np.asarray(Lats_)
    Lons_ = np.asarray(Lons_)
    
   
    
    lonmin1= lonmin
    lonmax1 = 180
    lonmin2 = -180
    lonmax2 = lonmax
    
    
            
    try:
        with xa.open_dataset(chlafile) as chla0:
            
            sslice = chla0.sel(lat=slice(latmax,latmin), lon=slice(lonmin1,lonmax1))
            lats0 = sslice['lat'].values
            
            lons0 = sslice['lon'].values
            data01 = sslice['chlor_a'].values
            
    except ValueError: 
        
        return None, 0
                
    try:
        with xa.open_dataset(chlafile) as chla1:
            
            sslice = chla1.sel(lat=slice(latmax,latmin), lon=slice(lonmin2,lonmax2))
            lats1 = sslice['lat'].values
            lons1 = sslice['lon'].values
            data02 = sslice['chlor_a'].values[:,:]
           
    except ValueError: 
        
        return None, 0
            
    data0 = np.hstack((data01, data02))

       
    #prepare landmask
    vkoef = data0.shape[0]/lm.shape[0] 
    hkoef =  data0.shape[0]/lm.shape[0] 
    
    lmsqueezed = interpolation.zoom(lm, (vkoef,hkoef), order = 0)
    
    landwhere = np.where(lmsqueezed==0)
    
    chla_markered_lm = np.copy(data0)
    chla_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(data0))
    hnan = np.where(np.isnan(data0))
    
    if (nnan[0].shape[0]==0):
         return None, 0
        
    
    lmarked_nan = np.where(np.isnan(chla_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, data0.shape[1]),
                                 np.arange(0, data0.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_chla = data0[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    chla_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_chla.ravel(),
                                     P, method='linear')  
    
    #print(sst_interp_data.shape, P.shape, (nnan[0].shape), data2.shape)
    data_it1 = np.copy(data0)
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    data_it1[lmarked_nan] = chla_interp_iter1
    
       
    # fill with nearest:
    chla_markered_lm = np.copy(data_it1)
    chla_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(data_it1))
    #hnan = np.where(np.isnan(data_it1))
    
    lmarked_nan = np.where(np.isnan(chla_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, data_it1.shape[1]),
                                 np.arange(0, data_it1.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_chla = data_it1[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    #sst_interp = np.copy(data2)
    # generate interpolated array of z-values
    chla_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_chla.ravel(),
                                     P, method='nearest')  
    
    #print(sst_interp_data.shape, P.shape, (nnan[0].shape), data2.shape)
    data_it2 = np.copy(data_it1)
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    data_it2[lmarked_nan] = chla_interp_iter2    
    chla_markered_lm[lmarked_nan] = chla_interp_iter2
            
    Vkoef = Lats_.shape[0] / chla_markered_lm.shape[0]
    Hkoef = Lons_.shape[0] / chla_markered_lm.shape[1]
    
    CHLA = interpolation.zoom(chla_markered_lm, (Vkoef,Hkoef), order = 0)
    
    checknan = np.where(~np.isnan(CHLA))[0].shape[0]
    #print('sstnan', checknan)
       
    return CHLA, checknan



# usage: collid =  'C2270392799-POCLOUD', model data
def prepare_ssha_model_splitted(sshafile, lm, config_):
    latmax,latmin,lonmax_,lonmin_ = quadrant(config_['bbox'])
    
    #sshafile = os.path.join('daydata','ssha','ssh_grids_v2205_2022092812.nc') 
    Lats_ = config_['lat']
    Lons_ = config_['lon']

    Lats_ = np.asarray(Lats_)
    Lons_ = np.asarray(Lons_)
    
    lonmin = (lonmin_)%360
    lonmax = (lonmax_)%360
         
    try:
        with xa.open_dataset(sshafile) as ssha0:
            
            sslice = ssha0.sel(Latitude=slice(latmin,latmax), Longitude=slice(lonmin,lonmax))
            lats0 = sslice['Latitude'].values
            lons0 = sslice['Longitude'].values
            data0 = sslice['SLA'].values[0,:,:]
           
    except ValueError: 
        
        return None, 0

            
    data0 = np.flip(data0,axis = 0) 

        
    #prepare landmask
    vkoef = data0.shape[0]/lm.shape[0] 
    hkoef =  data0.shape[0]/lm.shape[0] 
    
    lmsqueezed = interpolation.zoom(lm, (vkoef,hkoef), order = 0)
    
    landwhere = np.where(lmsqueezed==0)
    
    ssha_markered_lm = np.copy(data0)
    ssha_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(data0))
    hnan = np.where(np.isnan(data0))
    
    if (nnan[0].shape[0]==0):
         return None, 0
        
    
    lmarked_nan = np.where(np.isnan(ssha_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, data0.shape[1]),
                                 np.arange(0, data0.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_ssha = data0[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    ssha_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_ssha.ravel(),
                                     P, method='linear')  
    
    #print(sst_interp_data.shape, P.shape, (nnan[0].shape), data2.shape)
    data_it1 = np.copy(data0)
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    data_it1[lmarked_nan] = ssha_interp_iter1
    
       
    # fill with nearest:
    ssha_markered_lm = np.copy(data_it1)
    ssha_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(data_it1))
    #hnan = np.where(np.isnan(data_it1))
    
    lmarked_nan = np.where(np.isnan(ssha_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, data_it1.shape[1]),
                                 np.arange(0, data_it1.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_ssha = data_it1[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    #sst_interp = np.copy(data2)
    # generate interpolated array of z-values
    ssha_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_ssha.ravel(),
                                     P, method='nearest')  
    
    #print(sst_interp_data.shape, P.shape, (nnan[0].shape), data2.shape)
    data_it2 = np.copy(data_it1)
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    data_it2[lmarked_nan] = ssha_interp_iter2    
    ssha_markered_lm[lmarked_nan] = ssha_interp_iter2
            
    Vkoef = Lats_.shape[0] / ssha_markered_lm.shape[0]
    Hkoef = Lons_.shape[0] / ssha_markered_lm.shape[1]
    
    SSHA = interpolation.zoom(ssha_markered_lm, (Vkoef,Hkoef), order = 0)
    
    checknan = np.where(~np.isnan(SSHA))[0].shape[0]
    #print('sstnan', checknan)
       
    return SSHA, checknan

def prepare_ssha_nrt_east(sshafiles, lm, config_):
    # collection = 'C2075141559-POCLOUD'
    
    latmax,latmin,lonmax_,lonmin_ = quadrant(config_['bbox']) 
   
             
    Lats_ = config_['lat']
    Lons_ = config_['lon'] 
    
    
    
    Lats_ = np.asarray(Lats_)
    Lons_ = np.asarray(Lons_) 
    
    lonmin = (lonmin_)%360
    lonmax = (lonmax_)%360
    

    deltalat = 0.5
    deltalon = 0.5
    
    lat_points = int(np.ceil((latmax - latmin)/deltalat))
    lon_points = int(np.ceil((lonmax - lonmin)/deltalon))
    
    #newlons_ = np.arange(lon_points)
    #newlats_ = np.arange(lat_points)
    
    Lons = np.linspace(lonmin, lonmax, lon_points)
    Lats = np.linspace(latmin, latmax, lat_points)
        
    
     
    numlats = Lats.shape[0]
    numlons = Lons.shape[0]
    
       
    patch_ssha = np.zeros((len(sshafiles), numlats,numlons))
    patch_ssha[:,:,:] = np.nan
    
    
    checknan = 1  
    
    
            
    for i in range(len(sshafiles)):
        #os.path.join('winds_tests', 'wind_test2')
      
        ssha0 = sshafiles[i]
        
        with xa.open_dataset(ssha0) as data_0:
            
            # Part 1
                
            lats = data_0['lat'].values
            lons = data_0['lon'].values
                        
            ssha_all = data_0['ssha'].values
            
                #
        wlat = np.where((lats > latmin) & (lats < latmax) ) 
                                
        lats1 = lats[wlat]
        lons1 = lons[wlat]
        slice1_ssha = ssha_all[wlat]
        
                
               
        wlon = np.where((lons1 > lonmin) & (lons1 < lonmax) )
        #wlon = np.where((lons1 > lonmin) )
        lats2 = lats1[wlon]
        lons2 = lons1[wlon]
        
        ssha_s = slice1_ssha[wlon]
        
                
        where_are_NaNs = np.isnan(ssha_s)
        ssha = ssha_s[~where_are_NaNs]
       
        lats3 = lats2[~where_are_NaNs]
        lons3 = lons2[~where_are_NaNs]
       
                
        if(ssha.shape[0]>0):
                                            
            idlons = ((lons3 - lonmin) // deltalon).astype(int)
            idlats = ((lats3 - latmin) // deltalat).astype(int)
                                
                    #Wind[idlats,idlons] = wind
            patch_ssha[i,idlats,idlons] = ssha
            
               
    #Windspeed_ = np.nanmean(patch_wspeed, axis = 0) 
   
    Ssha = np.nanmean(patch_ssha, axis = 0)
    
        
    Ssha = np.flip(Ssha, axis = 0)
    
    
            
    #---------- apply interpolation
    
    vkoef = Ssha.shape[0]/lm.shape[0] 
    hkoef = Ssha.shape[1]/lm.shape[1] 
    
    lmsqueezed = interpolation.zoom(lm, (vkoef,hkoef), order = 0)
    
    landwhere = np.where(lmsqueezed==0)
    
    ssha_markered_lm = np.copy(Ssha)
    ssha_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(Ssha))
    
    if (nnan[0].shape[0]==0):
         return None, None, 0
    #hnan = np.where(np.isnan(Wind))
    
    lmarked_nan = np.where(np.isnan(ssha_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, Ssha.shape[1]),
                                np.arange(0, Ssha.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_ssha = Ssha[nnan]
    
    
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    # generate interpolated array of z-values
    ssha_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_ssha.ravel(),
                                     P, method='linear')   
    
    
    
    #data_it1 = np.copy(Windspeed)
    ssha_it1 = np.copy(Ssha)
    
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    ssha_it1[lmarked_nan] = ssha_interp_iter1
    
    
    
    # 2nd iteration
    ssha_markered_lm = np.copy(ssha_it1)
    ssha_markered_lm[landwhere] = -1
    
    
    nnan = np.where(~np.isnan(ssha_it1))
    #hnan = np.where(np.isnan(data_it1))
    
    lmarked_nan = np.where(np.isnan(ssha_markered_lm))
    #nonlmarked_nan = np.where(~np.isnan(wind_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, ssha_it1.shape[1]),
                                 np.arange(0, ssha_it1.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_ssha =ssha_it1[nnan]
   
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    #sst_interp = np.copy(data2)
    # generate interpolated array of z-values
    ssha_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_ssha.ravel(),
                                     P, method='nearest')  
    
     
    
    ssha_markered_lm[lmarked_nan] = ssha_interp_iter2
   
            
    # Expand to the grid:
    Vkoef = Lats_.shape[0] / ssha_markered_lm.shape[0]
    Hkoef = Lons_.shape[0] / ssha_markered_lm.shape[1]
    
    SSHA = interpolation.zoom(ssha_markered_lm, (Vkoef,Hkoef), order = 0)
    

    checknan = np.where(~np.isnan(SSHA))[0].shape[0]
    
       
    return SSHA,checknan

#------------------------------------------------------
#-------------------------------------------------------------------------
def prepare_ssha_nrt_west(sshafiles, LM, config_):
    
    
    latmax,latmin,lonmax_,lonmin_ = quadrant(config_['bbox']) 
             
    Lats_ = config_['lat']
    Lons_ = config_['lon'] 
    
    
    Lons0_ = config_['lon']

    Lats_ = np.asarray(Lats_)
    Lons0_ = np.asarray(Lons0_)
    
    ind0 = np.where(Lons0_>0)[0][0]
        
    
    
    lm1 = LM[:,:ind0]
    lm2 = LM[:,ind0:]
    lm_ = [lm1,lm2]
    
    LONS_ = [Lons0_[:ind0], Lons0_[ind0:]] 
    lonmin_ = (lonmin_)%360
    lonmax_ = (lonmax_)%360
    
   
    
    #(360 - 0.00001)
    Lonrange = [[lonmin_,360 - 0.00001],[0, lonmax_] ]

    deltalat = 0.5
    deltalon = 0.5
    
    Lats_ = np.asarray(Lats_)
    Lons_ = np.asarray(Lons_) 
    
    checknan = 1  
    SSHA_ = []
    
    for l in range(2):
        
        lm = lm_[l]
        #Lons_ = LONS_[l]
        lonmin = Lonrange [l][0]
        lonmax = Lonrange [l][1]
         
        deltalat = 0.5
        deltalon = 0.5
        
     
        
        #+++++++++++++
        
        lat_points = int(np.ceil((latmax - latmin)/deltalat))
        lon_points = int(np.ceil((lonmax - lonmin)/deltalon))
        
        #newlons_ = np.arange(lon_points)
        #newlats_ = np.arange(lat_points)
        
        Lons = np.linspace(lonmin, lonmax, lon_points)
        Lats = np.linspace(latmin, latmax, lat_points)
            
        
         
        numlats = Lats.shape[0]
        numlons = Lons.shape[0]
        
               
        patch_ssha = np.zeros((len(sshafiles), numlats,numlons))
        patch_ssha[:,:,:] = np.nan
       
        
        
        # reproject lons:
                  
        for i in range(len(sshafiles)):
            #os.path.join('winds_tests', 'wind_test2')
          
            
            ssha0 = sshafiles[i]
            
            with xa.open_dataset(ssha0) as data_0:
                #print(data_0)
                #print(aa)
                # Part 1
                    
                lats = data_0['lat'].values
                lons = data_0['lon'].values 
                
                            
                ssha_all = data_0['ssha'].values
                
                   
            wlat = np.where((lats > latmin) & (lats < latmax) ) 
                                    
            lats1 = lats[wlat]
            lons1 = lons[wlat]
            
            slice1_ssha = ssha_all[wlat]
            
                    
                   
            wlon = np.where((lons1 > lonmin) & (lons1 < lonmax) )
            #wlon = np.where((lons1 > lonmin) )
            lats2 = lats1[wlon]
            lons2 = lons1[wlon]
            
            ssha_s = slice1_ssha[wlon]
            
                    
            where_are_NaNs = np.isnan(ssha_s)
            ssha = ssha_s[~where_are_NaNs]
           
            lats3 = lats2[~where_are_NaNs]
            lons3 = lons2[~where_are_NaNs]
            #print(i, 'nans')
            #print(where_are_NaNs.shape)    
                    
            if(ssha.shape[0]>0):
                                                
                idlons = ((lons3 - lonmin) // deltalon).astype(int)
                idlats = ((lats3 - latmin) // deltalat).astype(int)
                                    
                        #Wind[idlats,idlons] = wind
                patch_ssha[i,idlats,idlons] = ssha
                
                   
        #Windspeed_ = np.nanmean(patch_wspeed, axis = 0) 
            
        Ssha2 = np.nanmean(patch_ssha, axis = 0)
       
            
        Ssha2 = np.flip(Ssha2, axis = 0)
        SSHA_.append(Ssha2)
        
    Ssha = np.hstack((SSHA_[0], SSHA_[1]))        
    #---------- apply interpolation
    #---------- apply interpolation
    
    vkoef = Ssha.shape[0]/LM.shape[0] 
    hkoef = Ssha.shape[1]/LM.shape[1] 
    
    lmsqueezed = interpolation.zoom(LM, (vkoef,hkoef), order = 0)
    
    landwhere = np.where(lmsqueezed==0)
    
    ssha_markered_lm = np.copy(Ssha)
    ssha_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(Ssha))
    
    if (nnan[0].shape[0]==0):
         return None, None, 0
    #hnan = np.where(np.isnan(Wind))
    
    lmarked_nan = np.where(np.isnan(ssha_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, Ssha.shape[1]),
                                np.arange(0, Ssha.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_ssha = Ssha[nnan]
    
    
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    # generate interpolated array of z-values
    ssha_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_ssha.ravel(),
                                     P, method='linear')   
    
    
    
    #data_it1 = np.copy(Windspeed)
    ssha_it1 = np.copy(Ssha)
    
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    ssha_it1[lmarked_nan] = ssha_interp_iter1
    
    
    
    # 2nd iteration
    ssha_markered_lm = np.copy(ssha_it1)
    ssha_markered_lm[landwhere] = -1
    
    
    nnan = np.where(~np.isnan(ssha_it1))
    #hnan = np.where(np.isnan(data_it1))
    
    lmarked_nan = np.where(np.isnan(ssha_markered_lm))
    #nonlmarked_nan = np.where(~np.isnan(wind_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, ssha_it1.shape[1]),
                                 np.arange(0, ssha_it1.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_ssha =ssha_it1[nnan]
   
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    #sst_interp = np.copy(data2)
    # generate interpolated array of z-values
    ssha_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_ssha.ravel(),
                                     P, method='nearest')  
    
     
    
    ssha_markered_lm[lmarked_nan] = ssha_interp_iter2
   
            
    # Expand to the grid:
    Vkoef = Lats_.shape[0] / ssha_markered_lm.shape[0]
    Hkoef = Lons_.shape[0] / ssha_markered_lm.shape[1]
    
    SSHA = interpolation.zoom(ssha_markered_lm, (Vkoef,Hkoef), order = 0)
    

    checknan = np.where(~np.isnan(SSHA))[0].shape[0]
    
       
    return SSHA,checknan     


# SSS level l2B brocessing

import h5py
import xarray as xa


def prepare_sss(sssfiles, lm, config_):
    latmax,latmin,lonmax,lonmin = quadrant(config_['bbox'])

    Lats_ = config_['lat']
    Lons_ = config_['lon']

    Lats_ = np.asarray(Lats_)
    Lons_ = np.asarray(Lons_)
        
    Lats_ = np.flip(Lats_)

    deltalat = 0.5
    deltalon = 0.5

    lat_points = int(np.ceil((latmax - latmin)/deltalat))
    lon_points = int(np.ceil((lonmax - lonmin)/deltalon))

    newlons_ = np.arange(lon_points)
    newlats_ = np.arange(lat_points)
    
    Lons = np.linspace(lonmin, lonmax, lon_points)
    Lats = np.linspace(latmin, latmax, lat_points)
  
    xx, yy= np.meshgrid(Lons, Lats) #config
    numlats = Lats.shape[0]
    numlons = Lons.shape[0]
    
    
    patch = np.zeros((len(sssfiles), numlats,numlons))
    patch[:,:,:] = np.nan

    shp = []
    
    
    for i in range (len(sssfiles)):
    
        fn = sssfiles[i]
       # key = smap_sss, lat, lon
        with h5py.File(fn, 'r') as h5:
            latdataset = h5['lat']
            #print(latdataset[0,0])
            alllats =  np.array(latdataset)
            
            londataset = h5.get('lon')
            alllons =  np.array(londataset)
            
            dataset = h5.get('smap_sss')
            alldata =  np.array(dataset)
                
        lats = np.copy(alllats)
        lats[lats<-500]=np.nan  
        data=np.copy(alldata)
        data[data<-500]=np.nan    
        lons = np.copy(alllons)
        lons[lons<-500]=np.nan  
                         
        #nlats = np.isnan(lats)
        #nlons = np.isnan(lons)
        ndata = np.isnan(data) 
        
        ndata_ind = np.where(~ndata)
        
        data0 = data[ndata_ind]
        lons0 = lons[ndata_ind]
        lats0 = lats[ndata_ind] 
                
        wlat = np.where((lats0 > latmin) & (lats0 < latmax) )         
        
        lats1 = lats0[wlat]
        lons1 = lons0[wlat]
        slice1_sss = data0[wlat] 
        
        wlon = np.where((lons1 > lonmin) & (lons1 < lonmax) )
                        
        lats2 = lats1[wlon]
        lons2 = lons1[wlon]
                        
        sss_ = slice1_sss[wlon]
        
        where_are_NaNs = np.isnan(sss_)
        sss = sss_[~where_are_NaNs]
        lats3 = lats2[~where_are_NaNs]
        lons3 = lons2[~where_are_NaNs]
                               
        if(sss.shape[0]>0):                                            
            idlons = ((lons3 - lonmin) // deltalon).astype(int)
            idlats = ((lats3 - latmin) // deltalat).astype(int)
                          
            patch[i,idlats,idlons] = sss            
    
    SSS = np.nanmean(patch, axis = 0)  
    nonnan = np.where(np.isnan(SSS))[0].shape[0] - 40*54
    #print('nnan', nonnan)
    checknan = np.where(~np.isnan(SSS))[0].shape[0] 
    
    SSS = np.flip(SSS, axis = 0)
    if (nonnan==0):
        #print('nnan', nonnan)
        return SSS, checknan

                    
    #---------- apply interpolation
        
    vkoef = SSS.shape[0]/lm.shape[0] 
    hkoef = SSS.shape[1]/lm.shape[1] 
        
    lmsqueezed = interpolation.zoom(lm, (vkoef,hkoef), order = 0)
        
    landwhere = np.where(lmsqueezed==0)
        
    sss_markered_lm = np.copy(SSS)
    sss_markered_lm[landwhere] = -1
        
    nnan = np.where(~np.isnan(SSS))
        
    #if (nnan[0].shape[0]==0):
    #    return None, None, 0
        #hnan = np.where(np.isnan(Wind))
        
    lmarked_nan = np.where(np.isnan(sss_markered_lm))
        
        # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, SSS.shape[1]),
                                     np.arange(0, SSS.shape[0]))
        
        # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_sss = SSS[nnan]
        
        
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
        
        # generate interpolated array of z-values
    sss_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_sss.ravel(),
                                         P, method='linear')   
    
        #data_it1 = np.copy(Windspeed)
    sss_it1 = np.copy(SSS)
        
        #print(np.max(sst_interp_data), np.max(sst_interp_data))
    sss_it1[lmarked_nan] = sss_interp_iter1
         
        # 2nd iteration
    sss_markered_lm = np.copy(sss_it1)
    sss_markered_lm[landwhere] = -1
    
        
    nnan = np.where(~np.isnan(sss_it1))
        #hnan = np.where(np.isnan(data_it1))
        
    lmarked_nan = np.where(np.isnan(sss_markered_lm))
        #nonlmarked_nan = np.where(~np.isnan(wind_markered_lm))
        
        # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, sss_it1.shape[1]),
                                     np.arange(0, sss_it1.shape[0]))
        
        # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_sss = sss_it1[nnan]
    
        
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
        
        #sst_interp = np.copy(data2)
        # generate interpolated array of z-values
    sss_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_sss.ravel(),
                                         P, method='nearest')  
    
        
    sss_markered_lm[lmarked_nan] = sss_interp_iter2
       
                
        # Expand to the grid:
    Vkoef = Lats_.shape[0] / sss_markered_lm.shape[0]
    Hkoef = Lons_.shape[0] / sss_markered_lm.shape[1]
        
    SSS_SMAP = interpolation.zoom(sss_markered_lm, (Vkoef,Hkoef), order = 0)
    
    
    
    checknan = np.where(~np.isnan(SSS_SMAP))[0].shape[0]
    
           
    return SSS_SMAP, checknan


def prepare_sss_mean(sssname, lm, config_):
    
    latmax,latmin,lonmax,lonmin = quadrant(config_['bbox'])

    Lats_ = config_['lat']
    Lons_ = config_['lon']

    Lats_ = np.asarray(Lats_)
    Lons_ = np.asarray(Lons_)
        
    Lats_ = np.flip(Lats_)

    deltalat = 0.5
    deltalon = 0.5

    lat_points = int(np.ceil((latmax - latmin)/deltalat))
    lon_points = int(np.ceil((lonmax - lonmin)/deltalon))

    newlons_ = np.arange(lon_points)
    newlats_ = np.arange(lat_points)
    
    Lons = np.linspace(lonmin, lonmax, lon_points)
    Lats = np.linspace(latmin, latmax, lat_points)
  
    xx, yy= np.meshgrid(Lons, Lats) #config
    numlats = Lats.shape[0]
    numlons = Lons.shape[0]
   
    try:
        with xa.open_dataset(sssname ) as data_1:
            #nnan = np.where(~np.isnan(data_1['smap_sss']))
    
            #print('nnan',nnan[0].shape)
            
            sslice = data_1.sel(latitude=slice(latmax,latmin), longitude=slice(lonmin,lonmax))
            ssslats = sslice['latitude'].values
            ssslons = sslice['longitude'].values
            rawsss =  sslice['smap_sss'].values
    except ValueError: 
        return None, 0
  
        
    vkoef = rawsss.shape[0]/lm.shape[0] 
    hkoef = rawsss.shape[0]/lm.shape[0] 
    
    lmsqueezed = interpolation.zoom(lm, (vkoef,hkoef), order = 0)
        
    landwhere = np.where(lmsqueezed==0)
    
    sss_markered_lm = np.copy(rawsss)
    sss_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(rawsss))
    hnan = np.where(np.isnan(rawsss))
    
    
    if (nnan[0].shape[0]==0):
         return None, 0
    
    lmarked_nan = np.where(np.isnan(sss_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, rawsss.shape[1]),
                                 np.arange(0, rawsss.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_sss = rawsss[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    #sst_interp = np.copy(data2)
    # generate interpolated array of z-values
    sss_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_sss.ravel(),
                                     P, method='linear')  
    
    #print(sst_interp_data.shape, P.shape, (nnan[0].shape), data2.shape)
    data_it1 = np.copy(rawsss)
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    data_it1[lmarked_nan] = sss_interp_iter1
    #plt.imshow(data_it1)
    #plt.show()
    #print(aa)
    
    # fill with nearest:
    sss_markered_lm = np.copy(data_it1)
    sss_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(data_it1))
    #hnan = np.where(np.isnan(data_it1))
    
    lmarked_nan = np.where(np.isnan(sss_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, data_it1.shape[1]),
                                 np.arange(0, data_it1.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_sss = data_it1[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    #sst_interp = np.copy(data2)
    # generate interpolated array of z-values
    sss_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_sss.ravel(),
                                     P, method='nearest')  
    sss_markered_lm[lmarked_nan] = sss_interp_iter2
           
    Vkoef = Lats_.shape[0] / sss_markered_lm.shape[0]
    Hkoef = Lons_.shape[0] / sss_markered_lm.shape[1]
    
    SSS = interpolation.zoom(sss_markered_lm, (Vkoef,Hkoef), order = 0)
        
    checknan = np.where(~np.isnan(SSS))[0].shape[0]
    #print('chacknan', checknan)
       
    return SSS, checknan  


def prepare_sss_splitted(sssfiles, LM, config_):
    # collection = 'C2208418228-POCLOUD'
    
    
    latmax,latmin,lonmax_,lonmin_ = quadrant(config_['bbox'])

    Lats_ = config_['lat']
    Lons0_ = config_['lon']

    Lats_ = np.asarray(Lats_)
    Lons0_ = np.asarray(Lons0_)
    
    ind0 = np.where(Lons0_<0)[0][0]
        
    #print(Lons0_[ind0], Lons0_[ind0-1])
    #print(Lons0_.shape, LM.shape)
    
    
    lm1 = LM[:,:ind0]
    lm2 = LM[:,ind0:]
    lm_ = [lm1,lm2]
    
    LONS_ = [Lons0_[:ind0], Lons0_[ind0:]] 
    
    lonmin1= lonmin_
    lonmax1 = 180
    lonmin2 = -180
    lonmax2 = lonmax_
    
    Lonrange = [[lonmin1, lonmax1],[lonmin2, lonmax2]]

    deltalat = 0.5
    deltalon = 0.5
    
    SSS_SMAP = []
    
    Lats_ = np.flip(Lats_)
    checknan = 1
    
    # split data
    
    for i in range(2):
        
        lm = lm_[i]
        Lons_ = LONS_[i]
        lonmin = Lonrange [i][0]
        lonmax = Lonrange [i][1]
        
    
        lat_points = int(np.ceil((latmax - latmin)/deltalat))
        lon_points = int(np.ceil((lonmax - lonmin)/deltalon))
    
        newlons_ = np.arange(lon_points)
        newlats_ = np.arange(lat_points)
        
        Lons = np.linspace(lonmin, lonmax, lon_points)
        Lats = np.linspace(latmin, latmax, lat_points)
      
        xx, yy= np.meshgrid(Lons, Lats) #config
        numlats = Lats.shape[0]
        numlons = Lons.shape[0]
        
        
        patch = np.zeros((len(sssfiles), numlats,numlons))
        patch[:,:,:] = np.nan
    
        shp = []
        
        
        for i in range (len(sssfiles)):
        
            fn = sssfiles[i]
            
           # key = smap_sss, lat, lon
            with h5py.File(fn, 'r') as h5:
                latdataset = h5['lat']
                #print(latdataset[0,0])
                alllats =  np.array(latdataset)
                
                londataset = h5.get('lon')
                alllons =  np.array(londataset)
                
                dataset = h5.get('smap_sss')
                alldata =  np.array(dataset)
                    
            lats = np.copy(alllats)
            lats[lats<-500]=np.nan  
            data=np.copy(alldata)
            data[data<-500]=np.nan    
            lons = np.copy(alllons)
            lons[lons<-500]=np.nan  
                             
            #nlats = np.isnan(lats)
            #nlons = np.isnan(lons)
            ndata = np.isnan(data) 
            
            ndata_ind = np.where(~ndata)
            
            data0 = data[ndata_ind]
            lons0 = lons[ndata_ind]
            lats0 = lats[ndata_ind] 
                    
            wlat = np.where((lats0 > latmin) & (lats0 < latmax) )         
            
            lats1 = lats0[wlat]
            lons1 = lons0[wlat]
            slice1_sss = data0[wlat] 
            
            wlon = np.where((lons1 > lonmin) & (lons1 < lonmax) )
                            
            lats2 = lats1[wlon]
            lons2 = lons1[wlon]
                            
            sss_ = slice1_sss[wlon]
            
            where_are_NaNs = np.isnan(sss_)
            sss = sss_[~where_are_NaNs]
            lats3 = lats2[~where_are_NaNs]
            lons3 = lons2[~where_are_NaNs]
                                   
            if(sss.shape[0]>0):                                            
                idlons = ((lons3 - lonmin) // deltalon).astype(int)
                idlats = ((lats3 - latmin) // deltalat).astype(int)
                              
                patch[i,idlats,idlons] = sss            
        
        SSS = np.nanmean(patch, axis = 0)  
        SSS = np.flip(SSS, axis = 0)
        nonnan = np.where(np.isnan(SSS))[0].shape[0] - 40*54
        #print('nnan', nonnan)
        checknan = np.where(~np.isnan(SSS))[0].shape[0]
        if (nonnan==0):
            #print('nnan', nonnan)
            return SSS, checknan
    
                        
        #---------- apply interpolation
            
        vkoef = SSS.shape[0]/lm.shape[0] 
        hkoef = SSS.shape[0]/lm.shape[0] 
            
        lmsqueezed = interpolation.zoom(lm, (vkoef,hkoef), order = 0)
            
        landwhere = np.where(lmsqueezed==0)
            
        sss_markered_lm = np.copy(SSS)
        sss_markered_lm[landwhere] = -1
            
        nnan = np.where(~np.isnan(SSS))
            
        #if (nnan[0].shape[0]==0):
        #    return None, None, 0
            #hnan = np.where(np.isnan(Wind))
            
        lmarked_nan = np.where(np.isnan(sss_markered_lm))
            
            # integer arrays for indexing
        x_indx, y_indx = np.meshgrid(np.arange(0, SSS.shape[1]),
                                         np.arange(0, SSS.shape[0]))
            
            # retrieve the valid, non-Nan, defined values
        valid_xs = x_indx[nnan]
        valid_ys = y_indx[nnan]
        valid_sss = SSS[nnan]
        
            
        if (nonnan==0):
            #print('nnan', nonnan)
            return SSS, checknan
            
            
        XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
        YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
        P = np.hstack((XX,YY))
            
            # generate interpolated array of z-values
        sss_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_sss.ravel(),
                                             P, method='linear')   
        
            #data_it1 = np.copy(Windspeed)
        sss_it1 = np.copy(SSS)
            
            #print(np.max(sst_interp_data), np.max(sst_interp_data))
        sss_it1[lmarked_nan] = sss_interp_iter1
             
            # 2nd iteration
        sss_markered_lm = np.copy(sss_it1)
        sss_markered_lm[landwhere] = -1
        
            
        nnan = np.where(~np.isnan(sss_it1))
            #hnan = np.where(np.isnan(data_it1))
            
        lmarked_nan = np.where(np.isnan(sss_markered_lm))
            #nonlmarked_nan = np.where(~np.isnan(wind_markered_lm))
            
            # integer arrays for indexing
        x_indx, y_indx = np.meshgrid(np.arange(0, sss_it1.shape[1]),
                                         np.arange(0, sss_it1.shape[0]))
            
            # retrieve the valid, non-Nan, defined values
        valid_xs = x_indx[nnan]
        valid_ys = y_indx[nnan]
        valid_sss = sss_it1[nnan]
        
            
        XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
        YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
        P = np.hstack((XX,YY))
            
            #sst_interp = np.copy(data2)
            # generate interpolated array of z-values
        sss_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_sss.ravel(),
                                             P, method='nearest')  
        
            
        sss_markered_lm[lmarked_nan] = sss_interp_iter2
           
                    
            # Expand to the grid:
        Vkoef = Lats_.shape[0] / sss_markered_lm.shape[0]
        Hkoef = Lons_.shape[0] / sss_markered_lm.shape[1]
            
        SSS_smap = interpolation.zoom(sss_markered_lm, (Vkoef,Hkoef), order = 0)
        SSS_SMAP.append(SSS_smap)
        
        
        checknan_ = np.where(~np.isnan(SSS_smap))[0].shape[0] 
        checknan = checknan*checknan_
        
                
        #plt.imshow(SSS_smap)
        #plt.show()
    
    SSSSMAP = np.hstack((SSS_SMAP[0], SSS_SMAP[1]))
           
    return SSSSMAP, checknan


def prepare_sss_mean_splitted(sssname, lm, config_):
    # collid =  'C2208425700-POCLOUD'
    
    latmax,latmin,lonmax,lonmin = quadrant(config_['bbox'])

    Lats_ = config_['lat']
    Lons_ = config_['lon']

    Lats_ = np.asarray(Lats_)
    Lons_ = np.asarray(Lons_)
        
    Lats_ = np.flip(Lats_)

    deltalat = 0.5
    deltalon = 0.5
    lonmin = lonmin%360
    lonmax = lonmax%360
   

   
    try:
        with xa.open_dataset(sssname ) as data_1:
                        
            sslice = data_1.sel(lat=slice(latmin,latmax), lon=slice(lonmin,lonmax))
            ssslats = sslice['lat'].values
            ssslons = sslice['lon'].values
            rawsss =  sslice['sss_smap'].values
    except ValueError: 
        return None, 0
  
    rawsss = np.flip(rawsss,axis = 0) 
    
    vkoef = rawsss.shape[0]/lm.shape[0] 
    hkoef = rawsss.shape[0]/lm.shape[0] 
    
    lmsqueezed = interpolation.zoom(lm, (vkoef,hkoef), order = 0)
        
    landwhere = np.where(lmsqueezed==0)
    
    sss_markered_lm = np.copy(rawsss)
    sss_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(rawsss))
    hnan = np.where(np.isnan(rawsss))
    
    
    if (nnan[0].shape[0]==0):
         return None, 0
    
    lmarked_nan = np.where(np.isnan(sss_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, rawsss.shape[1]),
                                 np.arange(0, rawsss.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_sss = rawsss[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    #sst_interp = np.copy(data2)
    # generate interpolated array of z-values
    sss_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_sss.ravel(),
                                     P, method='linear')  
    
    #print(sst_interp_data.shape, P.shape, (nnan[0].shape), data2.shape)
    data_it1 = np.copy(rawsss)
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    data_it1[lmarked_nan] = sss_interp_iter1
    #plt.imshow(data_it1)
    #plt.show()
    #print(aa)
    
    # fill with nearest:
    sss_markered_lm = np.copy(data_it1)
    sss_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(data_it1))
    #hnan = np.where(np.isnan(data_it1))
    
    lmarked_nan = np.where(np.isnan(sss_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, data_it1.shape[1]),
                                 np.arange(0, data_it1.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_sss = data_it1[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    #sst_interp = np.copy(data2)
    # generate interpolated array of z-values
    sss_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_sss.ravel(),
                                     P, method='nearest')  
    sss_markered_lm[lmarked_nan] = sss_interp_iter2
           
    Vkoef = Lats_.shape[0] / sss_markered_lm.shape[0]
    Hkoef = Lons_.shape[0] / sss_markered_lm.shape[1]
    
    SSS = interpolation.zoom(sss_markered_lm, (Vkoef,Hkoef), order = 0)
        
    checknan = np.where(~np.isnan(SSS))[0].shape[0]
    #print('chacknan', checknan)
       
    return SSS, checknan  
          
    

def prepare_sst_west(sstfile, lm, config_):
    latmax,latmin,lonmax,lonmin = quadrant(config_['bbox'])

    Lats_ = config_['lat']
    Lons_ = config_['lon']

    Lats_ = np.asarray(Lats_)
    Lons_ = np.asarray(Lons_)
            
    try:
        with xa.open_dataset(sstfile) as sst0:
            #print(sst0)
            sslice = sst0.sel(lat=slice(latmin,latmax), lon=slice(lonmin,lonmax))
            lats0 = sslice['lat'].values
            lons0 = sslice['lon'].values
            data0 = sslice['sea_surface_temperature'].values[0,:,:]
    except ValueError: 
        
        return None, 0
    
    
    data0 = np.flip(data0,axis = 0) 
    print(data0)
    data0 = data0 - 273.1 # to Centrigrade 
        
    #prepare landmask
    vkoef = data0.shape[0]/lm.shape[0] 
    hkoef =  data0.shape[0]/lm.shape[0] 
    
    lmsqueezed = interpolation.zoom(lm, (vkoef,hkoef), order = 0)
    
    landwhere = np.where(lmsqueezed==0)
    
    sst_markered_lm = np.copy(data0)
    sst_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(data0))
    hnan = np.where(np.isnan(data0))
    
    if (nnan[0].shape[0]==0):
         return None, 0
        
    
    lmarked_nan = np.where(np.isnan(sst_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, data0.shape[1]),
                                 np.arange(0, data0.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_sst = data0[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    sst_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_sst.ravel(),
                                     P, method='linear')  
    
    #print(sst_interp_data.shape, P.shape, (nnan[0].shape), data2.shape)
    data_it1 = np.copy(data0)
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    data_it1[lmarked_nan] = sst_interp_iter1
    
       
    # fill with nearest:
    sst_markered_lm = np.copy(data_it1)
    sst_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(data_it1))
    #hnan = np.where(np.isnan(data_it1))
    
    lmarked_nan = np.where(np.isnan(sst_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, data_it1.shape[1]),
                                 np.arange(0, data_it1.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_sst = data_it1[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    #sst_interp = np.copy(data2)
    # generate interpolated array of z-values
    sst_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_sst.ravel(),
                                     P, method='nearest')  
    
    #print(sst_interp_data.shape, P.shape, (nnan[0].shape), data2.shape)
    data_it2 = np.copy(data_it1)
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    data_it2[lmarked_nan] = sst_interp_iter2    
    sst_markered_lm[lmarked_nan] = sst_interp_iter2
            
    Vkoef = Lats_.shape[0] / sst_markered_lm.shape[0]
    Hkoef = Lons_.shape[0] / sst_markered_lm.shape[1]
    
    SST = interpolation.zoom(sst_markered_lm, (Vkoef,Hkoef), order = 0)
    
    checknan = np.where(~np.isnan(SST))[0].shape[0]
    print('sstnan', checknan)
       
    return SST, checknan




def prepare_sst_east(sstfile, lm, config_):
    latmax,latmin,lonmax,lonmin = quadrant(config_['bbox'])

    Lats_ = config_['lat']
    Lons_ = config_['lon']

    Lats_ = np.asarray(Lats_)
    Lons_ = np.asarray(Lons_)
    
    lonmin1= lonmin
    lonmax1 = 180
    
    lonmin2 = -180
    lonmax2 = lonmax
            
    try:
        with xa.open_dataset(sstfile) as sst0:
            #print(data2_all)
            sslice = sst0.sel(lat=slice(latmin,latmax), lon=slice(lonmin1,lonmax1))
            lats0 = sslice['lat'].values
            lons0 = sslice['lon'].values
            data01 = sslice['sea_surface_temperature'].values[0,:,:]
    except ValueError: 
        
        return None, 0

            
    try:
        with xa.open_dataset(sstfile) as sst0:
            #print(data2_all)
            sslice = sst0.sel(lat=slice(latmin,latmax), lon=slice(lonmin2,lonmax2))
            lats0 = sslice['lat'].values
            lons0 = sslice['lon'].values
            data02 = sslice['sea_surface_temperature'].values[0,:,:]
    except ValueError: 
        
        return None, 0
    
    #print(lats0)
    data0 = np.hstack((data01, data02))
    data0 = np.flip(data0,axis = 0) 
    data0 = data0 - 273.1 # to Centrigrade 
        
    #prepare landmask
    vkoef = data0.shape[0]/lm.shape[0] 
    hkoef =  data0.shape[0]/lm.shape[0] 
    
    lmsqueezed = interpolation.zoom(lm, (vkoef,hkoef), order = 0)
    
    landwhere = np.where(lmsqueezed==0)
    
    sst_markered_lm = np.copy(data0)
    sst_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(data0))
    hnan = np.where(np.isnan(data0))
    
    if (nnan[0].shape[0]==0):
         return None, 0
        
    
    lmarked_nan = np.where(np.isnan(sst_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, data0.shape[1]),
                                 np.arange(0, data0.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_sst = data0[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    sst_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_sst.ravel(),
                                     P, method='linear')  
    
    #print(sst_interp_data.shape, P.shape, (nnan[0].shape), data2.shape)
    data_it1 = np.copy(data0)
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    data_it1[lmarked_nan] = sst_interp_iter1
    
       
    # fill with nearest:
    sst_markered_lm = np.copy(data_it1)
    sst_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(data_it1))
    #hnan = np.where(np.isnan(data_it1))
    
    lmarked_nan = np.where(np.isnan(sst_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, data_it1.shape[1]),
                                 np.arange(0, data_it1.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_sst = data_it1[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    #sst_interp = np.copy(data2)
    # generate interpolated array of z-values
    sst_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_sst.ravel(),
                                     P, method='nearest')  
    
    #print(sst_interp_data.shape, P.shape, (nnan[0].shape), data2.shape)
    data_it2 = np.copy(data_it1)
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    data_it2[lmarked_nan] = sst_interp_iter2    
    sst_markered_lm[lmarked_nan] = sst_interp_iter2
            
    Vkoef = Lats_.shape[0] / sst_markered_lm.shape[0]
    Hkoef = Lons_.shape[0] / sst_markered_lm.shape[1]
    
    SST = interpolation.zoom(sst_markered_lm, (Vkoef,Hkoef), order = 0)
    
    checknan = np.where(~np.isnan(SST))[0].shape[0]
    #print('sstnan', checknan)
       
    return SST, checknan


#-------------------------------------------------------------------------
def prepare_wind_east(windfiles, lm, config_):
    # collection = 'C2075141559-POCLOUD'
    
    latmax,latmin,lonmax_,lonmin_ = quadrant(config_['bbox']) 
   
             
    Lats_ = config_['lat']
    Lons_ = config_['lon'] 
    
    
    
    Lats_ = np.asarray(Lats_)
    Lons_ = np.asarray(Lons_) 
    
    lonmin = (lonmin_)%360
    lonmax = (lonmax_)%360

    deltalat = 0.5
    deltalon = 0.5
    
    lat_points = int(np.ceil((latmax - latmin)/deltalat))
    lon_points = int(np.ceil((lonmax - lonmin)/deltalon))
    
    #newlons_ = np.arange(lon_points)
    #newlats_ = np.arange(lat_points)
    
    Lons = np.linspace(lonmin, lonmax, lon_points)
    Lats = np.linspace(latmin, latmax, lat_points)
        
    xx, yy= np.meshgrid(Lons, Lats) #config
                
        # 0- speed
        # 1- direction
     
    numlats = Lats.shape[0]
    numlons = Lons.shape[0]
    
       
    patch_wspeed = np.zeros((len(windfiles), numlats,numlons))
    patch_wspeed[:,:,:] = np.nan
    patch_wdir = np.zeros((len(windfiles), numlats,numlons))
    patch_wdir[:,:,:] = np.nan
    
    WIND_SPEED = []
    WIND_DIR = []
       
    
    checknan = 1  
    
    
            
    for i in range(len(windfiles)):
        #os.path.join('winds_tests', 'wind_test2')
      
        wind0 = windfiles[i]
        
        with xa.open_dataset(wind0) as data_0:
            #print(data_0)
            #print(aa)
            # Part 1
                
            lats = data_0['lat'].values
            lons = data_0['lon'].values
                        
            wind_speed_all = data_0['wind_speed'].values
            wind_dir_all = data_0['wind_dir'].values
                #
        wlat = np.where((lats > latmin) & (lats < latmax) ) 
                                
        lats1 = lats[wlat]
        lons1 = lons[wlat]
        slice1_wspeed = wind_speed_all[wlat]
        slice1_wdir = wind_dir_all[wlat]
                
               
        wlon = np.where((lons1 > lonmin) & (lons1 < lonmax) )
        #wlon = np.where((lons1 > lonmin) )
        lats2 = lats1[wlon]
        lons2 = lons1[wlon]
        
        wwind_speed = slice1_wspeed[wlon]
        wwind_dir = slice1_wdir[wlon]
                
        where_are_NaNs = np.isnan(wwind_speed)
        wind_speed = wwind_speed[~where_are_NaNs]
        wind_dir = wwind_dir[~where_are_NaNs]
        lats3 = lats2[~where_are_NaNs]
        lons3 = lons2[~where_are_NaNs]
        #print(i, 'nans')
        #print(where_are_NaNs.shape)    
                
        if(wind_speed.shape[0]>0):
                                            
            idlons = ((lons3 - lonmin) // deltalon).astype(int)
            idlats = ((lats3 - latmin) // deltalat).astype(int)
                                
                    #Wind[idlats,idlons] = wind
            patch_wspeed[i,idlats,idlons] = wind_speed
            patch_wdir[i,idlats,idlons] = wind_dir
               
    #Windspeed_ = np.nanmean(patch_wspeed, axis = 0) 
   
    Windspeed = np.nanmean(patch_wspeed, axis = 0)
    Winddir = np.nanmean(patch_wdir, axis = 0)
        
    Windspeed = np.flip(Windspeed, axis = 0)
    Winddir = np.flip(Winddir, axis = 0)
    
            
    #---------- apply interpolation
    
    vkoef = Windspeed.shape[0]/lm.shape[0] 
    hkoef = Windspeed.shape[0]/lm.shape[0] 
    
    lmsqueezed = interpolation.zoom(lm, (vkoef,hkoef), order = 0)
    
    landwhere = np.where(lmsqueezed==0)
    
    winds_markered_lm = np.copy(Windspeed)
    winds_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(Windspeed))
    
    if (nnan[0].shape[0]==0):
         return None, None, 0
    #hnan = np.where(np.isnan(Wind))
    
    lmarked_nan = np.where(np.isnan(winds_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, Windspeed.shape[1]),
                                 np.arange(0, Windspeed.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_windspeed = Windspeed[nnan]
    valid_winddir = Winddir[nnan]
    
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    # generate interpolated array of z-values
    windspeed_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_windspeed.ravel(),
                                     P, method='linear')   
    
    winddir_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_winddir.ravel(),
                                     P, method='linear')   
    
    #data_it1 = np.copy(Windspeed)
    windspeed_it1 = np.copy(Windspeed)
    winddir_it1 = np.copy(Winddir)
    #print(np.max(sst_interp_data), np.max(sst_interp_data))
    windspeed_it1[lmarked_nan] = windspeed_interp_iter1
    winddir_it1[lmarked_nan] = winddir_interp_iter1
    
    
    # 2nd iteration
    windspeed_markered_lm = np.copy(windspeed_it1)
    windspeed_markered_lm[landwhere] = -1
    
    winddir_markered_lm = np.copy(winddir_it1)
    winddir_markered_lm[landwhere] = -1
    
    nnan = np.where(~np.isnan(windspeed_it1))
    #hnan = np.where(np.isnan(data_it1))
    
    lmarked_nan = np.where(np.isnan(windspeed_markered_lm))
    #nonlmarked_nan = np.where(~np.isnan(wind_markered_lm))
    
    # integer arrays for indexing
    x_indx, y_indx = np.meshgrid(np.arange(0, windspeed_it1.shape[1]),
                                 np.arange(0, windspeed_it1.shape[0]))
    
    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[nnan]
    valid_ys = y_indx[nnan]
    valid_windspeed = windspeed_it1[nnan]
    valid_winddir = winddir_it1[nnan]
    
    XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
    YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
    P = np.hstack((XX,YY))
    
    #sst_interp = np.copy(data2)
    # generate interpolated array of z-values
    windspeed_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_windspeed.ravel(),
                                     P, method='nearest')  
    
    winddir_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_winddir.ravel(),
                                     P, method='nearest')  
    
    windspeed_markered_lm[lmarked_nan] = windspeed_interp_iter2
    winddir_markered_lm[lmarked_nan] = winddir_interp_iter2
            
    # Expand to the grid:
    Vkoef = Lats_.shape[0] / windspeed_markered_lm.shape[0]
    Hkoef = Lons_.shape[0] / windspeed_markered_lm.shape[1]
    
    WINDSPEED = interpolation.zoom(windspeed_markered_lm, (Vkoef,Hkoef), order = 0)
    WINDDIR = interpolation.zoom(winddir_markered_lm, (Vkoef,Hkoef), order = 0)
    

    checknan = np.where(~np.isnan(WINDSPEED))[0].shape[0]
    
       
    return WINDSPEED,WINDDIR ,checknan

#------------------------------------------------------
#-------------------------------------------------------------------------
def prepare_wind_west(windfiles, LM, config_):
    # collection = 'C2075141559-POCLOUD'
    
    latmax,latmin,lonmax_,lonmin_ = quadrant(config_['bbox']) 
             
    Lats_ = config_['lat']
    Lons_ = config_['lon'] 
    
    
    Lons0_ = config_['lon']

    Lats_ = np.asarray(Lats_)
    Lons0_ = np.asarray(Lons0_)
    
    ind0 = np.where(Lons0_>0)[0][0]
        
    
    
    lm1 = LM[:,:ind0]
    lm2 = LM[:,ind0:]
    lm_ = [lm1,lm2]
    
    LONS_ = [Lons0_[:ind0], Lons0_[ind0:]] 
    lonmin_ = (lonmin_)%360
    lonmax_ = (lonmax_)%360
    
   
    
    #(360 - 0.00001)
    Lonrange = [[lonmin_,360 - 0.00001],[0, lonmax_] ]

    deltalat = 0.5
    deltalon = 0.5
    
    WIND_SPEED = []
    WIND_DIR = []
    
    Lats_ = np.asarray(Lats_)
    Lons_ = np.asarray(Lons_) 
    
    checknan = 1  
    
    for l in range(2):
        
        lm = lm_[l]
        Lons_ = LONS_[l]
        lonmin = Lonrange [l][0]
        lonmax = Lonrange [l][1]
         
        deltalat = 0.5
        deltalon = 0.5
        
     
        
        #+++++++++++++
        
        lat_points = int(np.ceil((latmax - latmin)/deltalat))
        lon_points = int(np.ceil((lonmax - lonmin)/deltalon))
        
        #newlons_ = np.arange(lon_points)
        #newlats_ = np.arange(lat_points)
        
        Lons = np.linspace(lonmin, lonmax, lon_points)
        Lats = np.linspace(latmin, latmax, lat_points)
            
        xx, yy= np.meshgrid(Lons, Lats) #config
                    
            # 0- speed
            # 1- direction
         
        numlats = Lats.shape[0]
        numlons = Lons.shape[0]
        
               
        patch_wspeed = np.zeros((len(windfiles), numlats,numlons))
        patch_wspeed[:,:,:] = np.nan
        patch_wdir = np.zeros((len(windfiles), numlats,numlons))
        patch_wdir[:,:,:] = np.nan
        
       
        
        # reproject lons:
                  
        for i in range(len(windfiles)):
            #os.path.join('winds_tests', 'wind_test2')
          
            
            wind0 = windfiles[i]
            
            with xa.open_dataset(wind0) as data_0:
                #print(data_0)
                #print(aa)
                # Part 1
                    
                lats = data_0['lat'].values
                lons = data_0['lon'].values 
                
                            
                wind_speed_all = data_0['wind_speed'].values
                wind_dir_all = data_0['wind_dir'].values
                   
            wlat = np.where((lats > latmin) & (lats < latmax) ) 
                                    
            lats1 = lats[wlat]
            lons1 = lons[wlat]
            
            slice1_wspeed = wind_speed_all[wlat]
            slice1_wdir = wind_dir_all[wlat]
                    
                   
            wlon = np.where((lons1 > lonmin) & (lons1 < lonmax) )
            #wlon = np.where((lons1 > lonmin) )
            lats2 = lats1[wlon]
            lons2 = lons1[wlon]
            
            wwind_speed = slice1_wspeed[wlon]
            wwind_dir = slice1_wdir[wlon]
                    
            where_are_NaNs = np.isnan(wwind_speed)
            wind_speed = wwind_speed[~where_are_NaNs]
            wind_dir = wwind_dir[~where_are_NaNs]
            lats3 = lats2[~where_are_NaNs]
            lons3 = lons2[~where_are_NaNs]
            #print(i, 'nans')
            #print(where_are_NaNs.shape)    
                    
            if(wind_speed.shape[0]>0):
                                                
                idlons = ((lons3 - lonmin) // deltalon).astype(int)
                idlats = ((lats3 - latmin) // deltalat).astype(int)
                                    
                        #Wind[idlats,idlons] = wind
                patch_wspeed[i,idlats,idlons] = wind_speed
                patch_wdir[i,idlats,idlons] = wind_dir
                   
        #Windspeed_ = np.nanmean(patch_wspeed, axis = 0) 
            
        Windspeed = np.nanmean(patch_wspeed, axis = 0)
        Winddir = np.nanmean(patch_wdir, axis = 0)
            
        Windspeed = np.flip(Windspeed, axis = 0)
        Winddir = np.flip(Winddir, axis = 0)
                
        #---------- apply interpolation
        
        vkoef = Windspeed.shape[0]/lm.shape[0] 
        hkoef = Windspeed.shape[0]/lm.shape[0] 
        
        lmsqueezed = interpolation.zoom(lm, (vkoef,hkoef), order = 0)
        
        landwhere = np.where(lmsqueezed==0)
        
        winds_markered_lm = np.copy(Windspeed)
        winds_markered_lm[landwhere] = -1
        
        nnan = np.where(~np.isnan(Windspeed))
        
        if (nnan[0].shape[0]==0):
             return None, None, 0
        #hnan = np.where(np.isnan(Wind))
        
        lmarked_nan = np.where(np.isnan(winds_markered_lm))
        
        # integer arrays for indexing
        x_indx, y_indx = np.meshgrid(np.arange(0, Windspeed.shape[1]),
                                     np.arange(0, Windspeed.shape[0]))
        
        # retrieve the valid, non-Nan, defined values
        valid_xs = x_indx[nnan]
        valid_ys = y_indx[nnan]
        valid_windspeed = Windspeed[nnan]
        valid_winddir = Winddir[nnan]
        
        
        XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
        YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
        P = np.hstack((XX,YY))
        
        # generate interpolated array of z-values
        windspeed_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_windspeed.ravel(),
                                         P, method='linear')   
        
        winddir_interp_iter1 = interpolate.griddata((valid_xs, valid_ys), valid_winddir.ravel(),
                                         P, method='linear')   
        
        #data_it1 = np.copy(Windspeed)
        windspeed_it1 = np.copy(Windspeed)
        winddir_it1 = np.copy(Winddir)
        #print(np.max(sst_interp_data), np.max(sst_interp_data))
        windspeed_it1[lmarked_nan] = windspeed_interp_iter1
        winddir_it1[lmarked_nan] = winddir_interp_iter1
        
        
        # 2nd iteration
        windspeed_markered_lm = np.copy(windspeed_it1)
        windspeed_markered_lm[landwhere] = -1
        
        winddir_markered_lm = np.copy(winddir_it1)
        winddir_markered_lm[landwhere] = -1
        
        nnan = np.where(~np.isnan(windspeed_it1))
        #hnan = np.where(np.isnan(data_it1))
        
        lmarked_nan = np.where(np.isnan(windspeed_markered_lm))
        #nonlmarked_nan = np.where(~np.isnan(wind_markered_lm))
        
        # integer arrays for indexing
        x_indx, y_indx = np.meshgrid(np.arange(0, windspeed_it1.shape[1]),
                                     np.arange(0, windspeed_it1.shape[0]))
        
        # retrieve the valid, non-Nan, defined values
        valid_xs = x_indx[nnan]
        valid_ys = y_indx[nnan]
        valid_windspeed = windspeed_it1[nnan]
        valid_winddir = winddir_it1[nnan]
        
        XX = np.expand_dims(x_indx[lmarked_nan],axis=1)
        YY = np.expand_dims(y_indx[lmarked_nan],axis=1)
        P = np.hstack((XX,YY))
        
        #sst_interp = np.copy(data2)
        # generate interpolated array of z-values
        windspeed_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_windspeed.ravel(),
                                         P, method='nearest')  
        
        winddir_interp_iter2 = interpolate.griddata((valid_xs, valid_ys), valid_winddir.ravel(),
                                         P, method='nearest')  
        
        windspeed_markered_lm[lmarked_nan] = windspeed_interp_iter2
        winddir_markered_lm[lmarked_nan] = winddir_interp_iter2
                
        # Expand to the grid:
        Vkoef = Lats_.shape[0] / windspeed_markered_lm.shape[0]
        Hkoef = Lons_.shape[0] / windspeed_markered_lm.shape[1]
        
        WINDSPEED_ = interpolation.zoom(windspeed_markered_lm, (Vkoef,Hkoef), order = 0)
        WINDDIR_ = interpolation.zoom(winddir_markered_lm, (Vkoef,Hkoef), order = 0)
        
    
        checknan = checknan*(np.where(~np.isnan(WINDSPEED_))[0].shape[0])
        WIND_SPEED.append(WINDSPEED_)
        WIND_DIR.append(WINDDIR_)
    
    WINDSPEED = np.hstack((WIND_SPEED[0], WIND_SPEED[1]))  
    WINDDIR = np.hstack((WIND_DIR[0], WIND_DIR[1]))  
    return WINDSPEED,WINDDIR ,checknan

