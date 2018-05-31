#!/usr/bin/python

import glob
import fabio
import json
import argparse
import pylab as plt
import sys
import os
import numpy as np
import glob
import h5py
from joblib import Parallel, delayed
import time

from streakaboo import find_peaks4

# class for smart loading many CXI files

#############
#plot=args.plot

######################

class Logger(object):
    def __init__(self, log_f):
        self.terminal = sys.stdout
        self.log = open(log_f, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass    


def main(fname, inds, JID, data_path):
    R_name = "pk_R_lcp.npy" # radius of each pixel (dist from cent) 
    MASK_name = "pk_mask_lcp.npy" #bool mask (bad=False, good=True),same shape as img 
    cent_name = "pk_center_lcp.npy" # this 
    MEDIAN_name = "pk_median_lcp.npy" # can set to =None is you dont have

    R = np.load(R_name) # # can set = None if you dont have, just set run_rad_med = False...
    MASK = np.load(MASK_name) #can set to np.ones( (SS_dim, FS_dim), dtype=np.bool) ) if dont have
    cent = np.load(cent_name) # center, can set cent= (fs_coor, ss_coor) 
    cent = (640.,640.)
    MEDIAN_IMG = np.load(MEDIAN_name) # can set to None if you dont have, just set median_sub=False


    #find_peaks4.plot_pks(d[3]*mask, make_sparse=True, nsigs=3.5, sig_G=.8,sig_G2=0.1,  thresh=1, sz=8,dist_cut=2, filt=True,filt2=False, min_dist=6, r_in=None, cent=[a,b], ret_subs=False,  R=R, rbins=linspace(R.min(), R.max(), 100), run_rad_med=True, peak_COM=True, min_snr=6.5, median_img=mask*Imed, median_sub=False,)

    PK_PAR =  { 'make_sparse':True,# leave true.. 
        'sig_G':1.1,'sig_G2':0.1,  # gaussian filter sigma (applied before local max filter, and then applied to streak detect)
        'thresh':15,  # local max with this value pixel (ADU) will be ignored
        'sz':8,  # sub image size used to  do local analysis
        'min_snr': 2.25,  # min snr, rough compute, optimized visually for each experiment, use this when 'filt' is True
        'filt': True, # this is my janky filtering I prob screwed it up in last commit, email me if bad... 
        'filt2':False, # this fits streaks to peaks, but it is slow. its hero mode.. 
        'min_dist':10, # min dist between pixels 
        'dist_cut':2, # dist cut is only for hero mode, verifies the streak is close to the peaks
        'r_in':None, # min res ring (only detect peaks within this radius
        'cent':cent,  # (fast scan, slow scan) center of pilatus
        'R':R, # provides radius of each pixel, pre-computed... same shape as pilatus
        'rbins':np.arange( R.min(),R.max(), 50), # sets median filter ring limits
        'nsigs':2.5, # how many absolute deviations from the median should a local max be to be a peak 
        'peak_COM':True, # if true uses center of mass (intnsity) to set peak pos, else uses 
        'run_rad_med':True, #whether to use the median radius thresholding
        'median_img':MEDIAN_IMG,  # median image (averaged over a run, can also be mean, its used to do background subctraction
        'median_sub':False,  # whether to do subtact the median image when doing streak detection 
        'mask':MASK} #mask im

    """
    PK_PAR =  { 'make_sparse':True,# leave true.. 
        'sig_G':0.8,'sig_G2':0.1,  # gaussian filter sigma (applied before local max filter, and then applied to streak detect)
        'thresh':1,  # local max with this value pixel (ADU) will be ignored
        'sz':8,  # sub image size used to  do local analysis
        'min_snr': 6.5,  # min snr, rough compute, optimized visually for each experiment, use this when 'filt' is True
        'filt': True, # this is my janky filtering I prob screwed it up in last commit, email me if bad... 
        'filt2':False, # this fits streaks to peaks, but it is slow. its hero mode.. 
        'min_dist':6, # min dist between pixels 
        'dist_cut':2, # dist cut is only for hero mode, verifies the streak is close to the peaks
        'r_in':None, # min res ring (only detect peaks within this radius
        'cent':cent,  # (fast scan, slow scan) center of pilatus
        'R':R, # provides radius of each pixel, pre-computed... same shape as pilatus
        'rbins':np.arange( R.min(),R.max(), 75), # sets median filter ring limits
        'nsigs':3.5, # how many absolute deviations from the median should a local max be to be a peak 
        'peak_COM':True, # if true uses center of mass (intnsity) to set peak pos, else uses 
        'run_rad_med':True, #whether to use the median radius thresholding
        'median_img':MEDIAN_IMG,  # median image (averaged over a run, can also be mean, its used to do background subctraction
        'median_sub':False,  # whether to do subtact the median image when doing streak detection 
        'mask':MASK} #mask im
    """
    log_s=""
#   one day ill log, for now... 
    if JID==0:
        print ("\n============================", PK_PAR, "\n==========================")
    H = h5py.File(fname, 'r')
    out = {"X":[], "Y":[] , "I":[], "fname":[], "dset_ind":[] }
    for counter, i in enumerate(inds):
        #fname, dset_ind, img = ALL_IMGS[i]
        img = H[data_path][i] 
        
        pk_pos, pkI = find_peaks4.pk_pos3(img*MASK, **PK_PAR)
        if not pk_pos:
            #print("BOOOS, found no peaks in %s, index %d"%(fname, i))
            s ="BOOOS, found no peaks in %s, index %d\n"%(fname, i)
            log_s += s
            print(s)
            pkY, pkX, pkI = [0],[0], [0]

        else:
            pkY,pkX = map(np.array, zip(*pk_pos))
        
        #print( "Job %d; img %d / %d; iteration %d; fname %s; found %d pks"%(JID, counter+1, len(inds), i,  fname, len(pkI))  )
        s ="Job %d; img %d / %d; iteration %d; fname %s; found %d pks\n"%(JID, counter+1, len(inds), i,  fname, len(pkI))  
        log_s += s
        print(s) 
        out['X'].append(pkX) # .append( [pkY, pkX, pkI, fname, dset_ind] )
        out['Y'].append(pkY)
        out['I'].append(pkI)
        out['dset_ind'].append( i)

    return out, log_s

def write_cxi_peaks( h5, peaks_path, pkX, pkY, pkI):
    assert( len(pkX) == len( pkY) == len( pkI) )
    
    npeaks = np.array( [len(x) for x in pkX] )
    max_n = max(npeaks)
    Nimg = len( pkX)
    
    data_x = np.zeros((Nimg, max_n), dtype=np.float32)
    data_y = np.zeros_like(data_x)
    data_I = np.zeros_like(data_x)

    for i in xrange( Nimg):
        n = int( npeaks[i] )
        data_x[i,:n] = pkX[i]
        data_y[i,:n] = pkY[i]
        data_I[i,:n] = pkI[i]
    
    peaks = h5.create_group(peaks_path)
    peaks.create_dataset( 'nPeaks' , data=npeaks)
    peaks.create_dataset( 'peakXPosRaw', data=data_x )
    peaks.create_dataset( 'peakYPosRaw', data=data_y )
    peaks.create_dataset( 'peakTotalIntensity', data=data_I ) 

if __name__ == "__main__":
    #fnames = glob.glob("./run*/*.cxi")
    
    fname = "pk_sim_8_mono.cxi" #glob.glob("/ufo1/taspase_pink/run*/*.cxi")
    data_path = "data"
    n_jobs = 3
    new_pk = "peaks"
    sys.stdout = Logger("pk_sim.log")
    
    h5 = h5py.File(fname,'r')
    N = h5[data_path].shape[0]
    inds = np.arange(N)
    h5.close()

    print("Found %d files and %d total hits"%( 1, N) )
    inds_split = np.array_split( inds, n_jobs)
    
    results = Parallel(n_jobs=n_jobs)( delayed(main)(fname, inds_split[JID], JID, data_path) \
        for JID in range( n_jobs))

    for out,log_s in results:
        print(log_s)

    # write the results
    with h5py.File( fname, 'r+')  as open_file:

        all_X = [out['X'] for out,_ in results ]
        all_Y = [out['Y'] for out,_ in results ]
        all_I = [out['I'] for out,_ in results ]
        
        inds = np.hstack( [ out['dset_ind'] for out,_ in results] )
      
        all_X = [x for sl in all_X for x in sl]
        all_Y = [x for sl in all_Y for x in sl]
        all_I = [x for sl in all_I for x in sl]

        write_cxi_peaks( open_file, new_pk, all_X,all_Y,all_I)

