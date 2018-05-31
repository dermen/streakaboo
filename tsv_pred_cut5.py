import numpy as np
import h5py
import pandas
import sys
from streakaboo import streak_peak

def get_im( fname, ind):
    h = h5py.File( fname, 'r')
    img = h['data'][ind]
    return img

input_pkl = sys.argv[1] #"small.refine.good.pred.pred.pkl" # known.pkl
#input_pkl = "omg-shit-2.pred.pkl" # known.pkl
output_pkl = sys.argv[2] #"result_orig" #output_temp.tsv
#mask_f = "Mar_a2a_newmask.npy" 
try_gauss=True #False
sig_G = 1.
dist_cut=6
mask_f = "airplane_mask.npy" 
mask = np.load(mask_f)

df = pandas.read_pickle(input_pkl)

output_h5 = sys.argv[3]

h5_out = h5py.File(output_h5, 'w')

df['I'] = 0. # integrated
df['sigma(I)'] = 0. # std
df['peak'] = 0. #? max intense
df['background'] = 0. 
df['panel'] = 'Mar'
df['pixels_in_streak'] = 0
df['has_signal'] = 0
df['circ'] = 0
df['used_gauss'] = 0
df['area'] = 0
df['round'] = 0
df['COM_fs'] = 0
df['COM_ss'] = 0

par = {'area_cut':1.2, 
    'connectivity_angle':10., 
    'min_points':0, 
    'radius_dev_cut':0.0, 
    'shape_cut':1.}




df.reset_index(inplace=True)
gb = df.groupby(('cxi_fname', 'dataset_index'))

print list( df ) 
NN = len( gb.groups)
count = 0
for (fname,ev), index in gb.groups.items():
    
    print count, NN
    img = mask* get_im( fname, ev)

    #df2 = gb.get_group( (fname, ev) )
    pkpos = df.loc[ index, \
        [ 'ss/px', 'fs/px' ]].values
    
    #pkpos = df2[[ 'ss/px', 'fs/px' ]].values

    #resol = df.loc[ index, '(1/d)/nm^-1'].values

    Y = pkpos[:,0]
    X = pkpos[:,1]

    img_sz = 20
    S = streak_peak.SubImages( img, Y,X, img_sz/2, mask=mask)
    I = []
    sigI = []
    BG = []
    peak = []

    circ = []
    used_gauss = []
    area = []
    round_ = []
    COM_ss = []
    COM_fs = []
    
    conn = []
    has_sig = []
     
    dset = h5_out.create_dataset( "%s_%d/data"%(fname, ev), 
        shape = (len( S.sub_imgs ), img_sz,img_sz) , 
        dtype=np.float)
    dset_mask = h5_out.create_dataset( "%s_%d/mask"%(fname, ev), 
        shape = (len( S.sub_imgs ), img_sz,img_sz), 
        dtype=np.bool )
    dset_sigmask = h5_out.create_dataset( "%s_%d/signal_mask"%(fname, ev), 
        shape = (len( S.sub_imgs ), img_sz,img_sz), 
        dtype=np.bool )
    dset_pixmask = h5_out.create_dataset( "%s_%d/pixel_mask"%(fname, ev), 
        shape = (len( S.sub_imgs ), img_sz,img_sz), 
        dtype=np.bool )
    dset_intens = h5_out.create_dataset( "%s_%d/intens"%(fname, ev), 
        shape = ( len( S.sub_imgs), ), 
        dtype=np.float )
    dset_noise = h5_out.create_dataset( "%s_%d/noise"%(fname, ev), 
        shape = ( len( S.sub_imgs), ), 
        dtype=np.float )
    #dset_res = h5_out.create_dataset( "%s_%d/resolution"%(fname, ev), 
    #    shape = ( len( S.sub_imgs), ), 
    #    dtype=np.float )

    
    empty_ar = np.zeros ( ( img_sz, img_sz ) )
    empty_ma = np.zeros ( ( img_sz, img_sz ) ).astype(bool)
    NN = len( S.sub_imgs)
    for i_s, s in enumerate(S.sub_imgs):
        if i_s%50==0:
            print("fname %d , sub img %d / %d" %( count ,i_s, NN) )
        s.integrate_pred_streak(dist_cut=dist_cut, sig_G=sig_G, try_gauss=try_gauss,**par)
        BG.append( s.bg)
        I.append( s.counts )
        sigI.append( s.sigma)
        peak.append( s.img[ s.rel_peak] ) # value at peak 
        conn.append( s.N_connected)
        circ.append( s.circ)
        area.append( s.area)
        has_sig.append( s.has_signal)
        used_gauss.append( s.used_gauss)
        ss_,fs_ = s.streak_COM
        COM_ss.append( ss_)
        COM_fs.append( fs_)
        round_.append( s.round)

        sy, sx = s.img.shape
        empty_ar[ :sy, :sx ] = s.img
        empty_ma [ :sy, :sx] = ~s.mask
        dset[i_s] = empty_ar
        dset_mask[i_s] = empty_ma 
        
        empty_ma*=False
        empty_ma [ :sy, :sx] = s.sig_mask
        dset_sigmask[i_s] = empty_ma
        
        
        empty_ma*=False
        empty_ma [ :sy, :sx] = s.pixmask
        dset_pixmask[i_s] = empty_ma

        dset_intens[i_s] = s.counts
        dset_noise[i_s] = s.sigma
        #dset_res[ i_s] = resol[ i_s] #s.radius

    df.loc[ index, [ 'I']] = I
    df.loc[ index, [ 'sigma(I)']] = sigI
    df.loc[ index, [ 'background']] = BG
    df.loc[ index, [ 'peak']] = peak
    df.loc[ index, [ 'pixels_in_streak']] =  conn
    
    df.loc[ index, [ 'circ']] = circ
    df.loc[ index, [ 'area']] = area
    df.loc[ index, [ 'round']] = round_
    df.loc[ index, [ 'COM_ss']] = COM_ss
    df.loc[ index, [ 'COM_fs']] = COM_fs
    df.loc[ index, [ 'used_gauss']] = used_gauss
    df.loc[ index, [ 'has_signal']] = has_sig
    
    count += 1


h5_out.close()
#output_tsv = "A2a_PDB.temp.tsv"
df.to_pickle(output_pkl)



