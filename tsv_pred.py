import numpy as np
import h5py
import pandas
import sys
from streakaboo import streak_peak

def get_im( fname, ind):
    h = h5py.File( fname, 'r')
    img = h['data'][ind]
    return img

input_pkl = "small.refine.good.pred.pred.pkl" # known.pkl
#input_pkl = "omg-shit-2.pred.pkl" # known.pkl
output_pkl = "result_orig" #output_temp.tsv
mask_f = "Mar_a2a_newmask.npy" 
mask = np.load(mask_f)

df = pandas.read_pickle(input_pkl)

h5_out = h5py.File(output_pkl + ".h5py", 'w')

df['I'] = 0. # integrated
df['sigma(I)'] = 0. # std
df['peak'] = 0. #? max intense
df['background'] = 0. 
df['panel'] = 'Mar'
df['pixels_in_streak'] = 0

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
    S = streak_peak.SubImages( img, Y,X, img_sz/2)
    I = []
    sigI = []
    BG = []
    peak = []
     
    #dset = h5_out.create_dataset( "%s_%d/data"%(fname, ev), 
    #    shape = (len( S.sub_imgs ), img_sz,img_sz) , 
    #    dtype=np.float)
    #dset_mask = h5_out.create_dataset( "%s_%d/mask"%(fname, ev), 
    #    shape = (len( S.sub_imgs ), img_sz,img_sz), 
    #    dtype=np.bool )
    #dset_intens = h5_out.create_dataset( "%s_%d/intens"%(fname, ev), 
    #    shape = ( len( S.sub_imgs), ), 
    #    dtype=np.float )
    #dset_res = h5_out.create_dataset( "%s_%d/resolution"%(fname, ev), 
    #    shape = ( len( S.sub_imgs), ), 
    #    dtype=np.float )
    
    #empty_ar = np.zeros ( ( img_sz, img_sz ) )
    #empty_ma = np.zeros ( ( img_sz, img_sz ) ).astype(bool)
    conn = []
    NN = len( S.sub_imgs)
    for i_s, s in enumerate(S.sub_imgs):
        if i_s%50==0:
            print("fname %d , sub img %d / %d" %( count ,i_s, NN) )
        s.integrate_pred_streak(**par)
        BG.append( s.bg)
        I.append( s.counts )
        sigI.append( s.sigma)
        peak.append( s.img[ s.rel_peak] ) # value at peak 
        conn.append( s.N_connected)
        #sy, sx = s.img.shape
        #empty_ar[ :sy, :sx ] = s.img
        #empty_ma [ :sy, :sx] = ~s.mask
        #dset[i_s] = empty_ar
        #dset_mask[i_s] = empty_ma 
        #dset_intens[i_s] = s.counts
        #dset_res[ i_s] = resol[ i_s] #s.radius

    df.loc[ index, [ 'I']] = I
    df.loc[ index, [ 'sigma(I)']] = sigI
    df.loc[ index, [ 'background']] = BG
    df.loc[ index, [ 'peak']] = peak
    df.loc[ index, [ 'pixels_in_streak']] =  conn
    count += 1

h5_out.close()
#output_tsv = "A2a_PDB.temp.tsv"
df.to_pickle(output_pkl)



