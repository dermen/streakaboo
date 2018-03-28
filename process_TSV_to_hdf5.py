import numpy as np
import h5py
import pandas
import sys
import streak_peak

def get_im( fname, ind):
    h = h5py.File( fname, 'r')
    img = h['data'][ind]
    return img

input_tsv = sys.argv[1] # FOUND.tsv
output_tsv = sys.argv[2] #output_temp.tsv

mask_f = sys.argv[3] # path to numpt mask
mask = np.load(mask_f)
#input_tsv = '/Users/damende/streaks/a2a/A2a_PDB.stream_Found.tsv' 

df = pandas.read_csv(input_tsv,
    sep='\t', header=0)

h5_out = h5py.File(output_tsv + ".h5py", 'w')

cols = {s:s.strip() for s in df }
df.rename( columns=cols, inplace=True)

df.cxi_file_name = [ '/work/a2a_weakpeaks/' \
    + f.strip() for f in df.cxi_file_name ]
#df.cxi_file_name = [ '/work/a2a_hits/' \
#    + f for f in df.cxi_file_name ]

df['I'] = 0. # integrated
df['sigma(I)'] = 0. # std
df['peak'] = 0. #? max intense
df['background'] = 0. 
df['panel'] = 'Mar'
df['pixels_in_streak'] = 0

par = {'area_cut':2, 
    'connectivity_angle':10., 
    'min_points':0, 
    'radius_dev_cut':0.0, 
    'shape_cut':2}

gb = df.groupby(('cxi_file_name', 'event_No'))


NN = len( gb.groups)
count = 0
for (fname,ev), index in gb.groups.items():
    print count, NN
    img = mask* get_im( fname, ev)
   
    pkpos = df.loc[ index, \
        [ 'ss/px', 'fs/px' ]].values

    resol = df.loc[ index, '(1/d)/nm^-1'].values

    Y = pkpos[:,0]
    X = pkpos[:,1]

    img_sz = 120
    S = streak_peak.SubImages( img, Y,X, img_sz/2)
    I = []
    sigI = []
    BG = []
    peak = []
     
    dset = h5_out.create_dataset( "%s_%d/data"%(fname, ev), 
        shape = (len( S.sub_imgs ), img_sz,img_sz) , 
        dtype=np.float)
    dset_mask = h5_out.create_dataset( "%s_%d/mask"%(fname, ev), 
        shape = (len( S.sub_imgs ), img_sz,img_sz), 
        dtype=np.bool )
    dset_intens = h5_out.create_dataset( "%s_%d/intens"%(fname, ev), 
        shape = ( len( S.sub_imgs), ), 
        dtype=np.float )
    dset_res = h5_out.create_dataset( "%s_%d/resolution"%(fname, ev), 
        shape = ( len( S.sub_imgs), ), 
        dtype=np.float )
    
    #dset_intens = h5_out.create_dataset( "%s_%d/intens"%(fname, ev), 
    #    shape = ( len( S.sub_imgs), ), 
    #    dtype=np.float )
    
    
    empty_ar = np.zeros ( ( img_sz, img_sz ) )
    empty_ma = np.zeros ( ( img_sz, img_sz ) ).astype(bool)
    conn = []
    print empty_ar.shape, dset.shape
    for i_s, s in enumerate(S.sub_imgs):
        s.integrate_streak(**par)
        BG.append( s.bg)
        I.append( s.counts )
        sigI.append( s.sigma)
        peak.append( s.img[ s.rel_peak] ) # value at peak 
        conn.append( s.N_connected)
        sy, sx = s.img.shape
        empty_ar[ :sy, :sx ] = s.img
        empty_ma [ :sy, :sx] = ~s.mask
        dset[i_s] = empty_ar
        dset_mask[i_s] = empty_ma 
        dset_intens[i_s] = s.counts
        dset_res[ i_s] = resol[ i_s] #s.radius

    df.loc[ index, [ 'I']] = I
    df.loc[ index, [ 'sigma(I)']] = sigI
    df.loc[ index, [ 'background']] = BG
    df.loc[ index, [ 'peak']] = peak
    df.loc[ index, [ 'pixels_in_streak']] =  conn
    count += 1

h5_out.close()
#output_tsv = "A2a_PDB.temp.tsv"
df.to_csv(output_tsv, 
    sep="\t", index=False)



