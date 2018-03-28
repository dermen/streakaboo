import h5py
import pandas
import sys
import streak_peak

def get_im( fname, ind):
    h = h5py.File( fname, 'r')
    img = h['data'][ind]
    return img

input_tsv = sys.argv[1]
output_tsv = sys.argv[2]

#input_tsv = '/Users/damende/streaks/a2a/A2a_PDB.stream_Found.tsv' 

df = pandas.read_csv(input_tsv,
    sep='\t', header=0)

cols = {s:s.strip() for s in df }
df.rename( columns=cols, inplace=True)

df.cxi_file_name = [ '/work/a2a_hits/' \
    + f for f in df.cxi_file_name ]

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
    img = get_im( fname, ev)
   
    pkpos = df.loc[ index, \
        [ 'ss/px', 'fs/px' ]].values

    Y = pkpos[:,0]
    X = pkpos[:,1]

    S = streak_peak.SubImages( img, Y,X, 30)
    I = []
    sigI = []
    BG = []
    peak = []
    conn = []
    for s in S.sub_imgs:
        s.integrate_streak(**par)
        BG.append( s.bg)
        I.append( s.counts )
        sigI.append( s.sigma)
        peak.append( s.img[ s.rel_peak] ) # value at peak 
        conn.append( s.N_connected)
    df.loc[ index, [ 'I']] = I
    df.loc[ index, [ 'sigma(I)']] = sigI
    df.loc[ index, [ 'background']] = BG
    df.loc[ index, [ 'peak']] = peak
    df.loc[ index, [ 'pixels_in_streak']] =  conn
    count += 1

#output_tsv = "A2a_PDB.temp.tsv"
df.to_csv(output_tsv, 
    sep="\t", index=False)


