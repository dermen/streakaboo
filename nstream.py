import pandas
import os
import numpy as np

head="""CrystFEL stream format 2.3
Generated by CrystFEL 0.6.3+77cf2edd09bb01ae331935f467064c751f6e338e
indexamajig -i small.txt -o small_refine.stream -g geom --peaks=cxi --hdf5-peaks=/peaks --indexing=mosflm-latt-cell,asdf-latt-cell --int-radius=5,7,9 --integration=rings -p cell -j 11 --no-revalidate --highres=3.7 --fix-bandwidth=0.07 --no-multi
----- Begin geometry file -----
adu_per_photon = 1
clen =  0.25
coffset = 0.0
photon_energy = 12000
res = 3762.23 

data = /data

dim0 = %
dim1 = ss
dim2 = fs

Mar/min_fs = 0
Mar/min_ss = 0
Mar/max_fs = 1279
Mar/max_ss = 1279
Mar/fs = x
Mar/ss = y
Mar/corner_x = -661.4340
Mar/corner_y = -655.6679


badregionA/min_fs = 638
badregionA/max_fs = 800
badregionA/min_ss = 955
badregionA/max_ss = 1279

badregionB/min_fs = 794
badregionB/max_fs = 960
badregionB/min_ss = 715
badregionB/max_ss = 760

badregionC/min_fs = 953
badregionC/max_fs = 1121
badregionC/min_ss = 194
badregionC/max_ss = 240

badregionD/min_fs = 635
badregionD/max_fs = 800
badregionD/min_ss = 75
badregionD/max_ss = 125


badregionE/min_fs = 0
badregionE/max_fs = 682
badregionE/min_ss = 638
badregionE/max_ss = 665

badregionF/min_fs = 645
badregionF/max_fs = 680
badregionF/min_ss = 635
badregionF/max_ss = 685


badregionA/panel = Mar
badregionB/panel = Mar
badregionC/panel = Mar
badregionD/panel = Mar
badregionE/panel = Mar
badregionF/panel = Mar


----- End geometry file -----
----- Begin unit cell -----
CrystFEL unit cell file version 1.0

lattice_type = orthorhombic
centering = C
a = 40.00 A
b = 179.00 A
c = 142.00 A
al = 90.00 deg
be = 90.00 deg
ga = 90.00 deg
; Please note: this is the target unit cell.
; The actual unit cells produced by indexing depend on many other factors.
----- End unit cell -----\n"""

I_form = lambda x: "%10.2f"%x
hkl_form = lambda x: "%4d"%x
px_form = lambda x: "%6.1f"
pred_forms={'h':hkl_form, 'k':hkl_form, 'l':hkl_form, 
     'I':I_form, 'sigma(I)':I_form, 'peak':I_form, 'background':I_form,
     'fs/px':px_form, 'ss/px':px_form}

pred_header = ['%4s'%'h', '%4s'%'k', '%4s'%'l', 
    '%12s'%'I', '%12s'%'sigma(I)', '%12s'%'peak', '%12s'%'background', '%6s'%'fs/px', '%6s'%'ss/px', 'panel']

#######################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
input_pkl = "result_orig"
#input_pkl = "small_refine.pred.pkl"
#input_pkl = "shuffle_surprise.pkl"
#pkl_pref = "omg-shit-2"
pkl_pref = "small.refine.good.pred"
#output_str = "small.refine.good.pred.int.stream"
output_str = "result_orig.stream"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#######################################

df = pandas.read_pickle(input_pkl) #,sep='\t')
df = df.fillna(0)
df=df.reset_index()
pred = df.groupby( ('cxi_fname', 'dataset_index'))

df1 = pandas.read_pickle("%s.cell.pkl"%pkl_pref)
df1=df1.reset_index()
cell = df1.groupby( ('cxi_fname', 'dataset_index'))

df2 = pandas.read_pickle("%s.known.pkl"%pkl_pref)
df2=df2.reset_index()
found = df2.groupby( ('cxi_fname', 'dataset_index'))


print list(df)
print list( df1)
print list( df2)

groups = pred.groups.items()
g = groups[0]

o = open(output_str,'w')
o.write(head)
counter = 0
for (fname, event), index in  pred.groups.items():
    print (counter)
    
    found_dat = found.get_group( ( fname, event) )[ ['fs/px', 'ss/px', '(1/d)/nm^-1', 'Intensity', 'Panel'] ]
    cell_dat = cell.get_group( ( fname, event) )

    o.write('----- Begin chunk -----\n')
    o.write('Image filename: %s\n'%fname)
    o.write('Event: //%d\n'%event)
    o.write('Image serial number: %d\n'%(counter+1))
    o.write('indexed_by = mosflm-nolatt-nocell\n')
    o.write('photon_energy_eV = 12000.000000\n')
    o.write('beam_divergence = 0.00e+00 rad\n')
    o.write('beam_bandwidth = 1.00e-08 (fraction)\n')
    o.write('average_camera_length = 0.250000 m\n')
    o.write('num_peaks = %d\n'%len( found_dat))
    o.write('num_saturated_peaks = 0\n')
    o.write('Peaks from peak search\n')
    
    S = "%7.2f %7.2f %10.2f %11.2f %5s\n"
    F = found_dat.values
    o.write("  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel\n")
    for row in F:
        o.write(S%tuple(row))
    
    o.write('End of peak list\n')
    
    o.write("--- Begin crystal\n")
    a,b,c,al,be,ga = cell_dat[  [ 'a','b','c','alpha','beta','gamma' ] ].values[0]
    o.write('Cell parameters %.5f %.5f %.5f nm, %.5f %.5f %.5f deg\n'% ( a*.1,b*.1,c*.1,al,be,ga ) )
    
    astar1,astar2,astar3 = cell_dat['astar1'], cell_dat['astar2'], cell_dat['astar3']
    bstar1,bstar2,bstar3 = cell_dat['bstar1'], cell_dat['bstar2'], cell_dat['bstar3']
    cstar1,cstar2,cstar3 = cell_dat['cstar1'], cell_dat['cstar2'], cell_dat['cstar3']
    
    o.write('astar = %+.7f %+.7f %+.7f nm^-1\n' %(astar1*10, astar2*10, astar3*10))
    o.write('bstar = %+.7f %+.7f %+.7f nm^-1\n' %(bstar1*10, bstar2*10, bstar3*10))
    o.write('cstar = %+.7f %+.7f %+.7f nm^-1\n' %(cstar1*10, cstar2*10, cstar3*10))
    
    o.write('lattice_type = %s\n'%cell_dat['lattice_type'].values[0] )
    o.write('centering = %s\n'%cell_dat['centering'].values[0] )
    o.write('unique_axis = ?\n')
    o.write('profile_radius = 0.00227 nm^-1\n')
    o.write('predict_refine/det_shift x = 0.152 y = -0.023 mm\n')
    
    Alim = cell_dat['diffraction_resolution_limit']
    qlim = 10./Alim
    o.write( 'diffraction_resolution_limit = %.2f nm^-1 or %.2f A\n' %(qlim, Alim ))
    o.write( 'num_reflections = %d\n'%len(index))
    o.write('num_saturated_reflections = 0\n')
    o.write( 'num_implausible_reflections = 0\n')
    o.write('Reflections measured after indexing\n')
    #pred_dat = df.loc[ index, ['h', 'k', 'l', 'I', 'sigma(I)', 'peak', 'background', 'fs/px', 'ss/px', 'panel']]
    pred_dat =  pred.get_group( (fname, event ) )\
        [ ['h', 'k', 'l', 'I', 'sigma(I)', 'peak', 'background', 'fs/px', 'ss/px', 'panel']]
    o.write("   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel\n")
    S = '%4d %4d %4d %10.2f %10.2f %10.2f %10.2f %6.1f %6.1f %3s\n'
    for row in pred_dat.values:
        o.write(S%tuple(row)) 
    
    o.write("End of reflections\n")
    o.write("--- End crystal\n")
    o.write('----- End chunk -----\n')
    counter += 1

o.close()


