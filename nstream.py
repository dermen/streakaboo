import pandas
import os
import numpy as np
#pred_dat = df.loc[ index, ['h', 'k', 'l', 'I', 'sigma(I)', 'peak', 'background', 'fs/px', 'ss/px', 'panel']]
#hkl %4d
#I sigma peak background %10.2f
#fs ss %6.1f
I_form = lambda x: "%10.2f"%x
hkl_form = lambda x: "%4d"%x
px_form = lambda x: "%6.1f"
pred_forms={'h':hkl_form, 'k':hkl_form, 'l':hkl_form, 
     'I':I_form, 'sigma(I)':I_form, 'peak':I_form, 'background':I_form,
     'fs/px':px_form, 'ss/px':px_form}

pred_header = ['%4s'%'h', '%4s'%'k', '%4s'%'l', 
    '%12s'%'I', '%12s'%'sigma(I)', '%12s'%'peak', '%12s'%'background', '%6s'%'fs/px', '%6s'%'ss/px', 'panel']

#form={'h':hkl_form, 'k':hkl_form, 'l':hkl_form}

input_tsv = "big_highres.stream_Found.tsv.temp"
pkl_pref = "../../indexing/a2a/big_highres"
output_str = "big.refine_2Ang_20perc_2+.stream"
hkl_fr=0.2

max_pix_per_streak = 120
min_pix_per_streak=3
df = pandas.read_csv(input_tsv,sep='\t')
N1 = len(df)
df = df.query( 'pixels_in_streak <= %d'%max_pix_per_streak).reset_index()
df = df.query( 'pixels_in_streak >= %d'%min_pix_per_streak).reset_index()

df['H_dev'] = np.abs( df.H_f - df.H_i)
df['L_dev'] = np.abs( df.L_f - df.L_i)
df['K_dev'] = np.abs( df.K_f - df.K_i)
Hgood = lambda x:np.logical_or( df.H_dev <=x, df.H_dev >=1-x)
Lgood = lambda x:np.logical_or( df.L_dev <=x, df.L_dev >=1-x)
Kgood = lambda x:np.logical_or( df.K_dev <=x, df.K_dev >=1-x)
df = df.loc[Hgood(hkl_fr)*Lgood(hkl_fr)*Kgood(hkl_fr)]

N2 = len(df)
print("Went from %d rows to %d rows after applying filters..."%(N1,N2 ))

#df = pandas.read_csv("A2a_PDB.stream_Found_INTGRT.tsv",sep='\t')
df.cxi_file_name = map( os.path.basename, df.cxi_file_name )
df.rename(columns= {'cxi_file_name': 'cxi_fname', "event_No":"dataset_index"}, inplace=True)
df.rename( columns={'H_i':'h', 'K_i':'k', 'L_i':'l'}, inplace=1) 
pred = df.groupby( ('cxi_fname', 'dataset_index'))

df1 = pandas.read_pickle("%s.cell.pkl"%pkl_pref)
df1.cxi_fname = map( os.path.basename, df1.cxi_fname )
cell = df1.groupby( ('cxi_fname', 'dataset_index'))

df2 = pandas.read_pickle("%s.known.pkl"%pkl_pref)
df2.cxi_fname = map( os.path.basename, df2.cxi_fname )
found = df2.groupby( ('cxi_fname', 'dataset_index'))

groups = pred.groups.items()
g = groups[0]

o = open(output_str,'w')
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
    #o.write(found_dat.to_string(index=False))
    
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
    pred_dat = df.loc[ index, ['h', 'k', 'l', 'I', 'sigma(I)', 'peak', 'background', 'fs/px', 'ss/px', 'panel']]
    #pred_dat = df.loc[ index, ['h', 'k', 'l', 'I', 'sigma(I)', 'peak', 'background', 'fs/px', 'ss/px', 'panel']]
    o.write("   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel\n")
    S = '%4d %4d %4d %10.2f %10.2f %10.2f %10.2f %6.1f %6.1f %3s\n'
    for row in pred_dat.values:
        o.write(S%tuple(row)) 
#hkl %4d
#I sigma peak background %10.2f
#fs ss %6.1f
#panel
    #o.write( " ".join(pred_header)+"\n")
    #o.write(pred_dat.to_string(formatters=pred_forms, index=False, header=False))
    o.write("End of reflections\n")
    o.write("--- End crystal\n")
    o.write('----- End chunk -----\n')
    counter += 1

o.close()


