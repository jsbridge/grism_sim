import os
import sys
import os.path
import shutil
import time

from pyraf import iraf
from iraf import stsdas, slitless, axesim
#from iraf import axesim

os.environ['AXE_IMAGE_PATH']   = './DATA/'
os.environ['AXE_CONFIG_PATH']  = './CONF/'
os.environ['AXE_OUTPUT_PATH']  = './OUTPUT/'
os.environ['AXE_OUTSIM_PATH']  = './OUTSIM/'
os.environ['AXE_SIMDATA_PATH'] = './SIMDATA/'

iraf.unlearn('simdata')

#print """
#---------------------------------
#
# Basic simulation for WFC3/G280
#"""
#iraf.unlearn('simdata')

# copy the model object table
#mot     = './save/one_object_G280.lis'
#mot_new = './DATA/one_object_G280.lis'
#print '\nCopy: ', mot, ' to: ', mot_new
#shutil.copy(mot, mot_new)

# make a basic simulation for WFC3/G280
#print '\n Simulating dispersed image for WFC3/G280 --> ./OUTSIM/WFC3_G280_slitless.fits'
#iraf.simdata(incat='one_object_G280.lis', config='WFC3.UV.CHIP1.TV3_sim.conf',
#             output_root='WFC3_G280', silent='NO', extraction='NO', adj_sens='YES')
#
#---------------------------------

#print """
#---------------------------------
#
# Basic simulation for WFC3/G280
#
# Please check the Model Object Table
# "./save/one_spec_im_object.lis"
# The fourth template spectrum in the list
# "input_spectra.lis" is used (#SPECTEMP = 4),
# it is redhifted to z=1.0. For flux shift the
# total passband "wfc3_uvis_f336w_tpass_m01.dat" is used
# The third template image in the list
# "input_images.lis" is used (#MODIMAGE = 3).
# Also a direct image is created (filter:
# "./SIMDATA/wfc3_uvis_f336w_tpass_m01.dat"), and noise is
# added to all ouput images!
#"""
#iraf.unlearn('simdata')

# copy the model object table
#mot     = './save/one_spec_G280.lis'
#mot_new = './DATA/one_spec_G280.lis'
#print '\nCopy: ', mot, ' to: ', mot_new
#shutil.copy(mot, mot_new)

# make a basic simulation for WFC3/G280
#print '\n Simulating dispersed image for WFC3/G280 --> ./OUTSIM/WFC3_G280_slitless.fits'
#iraf.simdata(incat='one_spec_G280.lis', config='WFC3.UV.CHIP1.TV3_sim.conf',
#             output_root='WFC3_templ_G280', silent='NO',
#             inlist_spec="input_spectra.lis", tpass_flux='wfc3_uvis_f336w_tpass_m01.dat',
#             inlist_ima="input_images.lis", exptime_disp=2000.0, bck_flux_disp=0.1,
#             extraction='no',
#             tpass_direct='wfc3_uvis_f336w_tpass_m01.dat', exptime_dir=50.0, bck_flux_dir=0.05)
#
#---------------------------------

#print """
#---------------------------------
#
# Basic simulation for WFC3/G102
#"""
#iraf.unlearn('simdata')

# copy the model object table
#mot     = './save/one_object_G102.lis'
#mot_new = './DATA/one_object_G102.lis'
#print '\nCopy: ', mot, ' to: ', mot_new
#shutil.copy(mot, mot_new)

# make a basic simulation for WFC3/G102
#print '\n Simulating dispersed image for WFC3/G102 --> ./OUTSIM/WFC3_G102_slitless.fits'
#iraf.simdata(incat='one_object_G102.lis', config='WFC3.IR.G102.SMOV.conf',
#             output_root='WFC3_G102',  silent='NO',extraction='yes')
#
#---------------------------------

#print """
#---------------------------------
#
# Basic simulation for WFC3/G102
#
# Please check the Model Object Table
# "./save/one_spec_im_object.lis"
# The fourth template spectrum in the list
# "input_spectra.lis" is used (#SPECTEMP = 4),
# it is redhifted to z=1.0. For flux shift the
# total passband "wfc3_ir_f110w_tpass_m01.dat" is used
# The third template image in the list
# "input_images.lis" is used (#MODIMAGE = 3).
# Also a direct image is created (filter:
# "./SIMDATA/wfc3_ir_f110w_tpass_m01.dat"), and noise is
# added to all ouput images!
#"""
#iraf.unlearn('simdata')

# copy the model object table
#mot     = './save/one_spec_G102.lis'
#mot_new = './DATA/one_spec_G102.lis'
#print '\nCopy: ', mot, ' to: ', mot_new
#shutil.copy(mot, mot_new)

# make a basic simulation for WFC3/G102
#print '\n Simulating dispersed image for WFC3/G102 --> ./OUTSIM/WFC3_templ_G102_slitless.fits'
#iraf.simdata(incat='one_spec_G102.lis', config='WFC3.IR.G102.SMOV.conf',
#             output_root='WFC3_templ_G102', silent='NO',
#             inlist_spec="input_spectra.lis", tpass_flux='wfc3_ir_f110w_tpass_m01.dat',
#             inlist_ima="input_images.lis", exptime_disp=500.0, bck_flux_disp=1.0,
#             extraction='YES',
#             tpass_direct='wfc3_ir_f110w_tpass_m01.dat', exptime_dir=50.0, bck_flux_dir=0.5)
#
#---------------------------------



#print """
#---------------------------------
#
# Basic simulation for WFC3/G141
#"""
#iraf.unlearn('simdata')

# copy the model object table
#mot     = './save/one_object_G141.lis'
#mot_new = './DATA/one_object_G141.lis'
#print '\nCopy: ', mot, ' to: ', mot_new
#shutil.copy(mot, mot_new)

# make a basic simulation for WFC3/G102
#print '\n Simulating dispersed image for WFC3/G141 --> ./OUTSIM/WFC3_G141_slitless.fits'
#iraf.simdata(incat='one_object_G141.lis', config='WFC3.IR.G141.SMOV.conf', output_root='WFC3_G141',
#              silent='NO',extraction='yes')
#
#---------------------------------

#print """
#---------------------------------
#
# Basic simulation for WFC3/G141
#
# Please check the Model Object Table
# "./save/one_spec_im_object.lis"
# The fourth template spectrum in the list
# "input_spectra.lis" is used (#SPECTEMP = 4),
# it is redhifted to z=1.0. For flux shift the
# total passband "wfc3_ir_f140w_tpass_m01.dat" is used
# The third template image in the list
# "input_images.lis" is used (#MODIMAGE = 3).
# Also a direct image is created (filter:
# "./SIMDATA/wfc3_ir_f140w_tpass_m01.dat"), and noise is
# added to all ouput images!
#"""
iraf.unlearn('simdata')

# copy the model object table
#mot     = './save/one_spec_G141.lis'
#mot_new = './DATA/one_spec_G141.lis'
#print '\nCopy: ', mot, ' to: ', mot_new
#shutil.copy(mot, mot_new)

# make a basic simulation for WFC3/G141
#print '\n Simulating dispersed image for WFC3/G141 --> ./OUTSIM/WFC3_G141_slitless.fits'
#iraf.simdata(incat='one_spec_G141.lis', config='WFC3.IR.G141.SMOV.conf',
#             output_root='WFC3_templ_G141',  silent='NO',inlist_spec="input_spectra.lis",
#             tpass_flux='wfc3_ir_f140w_tpass_m01.dat', inlist_ima="input_images.lis",
#             exptime_disp=500.0, bck_flux_disp=1.0, extraction='YES',
#             tpass_direct='wfc3_ir_f140w_tpass_m01.dat', exptime_dir=50.0, bck_flux_dir=0.5)


# Added by JSB 10/2014
def axe(output_root):
    mot     = './save/one_spec_G141.lis'
    mot_new = './DATA/one_spec_G141.lis'
    print '\nCopy: ', mot, ' to: ', mot_new
    shutil.copy(mot, mot_new)
    # make a fake galaxy simulation
    print '\n Simulating dispersed image for WFC3/G141 --> ./OUTSIM/WFC3_G141_slitless.fits'
    iraf.simdata(incat='one_spec_G141.lis', config='WFC3.IR.G141.SMOV.conf',
             output_root=output_root,  silent='YES', inlist_spec="input_spectra.lis",
             tpass_flux='wfc3_uvis_f475w_tpass_m01.dat', inlist_ima="input_images.lis",
             exptime_disp=4511.0, bck_flux_disp=4.0, extraction='YES',
             tpass_direct='wfc3_uvis_f475w_tpass_m01.dat', exptime_dir=811.0, bck_flux_dir=3.2)
    
    return
#
#---------------------------------

