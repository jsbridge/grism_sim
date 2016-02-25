########################################################################################
## This script and all it's functions should run all the simulations through aXeSIM      ##
##                                                                                       ##
## Joanna Bridge, 2015                                                                   ##
##                                                                                       ##
## Notes: 1.) All runs of aXeSIM use the 'exponential_disk.fits' and the 1400nm PSF      ##
##            fits files, i.e., no need to use mkobjects code                            ##
##        2.) Assumes a set AGN line ratio of [O III]/Hbeta = 1                          ##
##        3.) Must use Jon's IDL code before executing sim.gradients                     ##
##                                                                                       ##
###########################################################################################

import numpy as np
#import axesim_wfc3_new
import scipy.interpolate
import pyfits
import matplotlib.pyplot as plt
import os
from astropy.cosmology import FlatLambdaCDM
from glob import glob
#import mkobjects
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D
#import multi_regression_mcmc_fit
import matplotlib
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib.gridspec as gridspec
#import pandas as pd
#import seaborn as sns


def simulate_one():

    redshift = [1.8]                            # Mid-grism for now, low and hi z allowed by G141 grism later, maybe
    mass = [9, 9.5, 10, 10.5, 11, 11.5]         # log solar masses
    sSFR = [-9.5, -9.25, -9, -8.75, -8.5, -8.25, -8, -7.75, -7.5, -7.25, -7]       # log yr^-1
    logLbolLedd = [-3.5, -3, -2.5, -2, -1.5, -1, -0.5]      # erg/s
    r_AGN = 1                                   # Fixed logOIII/Hb for AGN
    mass_str = ['9.00', '9.50', '10.0', '10.5', '11.0', '11.5']
    
    for z in redshift:
        for i, m in enumerate(mass):
            for j, s in enumerate(sSFR):
                for k, t in enumerate(logLbolLedd):
                    #cont_mag = mass2mag(m)                              # Convert masses to continuum mags
                    #EW_disk = ssfr2EWgal(s, m, cont_mag, z)             # Find star formation line EW
                    #r_disk, met = SFRandMtoR(s, m)

                    #edit_onespec(z, cont_mag)                           # Edit the onespec file for running aXeSIM
                    
                    #w, d = spectrum_disk('disk.dat', cont_mag, EW_disk, r_disk) # Make the disk spectrum
                    #w, a = spectrum_AGN('AGN.dat', m, t, r_AGN, z)              # Make the AGN spectrum

                    file_ext = '1.8z_'+mass_str[i]+'M_'+str(s)+'sSFR_'+str(t)+'logLfrac'
                    #tabulate_rdisk(file_ext, r_disk, met)
                    #os.mkdir('../FITS_prop/'+file_ext)
                    for p in range(100):
                        file_ext1 = file_ext+'_'+str(p+1)   
                        print file_ext1                                    # Naming scheme, probably shit
                        #axesim_wfc3_new.axe(file_ext1)                     # Run aXeSIM
                        #interp(file_ext, file_ext1)                        # Interpolate the resulting spectrum
                        edit_testlist(file_ext, file_ext1, z)              # Edit testlist.dat for to use for Jon's code
                    
                    #plt.plot(w,d)
                    #plt.plot(w,a)
                    #plt.show()

    return


def simulate_many():

    redshift = [1.8]                                                          # Mid-grism 
    mass = [9, 9.5, 10, 10.5, 11, 11.5]                                       # log solar masses
    sSFR = [-9.5, -9.25, -9,-8.75, -8.5,-8.25 ,-8, -7.75,-7.5,-7.25, -7]  # log yr^-1 
    logLbolLedd = [-3.5, -3,-2.5,-2, -1.5, -1,-0.5]                        # erg/s
    mass_str = ['9.00', '9.50', '10.0', '10.5', '11.0', '11.5']
    r_AGN = 1                                                                # Fixed logOIII/Hb for AGN

    for z in redshift:
        for i, m in enumerate(mass):
            for j, s in enumerate(sSFR):
                for k, t in enumerate(logLbolLedd):
                    #cont_mag = mass2mag(m)                              # Convert masses to continuum mags
                    #EW_disk = ssfr2EWgal(s, m, cont_mag, z)             # Find star formation line EW
                    #r_disk, met = SFRandMtoR(s, m)

                    #kpc, pix = [],[]
                    #for b in xrange(100):
                    #    d, g = calc_Reff(m)
                    #    kpc.append(d)
                    #    pix.append(g)
             
                    #edit_onespec_many(z, cont_mag, kpc, pix)                       # Edit the onespec file for running aXeSIM
                    
                    #w, d = spectrum_disk('disk.dat', cont_mag, EW_disk, r_disk) # Make the disk spectrum
                    #w, a = spectrum_AGN('AGN.dat', m, t, r_AGN, z)              # Make the AGN spectrum

                    file_ext = '1.8z_'+mass_str[i]+'M_'+str(s)+'sSFR_'+str(t)+'logLfrac'
                    #if os.path.exists('../FITS/'+file_ext) == True:     # If the file exists already, delete it
                        #os.remove('../FITS/'+file_ext+'/*.fits')        # This line doesn't work, btw
                        #os.rmdir('../FITS/'+file_ext)
                        #print 'Deleted old FITS/'+file_ext+' first'
                    #os.mkdir('../FITS_1.4/'+file_ext)                      # Naming scheme, probably shit
                    #axesim_wfc3_new.axe(file_ext)                     # Run aXeSIM
                    #interp_many(file_ext)                             # Interpolate the resulting spectrum
                    for p in range(100):
                        file_ext1 = file_ext+'_'+str(p+1)                          
                        edit_testlist(file_ext, file_ext1, z)               # Edit testlist.dat for to use for Jon's code
                    
    return


def simulate_many_noAGN():

    redshift = [1.8]                                                          # Mid-grism 
    mass = [9, 9.5, 10, 10.5, 11, 11.5]                                       # log solar masses
    sSFR = [-9.5, -9.25, -9, -8.75, -8.5, -8.25, -8, -7.75, -7.5, -7.25, -7]  # log yr^-1 
    mass_str = ['9.00', '9.50', '10.0', '10.5', '11.0', '11.5']
    r_AGN = 0.8                                                                 # Fixed logOIII/Hb for AGN

    for z in redshift:
        for i, m in enumerate(mass):
            for j, s in enumerate(sSFR):
                
                cont_mag = mass2mag(m)                              # Convert masses to continuum mags
                EW_disk = ssfr2EWgal(s, m, cont_mag, z)             # Find star formation line EW
                r_disk, met = SFRandMtoR(s, m)

                kpc, pix = [],[]
                for b in xrange(100):
                    d, g = calc_Reff(m)
                    kpc.append(d)
                    pix.append(g)
             
                edit_onespec_noAGN(z, cont_mag, kpc, pix)                       # Edit the onespec file for running aXeSIM
                    
                w, d = spectrum_disk('disk.dat', cont_mag, EW_disk, r_disk) # Make the disk spectrum
               
                file_ext = '1.8z_'+mass_str[i]+'M_'+str(s)+'sSFR'

                for p in range(100):
                    file_ext1 = file_ext+'_'+str(p+1)
                    edit_testlist(file_ext, file_ext1, z)
                
                os.mkdir('../FITS_noAGN/'+file_ext)                      # Naming scheme, probably shit
                axesim_wfc3_new.axe(file_ext)                     # Run aXeSIM
                interp_many(file_ext)                             # Interpolate the resulting spectrum
                    
    return


def mkobj(kpc, pix):   # This makes the 'exponential_disk.fits' files for a range of sizes

    for i,r in enumerate(kpc):
        f = open('objfile.txt', 'w')
        f.write('  45     45    25   expdisk   '+str(r)+'    1     0.0')
        f.close()
        a = str(kpc[i]).split('.')
        mkobjects.mkobjects_iraf('input_fits.fits', 'SIMDATA/exponential_disk_'+a[0]+'p'+a[1]+'.fits', 'objfile.txt', 0, 25)
        f = open('input_images.lis', 'a')
        f.write('exponential_disk_'+a[0]+'p'+a[1]+'.fits\n')
        f.close()

    return


def edit_onespec_many(z, cont_mag, kpc, pix):      # spectemp is the line number of the fake galaxy in input_spectra.lis
    
    # read in the one_spec file
    fname = './save/one_spec_G141.lis'
    f = open(fname, 'w')
    f.write('# 1  NUMBER\n# 2  X_IMAGE\n# 3  Y_IMAGE\n# 4  A_IMAGE\n# 5  B_IMAGE\n# 6  THETA_IMAGE\n# 7  MAG_F475W\n# 8  Z\n# 9  SPECTEMP\n# 10 MODIMAGE\n')

    x = [0, 200, 400, 600, 800]
    y = [20, 60, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500, 540, 580, 620, 660, 700, 740, 780]
    
    i=0
    n=0
    for k in x:
        for u, l in enumerate(y):
            
            blah = np.arange(0.5, 10, 0.1)
            ind = [j for j, a in enumerate(blah) if "{0:.1f}".format(a) == "{0:.1f}".format(kpc[n])]
            if not ind:
                ind = [0]
                pix[n] = 0.5
            
            f.write(str(i+1))
            f.write(' '+str(k)+' '+str(l))
            f.write(' '+"{0:.2f}".format(pix[n])+' '+"{0:.2f}".format(pix[n]))
            f.write(' 90.0 30 ')
            f.write(str(z))
            f.write(' 1 1\n')
            f.write(str(i+2))
            f.write(' '+str(k)+' '+str(l))
            f.write(' '+"{0:.2f}".format(2*pix[n])+' '+"{0:.2f}".format(2*pix[n]))
            f.write(' 90.0 ')
            f.write(str(cont_mag))
            f.write(' '+str(z)+' ')
            f.write('2 '+str(2+ind[0])+'\n')

            i += 2
            n += 1
    return 


def interp_many(file_ext):   #Drizzle fits file from aXeSIM
    
    g = pyfits.open('OUTSIM/'+file_ext+'_slitless.fits')
    dat = g[1].data
    err = g[2].data

    x = np.arange(0, dat.shape[1])
    y = np.arange(0, dat.shape[0])
    xnew = np.arange(0, dat.shape[1]*2)/2.
    ynew = np.arange(0, dat.shape[0]*2)/2.
    
    f = scipy.interpolate.interp2d(x, y, dat, kind = 'cubic')
    dnew = f(xnew, ynew)
    dnew = (dnew * 4511 - np.median(dnew * 4511))/4511  # Subtract off the background (exptime = 4511)

    f = scipy.interpolate.interp2d(x, y, err, kind = 'cubic')
    enew = f(xnew, ynew)
    enew = np.sqrt((enew * 4511)**2 + np.median(dnew * 4511))/4511  # New error map, propagation of above errors
    
    i, k = 0, 1
    while i < 2000:
        j = 0
        p = 0
        while j < 1540:
            
            d = dnew[j:j+77, i:i+400]
            #m = np.argmax(chunk)
            #b = np.unravel_index(m, chunk.shape)
            #print b
            #d = chunk[(b[0]-28):(b[0]+27), (b[1]-161):(b[1]+160)]
            e = enew[j:j+80, i:i+400]
            #e = err_chunk[(b[0]-28):(b[0]+27), (b[1]-161):(b[1]+160)]

            if os.path.exists('../FITS_1.4/'+file_ext+'/driz_'+file_ext+'_'+str(k)+'.fits') == True:     # If the file exists already, delete it
                os.remove('../FITS_1.4/'+file_ext+'/driz_'+file_ext+'_'+str(k)+'.fits')
                #print 'Deleted old driz_'+file_ext+'_'+str(k)+'.fits first'
            pyfits.writeto('../FITS_1.4/'+file_ext+'/driz_'+file_ext+'_'+str(k)+'.fits', (), header=g[0].header)
            pyfits.append('../FITS_1.4/'+file_ext+'/driz_'+file_ext+'_'+str(k)+'.fits', d)            
            pyfits.append('../FITS_1.4/'+file_ext+'/driz_'+file_ext+'_'+str(k)+'.fits', e)

            p += 1
            j += 77
            k += 1
        i += 400
            
    os.remove('OUTSIM/'+file_ext+'_slitless_2.STP.fits')
    os.remove('OUTSIM/'+file_ext+'_slitless_2.SPC.fits')
    os.remove('OUTSIM/'+file_ext+'_images.fits')
    os.remove('OUTSIM/'+file_ext+'_direct.fits')
    os.remove('OUTSIM/'+file_ext+'_spectra.fits')
    #os.remove('OUTSIM/'+file_ext+'_slitless.fits')
    os.remove('DATA/'+file_ext+'_spectra.fits')
    os.remove('DATA/'+file_ext+'_images.fits')

    return


def tabulate_rdisk(file_ext, r_disk, met):   # Put disk OIII/Hbeta ratio in file

    f = open('../r_disk.dat', 'a')
    f.write(file_ext+'   '+str(r_disk)+'    '+str(met)+'\n')
    f.close()
    
    return


def interp(file_ext, file):   #Drizzle fits file from aXeSIM
    
    if os.path.exists('../FITS/'+file_ext+'/driz_'+file+'.fits') == True:     # If the file exists already, delete it
        os.remove('../FITS/'+file_ext+'/driz_'+file+'.fits')
        print 'Deleted old driz_'+file+'.fits first'
    
    p = pyfits.open('OUTSIM/'+file+'_slitless.fits')
    
    dat = p[1].data
    d = dat[490:513, 513:696]
    x = np.arange(0, d.shape[1])
    y = np.arange(0, d.shape[0])
    f = scipy.interpolate.interp2d(x, y, d, kind = 'cubic')
    xnew = np.arange(0, d.shape[1]*2)/2.
    ynew = np.arange(0, d.shape[0]*2)/2.
    dnew = f(xnew, ynew)
    dnew = (dnew * 4511 - np.median(dnew * 4511))/4511  # Subtract off the background (exptime = 4511)
    
    pyfits.writeto('../FITS/'+file_ext+'/driz_'+file+'.fits', (), header=p[0].header)
    pyfits.append('../FITS/'+file_ext+'/driz_'+file+'.fits', dnew)

    err = p[2].data
    e = err[490:513, 513:696]
    x = np.arange(0, e.shape[1])
    y = np.arange(0, e.shape[0])
    f = scipy.interpolate.interp2d(x, y, e, kind = 'cubic')
    xnew = np.arange(0, e.shape[1]*2)/2.
    ynew = np.arange(0, e.shape[0]*2)/2.
    enew = f(xnew, ynew)
    enew = np.sqrt((enew * 4511)**2 + np.median(dnew * 4511))/4511  # New error map, propagation of above errors
    
    pyfits.append('../FITS/'+file_ext+'/driz_'+file+'.fits', enew)

    os.remove('OUTSIM/'+file+'_slitless_2.STP.fits')
    os.remove('OUTSIM/'+file+'_slitless_2.SPC.fits')
    os.remove('OUTSIM/'+file+'_images.fits')
    os.remove('OUTSIM/'+file+'_direct.fits')
    os.remove('OUTSIM/'+file+'_spectra.fits')
    os.remove('OUTSIM/'+file+'_slitless.fits')
    os.remove('DATA/'+file+'_spectra.fits')
    os.remove('DATA/'+file+'_images.fits')

    return


def edit_testlist(file_ext, file_ext1, z):

    # read in the one_spec file
    fname = '../testlist.dat'
    f = open(fname, 'a')
    f.write(file_ext1+' '+str(z)+' FITS/'+file_ext+'/driz_'+file_ext1+'.fits\n')
    
    f.close()

    return


def spectrum_AGN(filename, mass, logLbolLedd, ratio, z):

    # logLbol/Ledd = logLbol - logMbh - 38.1
    # (Lbol/10^40) = 112(LOIII/10^40)^1.2
    # logLbol = 1.2LogOIII - 5.95
    # logMbh = logM*-3
    # ...math...
    # logLOIII/M* = (5/6)logLbol/Ledd - (1/6)logM* + 34.2
    # lobHb/M* = logLOIII/M* + 0.6

    logLhbMstar = (5*logLbolLedd/6.) - (mass/6.) + 34.8
    logLhb = logLhbMstar + mass
    cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)
    d = (cosmo.luminosity_distance(z)).value       # Mpc, must change to cm
    d = d * 3.0857e24
    C = 10**logLhb/(4 * np.pi * d**2)

    # Assumptions: [O III 5007]/[O III 4959] = 2.98, std = 10
    cont = 10**(-(30-23.9)/2.5) * 10**(-29)*3e18/(4830**2) # Add 30 mag as cont to match aXeSIM input
    sigma = 10

    wave = np.arange(0, 10000, 0.5)
    a = 1/np.sqrt(2*np.pi*sigma**2)
    g = gaussian(wave, a, 4861, sigma)   # This code is pinned on Hbeta
    rOIIIHb = 10**ratio
    

    Hbeta = C * g
    OIII5007 = C * rOIIIHb * gaussian(wave, a, 5007, sigma)
    OIII4959 = (C * rOIIIHb/2.98) * gaussian(wave, a, 4959, sigma)
    G = Hbeta + OIII5007 + cont + OIII4959

    f = open('SIMDATA/'+filename, 'w')
    for i, w in enumerate(wave):
        f.write('%.5e  ' % w)
        f.write('%.5e\n' % G[i])
    f.close()

    return wave, G


def mass2mag(input_mass):   # Single value input

    # mag2mass.txt is a file output from EzGal for mag range 17-30 and masses
    # Interpolate over to get exact mag for mass desired
    
    cmag, mass = np.loadtxt('mag2mass.txt', unpack=True)
    mass = np.log10(mass)
    f = scipy.interpolate.interp1d(mass, cmag)
    mass_new = np.arange(mass[-1], mass[0], 0.0001)
    cont_mag = f(mass_new)

    ind = (np.abs(mass_new-input_mass)).argmin()  # Index of closest mass value to input mass

    return cont_mag[ind]


def ssfr2EWgal(ssfr, mass, cont_mag, z):

    SFR = ssfr + mass
    L = 10**SFR/(7.9e-42)                          # Kennicutt 1998, also, dustless
    cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)
    d = (cosmo.luminosity_distance(z)).value       # Mpc, must change to cm
    d = d * 3.0857e24
    lineflux = L/(4 * np.pi * d**2)
    lineflux = lineflux/2.97
    cont = 10**(-(cont_mag-23.9)/2.5) * 10**(-29)  # This is in microJanskys, the 10^-29 puts in ergs/s/cm^2/Hz
    cont = cont * (3e18)/(4830**2)                 # Multiply by c/lambda^2 where wavelength is in Angtsroms, then ergs/s/cm^2/A
    EW = lineflux/cont
    
    return EW


def gaussian(x,a,x0,sigma):
    
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def calc_Reff(mass):      # van der Wel+ 2014 mass-size relation

    # A and a are z = 1.75, close to 1.8
    A = 10**0.65
    alpha = 0.23
    sigma = 10**0.18
    R = A * (10**mass/5e10)**alpha

    kpc = np.random.normal(R, sigma)
    arcs = kpc/8.443                    # 8.443 kpc/"
    pix = arcs/0.12825                  # 0.12825 "/pix
    
    return kpc, pix
    

def spectrum_disk(filename, cont_mag, EW, ratio):    # Makes the trio o' Gaussians

    # convert magnitude to f_lambda continuum
    cont = 10**(-(cont_mag-23.9)/2.5) * 10**(-29)
    # N.B. the wavelength below must be matched with the magnitude given in one_spec_G141.lis
    # i.e., currently it is set as MAG_F475W, so I use the pivot wavelength for that filter of 4830 A
    cont = cont * (3e18)/(4830**2)# * 1e18
    
    # Assumptions: [O III 5007]/[O III 4959] = 2.98, std = 10
    EW = EW
    sigma = 10

    wave = np.arange(0, 10000, 0.5)
    a = 1/np.sqrt(2*np.pi*sigma**2)
    g = gaussian(wave, a, 4861, sigma)   # This code is pinned on Hbeta
    C = EW * cont/np.trapz(g, wave)
    rOIIIHb = 10**ratio

    Hbeta = C * g
    OIII5007 = C * rOIIIHb * gaussian(wave, a, 5007, sigma)
    OIII4959 = (C * rOIIIHb/2.98) * gaussian(wave, a, 4959, sigma)
    G = Hbeta + OIII5007 + cont + OIII4959
    
    f = open('SIMDATA/'+filename, 'w')
    for i, w in enumerate(wave):
        f.write('%.5e  ' % w)
        f.write('%.5e\n' % G[i])
    f.close()

    return wave, G


def edit_onespec(z, cont_mag):      # spectemp is the line number of the fake galaxy in input_spectra.lis
    
    # read in the one_spec file
    fname = './save/one_spec_G141.lis'
    f = open(fname, 'r')
    lines = f.readlines()
    
    # replace the data with what I choose
    line = lines[-1].split(' ')
    line[7] = str(z)
    lines[-1] = ' '.join(line)

    line = lines[-2].split(' ')
    line[6] = str(cont_mag)
    line[7] = str(z)
    lines[-2] = ' '.join(line)
    
    f.close()

    # rewrite the lines to the same file
    f = open(fname, 'w')
    for l in lines:
        f.write(l)

    f.close()

    return 


def SFRandMtoR(sSFR, mass):

    #From Mannucci+ 2010, use SFR and mass to get 12+log(O/H)
    s = sSFR+mass
    #mu = mass - 0.32*s
    m = mass - 10
    #if mu >= 9.5:
    #    met = 8.9 + 0.37*m - 0.14*s - 0.19*m**2 + 0.12*m*s - 0.054*s**2
    #if mu < 9.5:
    #    met = 8.93 + 0.51*(mu - 10)
    met = 8.9 + 0.37*m - 0.14*s - 0.19*m**2 + 0.12*m*s - 0.054*s**2

    # From Maiolino+ 2008
    # x = 12 + logO/H - 8.69
    c0 = 0.1549
    c1 = -1.5031
    c2 = -0.9790
    c3 = -0.0297
    c4 = 0  
    x = met - 8.69
    logOIIIHb = c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4
    
    return logOIIIHb, met


def gradient():

    files = glob('../extract1dnew_no/*_rout.dat')
    x = open('../gradients_no.dat', 'w')
    x.write('#             ID                  z   logmasslogsSFR logLedd/Lbol  in_ratio       in_err       out_ratio        out_err\n')
    direct = 'extract1dnew_no/'
     
    for k, f in enumerate(files):

        file_ext = f.rstrip('_rout.dat')
        
        wave_in, flux_in = np.loadtxt(file_ext+'_rinn.dat', unpack=True)
        wave_in, err_in = np.loadtxt(file_ext+'_rinn_err.dat', unpack=True)
        wave_out, flux_out = np.loadtxt(file_ext+'_rout.dat', unpack=True)
        wave_out, err_out = np.loadtxt(file_ext+'_rout_err.dat', unpack=True)

        #wave_in = [ii for ii in wave_in_old if ii > 4800 and ii < 5060]
        #flux_in = [a for ii, a in enumerate(flux_in) if wave_in_old[ii] > 4800 and wave_in_old[ii] < 5060]
        #err_in = [a for ii, a in enumerate(err_in) if wave_in_old[ii] > 4800 and wave_in_old[ii] < 5060]
        #wave_out= [ii for ii in wave_out_old if ii > 4800 and ii < 5060]
        #flux_out= [a for ii, a in enumerate(flux_out) if wave_out_old[ii] > 4800 and wave_out_old[ii] < 5060]
        #err_out= [a for ii, a in enumerate(err_out) if wave_out_old[ii] > 4800 and wave_out_old[ii] < 5060]

        #wave_in = np.array(wave_in)
        #wave_out = np.array(wave_out)
        #flux_in = np.array(flux_in)
        #flux_out = np.array(flux_out)
        #err_in = np.array(err_in)
        #err_out = np.array(err_out)

        flux_new_in, flux_new_out = flux_in, flux_out
        r = 0
        while r < 2:
            sort = sorted(flux_new_in)
            size = np.size(flux_new_in)
            bad = np.logical_and(flux_new_in > sort[int(0.15*size)], (flux_new_in < sort[int(0.85*size)]))
            fit = np.poly1d(np.polyfit(np.array(wave_in)[bad], np.array(flux_new_in)[bad], 2))
            #fit = np.poly1d(np.polyfit([wave_in[0], wave_in[1]], [flux_new_in[0], flux_new_in[1]], 1))
            flux_new_in = flux_new_in - fit(wave_in)
            sort = sorted(flux_new_out)
            size = np.size(flux_new_out)
            bad = np.logical_and(flux_new_out > sort[int(0.15*size)], (flux_new_out < sort[int(0.85*size)]))
            fit = np.poly1d(np.polyfit(np.array(wave_in)[bad], np.array(flux_new_out)[bad], 2))
            #fit = np.poly1d(np.polyfit(wave_out, flux_new_out, 1))
            #fit = np.poly1d(np.polyfit([wave_out[0], wave_out[1]], [flux_new_out[0], flux_new_out[1]], 1))
            flux_new_out = flux_new_out - fit(wave_out)
            r += 1
        flux_in = flux_new_in
        flux_out = flux_new_out
    
        #plt.step(wave_in, flux_in)
        #plt.step(wave_in, flux_out)
        #plt.xlim(4500, 5600)
        #plt.savefig('help/'+str(k))

        lines = [4861, 4959, 5007]
        ind = []
        for l in lines:
            ind.append((np.abs(wave_in-l)).argmin())

        #print ind
        #print wave_in[ind[0]], wave_in[ind[1]], wave_in[ind[2]]
        #print flux_in[5], flux_in[6], flux_in[7], flux_in[8], flux_in[9]
        #print flux_in[17], flux_in[18], flux_in[19], flux_in[20], flux_in[21]
        #print flux_in[23], flux_in[24], flux_in[25], flux_in[26], flux_in[27]
        #print flux_in[202], flux_in[203], flux_in[204], flux_in[205], flux_in[206]
        #print flux_in[214], flux_in[215], flux_in[216], flux_in[217], flux_in[218]
        #print flux_in[219], flux_in[220], flux_in[221], flux_in[222], flux_in[223], flux_in[224], flux_in[225]
        #print flux_in[288], flux_in[289], flux_in[290], flux_in[291], flux_in[292]
        #print flux_in[302], flux_in[303], flux_in[304], flux_in[305], flux_in[306]
        #print flux_in[309], flux_in[310], flux_in[311], flux_in[312], flux_in[313]
        #print flux_in[116], flux_in[117], flux_in[118], flux_in[119], flux_in[120]
        #print flux_in[126], flux_in[127], flux_in[128], flux_in[129], flux_in[130]
        #print flux_in[131], flux_in[132], flux_in[133], flux_in[134], flux_in[135]
        #print flux_in[176], flux_in[177], flux_in[178], flux_in[179], flux_in[180]
        #print flux_in[188], flux_in[189], flux_in[190], flux_in[191], flux_in[192]
        #print flux_in[194], flux_in[195], flux_in[196], flux_in[197], flux_in[198]
        Hbeta_in = flux_in[ind[0]]
        Hbeta_in_err = err_in[ind[0]]
        Hbeta_out = flux_out[ind[0]]
        Hbeta_out_err = err_out[ind[0]]
        OIII4_in = flux_in[ind[1]-1]
        OIII4_in_err = err_in[ind[1]-1]
        OIII4_out = flux_out[ind[1]-1]
        OIII4_out_err = err_out[ind[1]-1]
        OIII5_in = flux_in[ind[2]]
        OIII5_in_err = err_in[ind[2]]
        OIII5_out = flux_out[ind[2]]
        OIII5_out_err = err_out[ind[2]]

        if Hbeta_in < 0:
            print file_ext+'Hbin'
            continue
        if Hbeta_out < 0:
            print file_ext+'Hbout'
            continue
        if OIII5_in < 0:
            print file_ext+'Oin'
            continue
        if OIII5_out < 0:
            print file_ext+'Oout'
            continue
        
        ratio_in = OIII5_in/Hbeta_in
        ratio_out = OIII5_out/Hbeta_out
        ratio_in_err = ratio_in * np.sqrt((OIII5_in_err/OIII5_in)**2 + (Hbeta_in_err/Hbeta_in)**2)
        ratio_out_err = ratio_out * np.sqrt((OIII5_out_err/OIII5_out)**2 + (Hbeta_out_err/Hbeta_out)**2)

        if ratio_in/ratio_in_err <= 3./np.sqrt(2):
            print file_ext+' in'
            continue
        if ratio_out/ratio_out_err <= 3./np.sqrt(2):
            print file_ext+' out'
            continue
        
        #print file_ext.lstrip('../'+direct)
        id = [i for i in str(1)+'.'+file_ext.lstrip('../'+direct)]
        #id = [i for i in file_ext.lstrip('../'+direct)]  
        
        z = id[0]+id[1]+id[2]
        #z =' 1.8'
        if (id[13] == '.') and (id[14] == '5'):
            if id[22] == '.':
                m = id[5]+id[6]+id[7]+id[8]
                s = id[11]+id[12]+id[13]+id[14]
                l = id[20]+id[21]+id[22]+id[23]
            else:
                m = id[5]+id[6]+id[7]+id[8]
                s = id[11]+id[12]+id[13]+id[14]
                l = id[20]+id[21]
        elif id[13] == '.' and (id[14] == '2' or id[14] == '7'):
            if id[23] == '.':
                m = id[5]+id[6]+id[7]+id[8]
                s = id[11]+id[12]+id[13]+id[14]+id[15]
                l = id[21]+id[22]+id[23]+id[24]
            else:
                m = id[5]+id[6]+id[7]+id[8]
                s = id[11]+id[12]+id[13]+id[14]+id[15]
                l = id[21]+id[22]
        elif id[13] != '.':
            if id[20] == '.':
                m = id[5]+id[6]+id[7]+id[8]
                s = id[11]+id[12]
                l = id[18]+id[19]+id[20]+id[21]
            else:
                m = id[5]+id[6]+id[7]+id[8]
                s = id[11]+id[12]
                l = id[18]+id[19]

        #if OIII5_in <= 0 or OIII5_out <= 0 or Hbeta_in <= 0 or Hbeta_out <= 0 or ratio_in/ratio_in_err <= 3./np.sqrt(2) or ratio_out/ratio_out_err <= 3./np.sqrt(2):
        #    x.write(str(1)+'.'+file_ext.lstrip('../'+direct))
        #    x.write('    '+z)
        #    x.write('    '+m)
        #    x.write('    '+s)
        #    x.write('    '+l)
        #    x.write('    nan')
        #    x.write('    nan')
        #    x.write('    nan')
        #    x.write('    nan\n')
        #else:
        x.write(str(1)+'.'+file_ext.lstrip('../'+direct))
        #x.write(file_ext.lstrip('../'+direct))
        #print ('1.8z'+file_ext.lstrip('../'+direct+'2.2z'))
        #x.write('1.8z'+file_ext.lstrip('../'+direct+'2.2z'))
        x.write('    '+z)
        x.write('    '+m)
        x.write('    '+s)
        x.write('    '+l)
        x.write('    '+str(ratio_in))
        x.write('    '+str(ratio_in_err))
        x.write('    '+str(ratio_out))
        x.write('    '+str(ratio_out_err)+'\n')

    x.close()
    
    return 


def gradient_noAGN():

    files = glob('../extract1dnew_noAGN/*_rout.dat')
    x = open('../gradients_noAGN_total.dat', 'w')
    x.write('#             ID                  z   logmasslogsSFR  in_ratio       in_err       out_ratio        out_err\n')
            
    for f in files:

        file_ext = f.rstrip('_rout.dat')
    
        wave_in, flux_in = np.loadtxt(file_ext+'_rinn.dat', unpack=True)
        wave_in, err_in = np.loadtxt(file_ext+'_rinn_err.dat', unpack=True)
        wave_out, flux_out = np.loadtxt(file_ext+'_rout.dat', unpack=True)
        wave_out, err_out = np.loadtxt(file_ext+'_rout_err.dat', unpack=True)
        
        flux_new_in, flux_new_out = flux_in, flux_out
        r = 0
        while r < 2:
            sort = sorted(flux_new_in)
            size = np.size(flux_new_in)
            bad = np.logical_and(flux_new_in > sort[int(0.15*size)], (flux_new_in < sort[int(0.85*size)]))
            fit = np.poly1d(np.polyfit(np.array(wave_in)[bad], np.array(flux_new_in)[bad], 2))
            flux_new_in = flux_new_in - fit(wave_in)
            sort = sorted(flux_new_out)
            size = np.size(flux_new_out)
            bad = np.logical_and(flux_new_out > sort[int(0.15*size)], (flux_new_out < sort[int(0.85*size)]))
            fit = np.poly1d(np.polyfit(np.array(wave_in)[bad], np.array(flux_new_out)[bad], 2))
            flux_new_out = flux_new_out - fit(wave_in)
            r += 1
        flux_in = flux_new_in
        flux_out = flux_new_out
    
        #plt.step(wave_in, flux_in)
        #plt.step(wave_in, flux_out)
        #plt.xlim(4500, 5600)
        #plt.show()

        lines = [4861, 4959, 5007]
        ind = []
        for l in lines:
            ind.append((np.abs(wave_in-l)).argmin())
                
        #print ind
        #print flux_in[202], flux_in[203], flux_in[204], flux_in[205], flux_in[206]       
        #print flux_in[214], flux_in[215], flux_in[216], flux_in[217], flux_in[218] 
        #print flux_in[219], flux_in[220], flux_in[221], flux_in[222], flux_in[223], flux_in[224], flux_in[225]     
        Hbeta_in = flux_in[ind[0]]
        Hbeta_in_err = err_in[ind[0]]
        Hbeta_out = flux_out[ind[0]]
        Hbeta_out_err = err_out[ind[0]]
        OIII4_in = flux_in[ind[1]-1]
        OIII4_in_err = err_in[ind[1]-1]
        OIII4_out = flux_out[ind[1]-1]
        OIII4_out_err = err_out[ind[1]-1]
        OIII5_in = flux_in[ind[2]]
        OIII5_in_err = err_in[ind[2]]
        OIII5_out = flux_out[ind[2]]
        OIII5_out_err = err_out[ind[2]]

        if Hbeta_in < 0:
            print file_ext
            #continue
        if Hbeta_out < 0:
            print file_ext
            #continue
        if OIII5_in < 0:
            print file_ext
            #continue
        if OIII5_out < 0:
            print file_ext
            #continue
        
        ratio_in = OIII5_in/Hbeta_in
        ratio_out = OIII5_out/Hbeta_out
        ratio_in_err = ratio_in * np.sqrt((OIII5_in_err/OIII5_in)**2 + (Hbeta_in_err/Hbeta_in)**2)
        ratio_out_err = ratio_out * np.sqrt((OIII5_out_err/OIII5_out)**2 + (Hbeta_out_err/Hbeta_out)**2)

        if ratio_in/ratio_in_err <= 3./np.sqrt(2):
            print file_ext+' in'
            #continue
        if ratio_out/ratio_out_err <= 3./np.sqrt(2):
            print file_ext+' out'
            #continue

        id = [i for i in str(1)+'.'+file_ext.lstrip('../extract1dnew_noAGN')]

        z = id[0]+id[1]+id[2]
        if (id[13] == '.') and (id[14] == '5'):
            m = id[5]+id[6]+id[7]+id[8]
            s = id[11]+id[12]+id[13]+id[14]
        elif id[13] == '.' and (id[14] == '2' or id[14] == '7'):
            m = id[5]+id[6]+id[7]+id[8]
            s = id[11]+id[12]+id[13]+id[14]+id[15]
        elif id[13] != '.':
            m = id[5]+id[6]+id[7]+id[8]
            s = id[11]+id[12]

        x.write(str(1)+'.'+file_ext.lstrip('../extract1dnew_noAGN'))
        x.write('    '+z)
        x.write('    '+m)
        x.write('    '+s)
        x.write('    '+str(ratio_in))
        x.write('    '+str(ratio_in_err))
        x.write('    '+str(ratio_out))
        x.write('    '+str(ratio_out_err)+'\n')

    x.close()
    
    return 


def medians():   # This little chunk of code reads in the results and puts like with like and finds medians of each

    z = 1.8
    mass, ssfr, lum, in_ratio, in_err, out_ratio, out_err = np.loadtxt('../gradients_no.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    ID = np.genfromtxt('../gradients_no.dat', unpack=True, usecols=(0), dtype='str')
    
    x = open('../medians_no.dat', 'w')
    x.write('#             ID                  z   logmasslogsSFR  logLbolLedd in_ratio          err_in        out_ratio      err_out \n')
    
    name = []
    for i, n in enumerate(ID):
        try:
            int(n[-3:])
            name.append(n[:-4])
        except ValueError:
            try:
                new = n[:-3]
                bad = n[-2:]
                int(bad)
                name.append(new)
            except ValueError:
                new = n[:-2]
                name.append(new)

    un = np.unique(name)  # Find the unique IDs in name
    for n in un:
        ind = [i for i, j in enumerate(name) if j == n]
        if len(mass[ind]) < 2:                                                                  
        #if (np.median(out_ratio[ind])/np.median(out_err[ind]) <= 3./np.sqrt(2)) or (np.median(in_ratio[ind])/np.median(in_err[ind]) <= 3./np.sqrt(2)):
            #print n
            continue      
            m = np.nan
            s = np.nan
            l = np.nan
            i_r = np.nan
            err_i = np.nan
            o_r = np.nan
            err_o = np.nan
        else:
            m = np.median(mass[ind])
            s = np.median(ssfr[ind])
            l = np.median(lum[ind])
            #plt.plot(sorted(in_ratio[ind]))
            #plt.show()
            inn = in_ratio[ind]
            i_r = np.median(inn[~np.isnan(inn)])
            err_i = np.std(inn[~np.isnan(inn)])
            out = out_ratio[ind]
            o_r = np.median(out[~np.isnan(out)])
            err_o = np.std(out[~np.isnan(out)])
            #i_r = np.median([r for r in in_ratio[ind] if r > 0])
            #err_i = np.median([in_err[i] for i,r in enumerate(in_ratio[ind]) if r > 0])
            #o_r = np.median([r for r in out_ratio[ind] if r > 0])
            #err_o = np.median([out_err[i] for i,r in enumerate(out_ratio[ind]) if r > 0])
        x.write(n)
        x.write('    '+str(z))
        x.write('    '+str(m))
        x.write('    '+str(s))
        x.write('    '+str(l))
        x.write('    '+str(i_r))
        x.write('    '+str(err_i))
        x.write('    '+str(o_r))
        x.write('    '+str(err_o)+'\n')
    
    x.close()

    return 


def int_medians():

    z = 1.8
    mass, ssfr, lum = np.loadtxt('../gradients_all.dat', unpack=True, usecols=(2,3,4))
    H,ratio = np.loadtxt('../integrated.dat', unpack=True, usecols=(1,3))
    ID = np.genfromtxt('../integrated.dat', unpack=True, usecols=(0), dtype='str')

    x = open('../int_medians.dat', 'w')
    x.write('#             ID                  z   logmass logsSFR logLedd/Lbol  ratio          err       \n')

    name = []
    for i, n in enumerate(ID):
        try:
            int(n[-3:])
            name.append(n[:-4])
        except ValueError:
            try:
                new = n[:-3]
                bad = n[-2:]
                int(bad)
                name.append(new)
            except ValueError:
                new = n[:-2]
                name.append(new)

    un = np.unique(name)  # Find the unique IDs in name                                                                   
    for n in un:
        ind = [i for i, j in enumerate(name) if j == n]
        m = np.median(mass[ind])
        s = np.median(ssfr[ind])
        l = np.median(lum[ind])
        r = np.median(ratio[ind])
        std = np.std(ratio[ind])
        
        x.write(n)
        x.write('    '+str(z))
        x.write('    '+str(m))
        x.write('    '+str(s))
        x.write('    '+str(l))
        x.write('    '+str(r))
        x.write('    '+str(std)+'\n')

    x.close()

    return


def plots():

    mass, ssfr, lum, in_ratio, in_err, out_ratio, out_err = np.loadtxt('../medians.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    #ID = np.genfromtxt('../medians.dat', unpack=True, usecols=(0), dtype='str')
    
    # Plot const mass bins with delta ratio vs sSFR
    # N.B. The following three arrays must be in the same order as these numbers occur in medians.dat!!
    mopt = [10, 10.5, 11, 11.5, 9, 9.5]
    sopt = [-7.25, -7.5, -7.75, -7, -8.25, -8.5, -8.75, -8, -9.25, -9.5, -9]
    lopt = [-0.5, -1.5, -1, -2.5, -2, -3.5, -3]
    #mopt = [9, 9.5, 10, 10.5, 11]
    #sopt = [-7.5, -7, -8.5, -8, -9]
    #lopt = [-0.5, -1.5, -2.5, -3.5] 
    
    m_constm = np.empty((len(mopt),77))
    s_constm = np.empty((len(mopt),77))
    l_constm = np.empty((len(mopt),77))
    inr_constm = np.empty((len(mopt),77))
    inerr_constm = np.empty((len(mopt),77))
    outr_constm = np.empty((len(mopt),77))
    outerr_constm = np.empty((len(mopt),77))
    
    for i, m in enumerate(mopt):
        s_constm[i] = ssfr[np.where(mass == m)]
        l_constm[i] = lum[np.where(mass == m)]
        inr_constm[i] = in_ratio[np.where(mass == m)]
        inerr_constm[i] = in_err[np.where(mass == m)]
        outr_constm[i] = out_ratio[np.where(mass == m)]
        outerr_constm[i] = out_err[np.where(mass == m)]

    for i in xrange(6):
        
        s = s_constm[i]
        l = l_constm[i]
        r = np.log10(inr_constm[i]/outr_constm[i])
        err_constm = r*np.sqrt((inerr_constm[i]/inr_constm[i])**2 + (outerr_constm[i]/outr_constm[i])**2)
        err_r = err_constm/(r*np.log(10))
        h, k, q, u, t, v, p = [],[],[],[],[],[],[]
        err_h, err_k, err_q, err_u, err_t, err_v, err_p = [],[],[],[],[],[],[]

        for j, a in enumerate(s):
            if l[j] == -3.5:
                h.append(r[j])
                err_h.append(err_r[j])
            if l[j] == -3:
                t.append(r[j])
                err_t.append(err_r[j])
            if l[j] == -2.5:
                k.append(r[j])
                err_k.append(err_r[j])
            if l[j] == -2:
                v.append(r[j])
                err_v.append(err_r[j])
            if l[j] == -1.5:
                q.append(r[j])
                err_q.append(err_r[j])
            if l[j] == -1:
                p.append(r[j])
                err_p.append(err_r[j])
            if l[j] == -0.5:
                u.append(r[j])
                err_u.append(err_r[j])
        h = np.array(h[0:])
        k = np.array(k[0:])
        q = np.array(q[0:])
        u = np.array(u[0:])
        t = np.array(t[0:])
        v = np.array(v[0:])
        p = np.array(p[0:])
        rawr = sorted(range(len(sopt)), key=lambda k: sopt[k])
        ssopt = sorted(sopt)
        plt.hlines(0.1, -10, -6, linestyle = '--')
        h = plt.errorbar(ssopt, [h[w] for w in rawr], yerr = [err_h[w] for w in rawr], ls = '-', color = 'r', marker = 'o', ms = 7, lw = 3, label='Edd ratio = -3.5')
        t = plt.errorbar(ssopt, [t[w] for w in rawr], yerr = [err_t[w] for w in rawr], ls = '-', color = 'c', marker = 'o', ms = 7, lw = 3, label='Edd ratio = -3.0')
        k = plt.errorbar(ssopt, [k[w] for w in rawr], yerr = [err_k[w] for w in rawr], ls = '-', color = 'b', marker = 'o', ms = 7, lw = 3, label='Edd ratio = -2.5')
        v = plt.errorbar(ssopt, [v[w] for w in rawr], yerr = [err_v[w] for w in rawr], ls = '-', color = 'k', marker = 'o', ms = 7, lw = 3, label='Edd ratio = -2.0')
        q = plt.errorbar(ssopt, [q[w] for w in rawr], yerr = [err_q[w] for w in rawr], ls = '-', color = 'g', marker = 'o', ms = 7, lw = 3, label='Edd ratio = -1.5')
        p = plt.errorbar(ssopt, [p[w] for w in rawr], yerr = [err_p[w] for w in rawr], ls = '-', color = 'm', marker = 'o', ms = 7, lw = 3, label='Edd ratio = -1.0')
        u = plt.errorbar(ssopt, [u[w] for w in rawr], yerr = [err_u[w] for w in rawr], ls = '-', color = 'y', marker = 'o', ms = 7, lw = 3, label='Edd ratio = -0.5')
        plt.xlabel(r'sSFR (yr$^{-1}$)')
        plt.ylabel(r'log([O III]/H$\beta$)$_{in}$ - log([O III]/H$\beta$)$_{out}$')
        plt.legend([h,t,k,v,q,p,u], [r'log($\lambda_{\rm Edd}$) = -3.5',r'log($\lambda_{\rm Edd}$) = -3.0',r'log($\lambda_{\rm Edd}$) = -2.5',r'log($\lambda_{\rm Edd}$) = -2.0',r'log($\lambda_{\rm Edd}$) = -1.5',r'log($\lambda_{\rm Edd}$) = -1.0',r'log($\lambda_{\rm Edd}$) = -0.5'], prop={'size':11})
        #plt.legend([t,k,v,q,p,u], [r'log($\lambda_{\rm Edd}$) = -3.0',r'log($\lambda_{\rm Edd}$) = -2.5',r'log($\lambda_{\rm Edd}$) = -2.0',r'log($\lambda_{\rm Edd}$) = -1.5',r'log($\lambda_{\rm Edd}$) = -1.0',r'log($\lambda_{\rm Edd}$) = -0.5'], prop={'size':11})
        plt.xlim(-9.6, -6.9)
        plt.ylim(-0.15, 0.75)
        name = str(mopt[i]).split('.')
        if len(name) == 1:
            plt.savefig('../gradient_plots/'+name[0]+'mass_ssfrvdelratio')
        if len(name) == 2:
            plt.savefig('../gradient_plots/'+name[0]+name[1]+'mass_ssfrvdelratio')
        if len(name) == 3:
            plt.savefig('../gradient_plots/'+name[0]+name[1]+name[2]+'mass_ssfrvdelratio')
        plt.close()



    # Plot const sSFR bins with delta ratio vs mass
    m_consts = np.empty((len(sopt),42))
    s_consts = np.empty((len(sopt),42))
    l_consts = np.empty((len(sopt),42))
    inr_consts = np.empty((len(sopt),42))
    inerr_consts = np.empty((len(sopt),42))
    outerr_consts = np.empty((len(sopt),42))
    outr_consts = np.empty((len(sopt),42))
    
    for i, s in enumerate(sopt):
        m_consts[i] = mass[np.where(ssfr == s)]
        s_consts[i] = ssfr[np.where(ssfr == s)]
        l_consts[i] = lum[np.where(ssfr == s)]
        inr_consts[i] = in_ratio[np.where(ssfr == s)]
        outr_consts[i] = out_ratio[np.where(ssfr == s)]
        inerr_consts[i] = in_err[np.where(ssfr == s)]
        outerr_consts[i] = out_err[np.where(ssfr == s)]
    
    for i in xrange(11):
        
        m = m_consts[i]
        l = l_consts[i]
        r = np.log10(inr_consts[i]/outr_consts[i])
        err_consts = r*np.sqrt((inerr_consts[i]/inr_consts[i])**2 + (outerr_consts[i]/outr_consts[i])**2)
        err_r = err_consts/(r*np.log(10))
        h, k, q, u, t, v, p = [],[],[],[],[],[],[]
        err_h, err_k, err_q, err_u, err_t, err_v, err_p = [],[],[],[],[],[],[]
     
        for j, a in enumerate(m):
            if l[j] == -3.5:
                h.append(r[j])
                err_h.append(err_r[j])
            if l[j] == -3:
                t.append(r[j])
                err_t.append(err_r[j])
            if l[j] == -2.5:
                k.append(r[j])
                err_k.append(err_r[j])
            if l[j] == -2:
                v.append(r[j])
                err_v.append(err_r[j])
            if l[j] == -1.5:
                q.append(r[j])
                err_q.append(err_r[j])
            if l[j] == -1:
                p.append(r[j])
                err_p.append(err_r[j])
            if l[j] == -0.5:
                u.append(r[j])
                err_u.append(err_r[j])
        h = np.array(h[0:])
        k = np.array(k[0:])
        q = np.array(q[0:])
        u = np.array(u[0:])
        t = np.array(t[0:])
        v = np.array(v[0:])
        p = np.array(p[0:])
        rawr = sorted(range(len(mopt)), key=lambda k: mopt[k])
        smopt = sorted(mopt)
        plt.hlines(0.1, 8, 12, linestyle = '--')
        h = plt.errorbar(smopt, [h[w] for w in rawr], yerr = [err_h[w] for w in rawr], ls = '-', color = 'r', marker = 'o', ms = 7, lw = 3, label='Edd ratio = -3.5')
        t = plt.errorbar(smopt, [t[w] for w in rawr], yerr = [err_t[w] for w in rawr], ls = '-', color = 'c', marker = 'o', ms = 7, lw = 3, label='Edd ratio = -3.0')
        k = plt.errorbar(smopt, [k[w] for w in rawr], yerr = [err_k[w] for w in rawr], ls = '-', color = 'b', marker = 'o', ms = 7, lw = 3, label='Edd ratio = -2.5')
        v = plt.errorbar(smopt, [v[w] for w in rawr], yerr = [err_v[w] for w in rawr], ls = '-', color = 'k', marker = 'o', ms = 7, lw = 3, label='Edd ratio = -1.5')
        q = plt.errorbar(smopt, [q[w] for w in rawr], yerr = [err_q[w] for w in rawr], ls = '-', color = 'g', marker = 'o', ms = 7, lw = 3, label='Edd ratio = -1.5')
        p = plt.errorbar(smopt, [p[w] for w in rawr], yerr = [err_p[w] for w in rawr], ls = '-', color = 'm', marker = 'o', ms = 7, lw = 3, label='Edd ratio = -2.5')
        u = plt.errorbar(smopt, [u[w] for w in rawr], yerr = [err_u[w] for w in rawr], ls = '-', color = 'y', marker = 'o', ms = 7, lw = 3, label='Edd ratio = -0.5')
        plt.xlim(8.9, 11.6)
        plt.ylim(-0.1, 0.6)
        plt.legend([h,t,k,v,q,p,u], [r'log($\lambda_{\rm Edd}$) = -3.5',r'log($\lambda_{\rm Edd}$) = -3.0',r'log($\lambda_{\rm Edd}$) = -2.5',r'log($\lambda_{\rm Edd}$) = -2.0',r'log($\lambda_{\rm Edd}$) = -1.5',r'log($\lambda_{\rm Edd}$) = -1.0',r'log($\lambda_{\rm Edd}$) = -0.5'], loc=2, prop={'size':11})
        #plt.legend([t,k,v,q,p,u], [r'log($\lambda_{\rm Edd}$) = -3.0',r'log($\lambda_{\rm Edd}$) = -2.5',r'log($\lambda_{\rm Edd}$) = -2.0',r'log($\lambda_{\rm Edd}$) = -1.5',r'log($\lambda_{\rm Edd}$) = -1.0',r'log($\lambda_{\rm Edd}$) = -0.5'], loc=2, prop={'size':11})       
        plt.xlabel(r'Mass (log(M$_*$/M$_{\odot}$)')
        plt.ylabel(r'log([O III]/H$\beta$)$_{in}$ - log([O III]/H$\beta$)$_{out}$')
        name = str(sopt[i]).split('.')
        if len(name) == 1:
            plt.savefig('../gradient_plots/'+name[0]+'ssfr_massvratio')
        if len(name) == 2:
            plt.savefig('../gradient_plots/'+name[0]+name[1]+'ssfr_massvratio')
        if len(name) == 3:
            plt.savefig('../gradient_plots/'+name[0]+name[1]+name[2]+'ssfr_massvratio')
        plt.close()
    

    # Plot Edd ratio as function of delta ratio
    s_constm = np.empty((len(mopt),77))
    l_constm = np.empty((len(mopt),77))
    inerr_constm = np.empty((len(mopt),77))
    outr_constm = np.empty((len(mopt),77))
    outerr_constm = np.empty((len(mopt),77))
    
    for i, m in enumerate(mopt):
        s_constm[i] = ssfr[np.where(mass == m)]
        l_constm[i] = lum[np.where(mass == m)]
        inr_constm[i] = in_ratio[np.where(mass == m)]
        inerr_constm[i] = in_err[np.where(mass == m)]
        outr_constm[i] = out_ratio[np.where(mass == m)]
        outerr_constm[i] = out_err[np.where(mass == m)]
        
    for i in xrange(6):
        
        s = s_constm[i]
        l = l_constm[i]
        r = np.log10(inr_constm[i]/outr_constm[i])
        err_constm = r*np.sqrt((inerr_constm[i]/inr_constm[i])**2 + (outerr_constm[i]/outr_constm[i])**2)
        err_r = err_constm/(r*np.log(10))
        err_h, err_k, err_q, err_u, err_t, err_v, err_p, err_y, err_aa, err_cc, err_b, err_dd = [],[],[],[],[],[],[],[],[],[],[],[]
        h, k, q, u, t, v, b, y, aa, cc, dd = [],[],[],[],[],[],[],[],[],[],[]

        for j, a in enumerate(l):
            if s[j] == -7:
                h.append(r[j])
                err_h.append(err_r[j])
            if s[j] == -7.25:
                t.append(r[j])
                err_t.append(err_r[j])
            if s[j] == -7.5:
                k.append(r[j])
                err_k.append(err_r[j])
            if s[j] == -7.75:
                y.append(r[j])
                err_y.append(err_r[j])
            if s[j] == -8:
                q.append(r[j])
                err_q.append(err_r[j])
            if s[j] == -8.25:
                aa.append(r[j])
                err_aa.append(err_r[j])
            if s[j] == -8.5:
                u.append(r[j])
                err_u.append(err_r[j])
            if s[j] == -8.75:
                cc.append(r[j])
                err_cc.append(err_r[j])
            if s[j] == -9:
                v.append(r[j])
                err_v.append(err_r[j])
            if s[j] == -9.25:
                dd.append(r[j])
                err_dd.append(err_r[j])
            if s[j] == -9.5:
                b.append(r[j])
                err_b.append(err_r[j])
        h = np.array(h)
        t = np.array(t)
        k = np.array(k)
        y = np.array(y)
        q = np.array(q)
        aa = np.array(aa)
        u = np.array(u)
        cc = np.array(cc)
        v = np.array(v)
        dd = np.array(dd)
        b = np.array(b)
        rawr = sorted(range(len(lopt)), key=lambda k: lopt[k])
        slopt = sorted(lopt)
        plt.vlines(0.1, -4, 0, linestyle = '--')
        h = plt.errorbar([h[w] for w in rawr], slopt, xerr = [err_h[w] for w in rawr], ls = '-', color = 'r', marker = 'o', ms = 7, lw = 3, label='sSFR = -7.0')
        #t = plt.errorbar([t[w] for w in rawr], slopt, xerr = [err_t[w] for w in rawr], ls = '-', color = 'm', marker = 'o', ms = 7, lw = 3, label='sSFR = -7.25')
        k = plt.errorbar([k[w] for w in rawr], slopt, xerr = [err_k[w] for w in rawr], ls = '-', color = 'b', marker = 'o', ms = 7, lw = 3, label='sSFR = -7.5')
        #y = plt.errorbar([y[w] for w in rawr], slopt, xerr = [err_y[w] for w in rawr], ls = '-', color = 'r', marker = 'o', ms = 7, lw = 3, label='sSFR = -7.75')
        q = plt.errorbar([q[w] for w in rawr], slopt, xerr = [err_q[w] for w in rawr], ls = '-', color = 'c', marker = 'o', ms = 7, lw = 3, label='sSFR = -8.0')
        #aa = plt.errorbar([aa[w] for w in rawr], slopt, xerr = [err_aa[w] for w in rawr], ls = '-', color = 'r', marker = 'o', ms = 7, lw = 3, label='sSFR = -8.25')
        u = plt.errorbar([u[w] for w in rawr], slopt, xerr = [err_u[w] for w in rawr], ls = '-', color = 'g', marker = 'o', ms = 7, lw = 3, label='sSFR = -8.5')
        #cc = plt.errorbar([cc[w] for w in rawr], slopt, xerr = [err_cc[w] for w in rawr], ls = '-', color = 'r', marker = 'o', ms = 7, lw = 3, label='sSFR = -8.75')
        v = plt.errorbar([v[w] for w in rawr], slopt, xerr = [err_v[w] for w in rawr], ls = '-', color = 'y', marker = 'o', ms = 7, lw = 3, label='sSFR = -9.0')
        #dd = plt.errorbar([dd[w] for w in rawr], slopt, xerr = [err_dd[w] for w in rawr], ls = '-', color = 'r', marker = 'o', ms = 7, lw = 3, label='sSFR = -9.25')
        b = plt.errorbar([b[w] for w in rawr], slopt, xerr = [err_b[w] for w in rawr], ls = '-', color = 'k', marker = 'o', ms = 7, lw = 3, label='sSFR = -9.5')
        plt.ylabel(r'Eddington Ratio (log $\lambda_{\rm Edd}$)')
        plt.xlabel(r'log([O III]/H$\beta$)$_{in}$ - log([O III]/H$\beta$)$_{out}$')
        plt.ylim(-3.7, -0.3)
        plt.xlim(-0.15,0.6)
        #plt.legend([h,t,k,y,q,aa,u,cc,v,dd,b], ['sSFR = -7.0','sSFR = -7.25','sSFR = -7.5','sSFR = -7.75', 'sSFR = -8.0','sSFR = -8.25', 'sSFR = -8.5','sSFR = -8.75','sSFR = -9.0','sSFR = -9.25','sSFR = -9.5'], loc = 0)
        plt.legend([h,k,q,u,v,b], ['log sSFR = -7.0','log sSFR = -7.5', 'log sSFR = -8.0', 'log sSFR = -8.5','log sSFR = -9.0','log sSFR = -9.5'], loc = 0, prop={'size':11})
        name = str(mopt[i]).split('.')
        if len(name) == 1:
            plt.savefig('../gradient_plots/'+name[0]+'mass_Leddvdelratio')
        if len(name) == 2:
            plt.savefig('../gradient_plots/'+name[0]+name[1]+'mass_Leddvdelratio')
        if len(name) == 3:
            plt.savefig('../gradient_plots/'+name[0]+name[1]+name[2]+'mass_Leddvdelratio')
        plt.close()

    return


def false_pos(cut):

    z = 1.8
    mass, ssfr, in_ratio, in_err, out_ratio, out_err = np.loadtxt('../gradients_no.dat', unpack=True, usecols=(2,3,4,5,6,7))
    ID = np.genfromtxt('../gradients_no.dat', unpack=True, usecols=(0), dtype='str')

    name = []
    for i, n in enumerate(ID):
        try:
            int(n[-3:])
            name.append(n[:-4])
        except ValueError:
            try:
                new = n[:-3]
                bad = n[-2:]
                int(bad)
                name.append(new)
            except ValueError:
                new = n[:-2]
                name.append(new)
        
    f = open('../false_positives.dat', 'w')
    un = np.unique(name)  # Find the unique IDs in name
    falses, ma, ss = [], [], []
    cat, meow, rr = [], [], []
    for n in un:
        ind = [i for i, j in enumerate(name) if j == n]
        m = np.median(mass[ind])
        s = np.median(ssfr[ind])
        i_r = in_ratio[ind]
        o_r = out_ratio[ind]
        i_e = in_err[ind]
        o_e = out_err[ind]
        io_err = (i_r/o_r)*np.sqrt(((i_e/i_r)**2 + (o_e/o_r)**2))
        r = np.log10(i_r/o_r)
        e = io_err/(r*np.log(10))
        
        a = 0
        for b,x in enumerate(r):
            if x > cut:# and m <= 10:
                a += 1.
        cat.append(len(ind))
        meow.append(m)
        perc_det = (a/len(ind))
        #print a
        f.write(n)
        f.write('    '+str(z))
        f.write('    '+str(m))
        f.write('    '+str(s))
        f.write('    '+str(np.nanmedian(r)))
        f.write('   '+str(perc_det)+'\n')
        rr.append(np.nanmedian(r))
        falses.append(perc_det)
        ma.append(m)
        ss.append(s)
        
    f.close()
    
    fig, ax = plt.subplots()
    im = ax.hexbin(ma, ss, gridsize = 6, C = falses cmap = plt.cm.YlGnBu)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('False Positives')
    ax.set_xlabel(r'Mass (log(M$_*$/M$_{\odot}$)')
    ax.set_ylabel(r'sSFR (yr$^{-1}$)')
    ax.set_xlim(8.5, 12)
    ax.set_ylim(-10, -6.5)
    im.set_clim(vmin =0,vmax=100)
    fig.savefig('../hex_plots/fals_pos_int')
    plt.close(fig)

    plt.plot(meow, cat, 'ko')
    plt.xlim(8.9, 11.6)
    plt.ylim(0, 105)
    plt.xlabel('mass')
    plt.ylabel('Number passed S/N cut out of 100')
    plt.savefig('../other_plots/massvsnum_tot')
    plt.close()

    return


def threshold():

    z = 1.8
    mass, ssfr, lum, in_ratio, in_err, out_ratio, out_err = np.loadtxt('../gradients.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    mass_no, ssfr_no, in_ratio_no, in_err_no, out_ratio_no, out_err_no = np.loadtxt('../gradients_noAGN.dat', unpack=True, usecols=(2,3,4,5,6,7))
    ID_no = np.genfromtxt('../gradients_noAGN.dat', unpack=True, usecols=(0), dtype='str')
    ID = np.genfromtxt('../gradients.dat', unpack=True, usecols=(0), dtype='str')
    
    name = []
    for i, n in enumerate(ID):
        try:
            int(n[-3:])
            name.append(n[:-4])
        except ValueError:
            try:
                new = n[:-3]
                bad = n[-2:]
                int(bad)
                name.append(new)
            except ValueError:
                new = n[:-2]
                name.append(new)

    fig, ax = plt.subplots(3,3, sharex=True, sharey=True)
    un = np.unique(name)  # Find the unique IDs in name
    r1, r2, r3, r4, r5, r6, r7, r8, r9 = [],[],[],[],[],[],[],[],[]
    r11, r22, r33, r44, r55, r66, r77, r88, r99 = [],[],[],[],[],[],[],[],[]
    for n in un:
        ind = [i for i, j in enumerate(name) if j == n]
        m = np.median(mass[ind])
        s = np.median(ssfr[ind])
        l = np.median(lum[ind])
        i_r = in_ratio[ind]
        o_r = out_ratio[ind]
        
        if m < 10 and s <= -8.5:
            if l > -2.5:
                r11 = np.append(r11, np.log10(i_r/o_r))
            r1 = np.append(r1, np.log10(i_r/o_r))
        if m < 10 and s > -8.5 and s <= -7.75:
            r2 = np.append(r2, np.log10(i_r/o_r))
            if l > -2.5:
                r22 = np.append(r22, np.log10(i_r/o_r))
        if m < 10 and s > -7.75 and s <= -7:
            r3 = np.append(r3, np.log10(i_r/o_r))
            if l > -2.5:
                r33 = np.append(r33, np.log10(i_r/o_r))
        if m < 11 and m >= 10 and s <= -8.5:
            if l > -2.5:
                r44 = np.append(r44, np.log10(i_r/o_r))
            r4 = np.append(r4, np.log10(i_r/o_r))
        if m < 11 and m >= 10 and s > -8.5 and s <= -7.75:
            if l > -2.5:
                r55 = np.append(r55, np.log10(i_r/o_r))
            r5 = np.append(r5, np.log10(i_r/o_r))
        if m < 11 and m >= 10 and s > -7.75 and s <= -7:
            if l > -2.5:
                r66 = np.append(r66, np.log10(i_r/o_r))
            r6 = np.append(r6, np.log10(i_r/o_r))
        if m >= 11 and s <= -8.5:
            if l > -2.5:
                r77 = np.append(r77, np.log10(i_r/o_r))
            r7 = np.append(r7, np.log10(i_r/o_r))
        if m >= 11 and s > -8.5 and s <= -7.75:
            if l > -2.5:
                r88 = np.append(r88, np.log10(i_r/o_r))
            r8 = np.append(r8, np.log10(i_r/o_r))
        if m >= 11 and s > -7.75 and s <= -7:
            if l > -2.5:
                r99 = np.append(r99, np.log10(i_r/o_r))
            r9 = np.append(r9, np.log10(i_r/o_r))
        
    values, base = np.histogram(r1, bins = 50)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[0,0].step(base[:-1], arr/(1.0*len(r1)), c = 'blue', lw=1.5)
    ax[0,0].set_yticklabels(['', '0.2', '0.4', '0.6', '0.8', '1.0'])

    values, base = np.histogram(r2, bins = 50)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[1,0].step(base[:-1], arr/(1.0*len(r2)), c = 'blue', lw=1.5)
    #ax[1,0].set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', ''])
    
    values, base = np.histogram(r3, bins = 50)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[2,0].step(base[:-1], arr/(1.0*len(r3)), c = 'blue', lw=1.5)
    ax[2,0].set_xlim(-1.5, 1)
    ax[2,0].set_xticklabels(['','-1.0', '','0.0','', ''])
    ax[2,0].xaxis.set_minor_locator(MultipleLocator(5))
    #ax[2,0].set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', ''])
        
    values, base = np.histogram(r4, bins = 50)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[0,1].step(base[:-1], arr/(1.0*len(r4)), c = 'blue', lw=1.5)

    values, base = np.histogram(r5, bins = 50)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[1,1].step(base[:-1], arr/(1.0*len(r5)), c = 'blue', lw=1.5)

    values, base = np.histogram(r6, bins = 50)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[2,1].step(base[:-1], arr/(1.0*len(r6)), c = 'blue', lw=1.5)
    ax[2,1].set_xlim(-1.5, 1)
    ax[2,1].set_xticklabels([' ','-1.0', ' ','0.0',' ', ' '])
    ax[2,1].xaxis.set_minor_locator(MultipleLocator(5))

    values, base = np.histogram(r7, bins = 50)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[0,2].step(base[:-1], arr/(1.0*len(r7)), c = 'blue', lw=1.5)

    values, base = np.histogram(r8, bins = 50)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[1,2].step(base[:-1], arr/(1.0*len(r8)), c = 'blue', lw=1.5)

    values, base = np.histogram(r9, bins = 50)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[2,2].step(base[:-1], arr/(1.0*len(r9)), c = 'blue', lw=1.5)
    ax[2,2].set_xlim(-1.5, 1)
    ax[2,2].set_xticklabels([' ','-1.0', ' ','0.0',' ', '1.0'])
    ax[2,2].xaxis.set_minor_locator(AutoMinorLocator(5))
    
    name = []
    for i, n in enumerate(ID_no):
        try:
            int(n[-3:])
            name.append(n[:-4])
        except ValueError:
            try:
                new = n[:-3]
                bad = n[-2:]
                int(bad)
                name.append(new)
            except ValueError:
                new = n[:-2]
                name.append(new)
                
    un = np.unique(name)  # Find the unique IDs in name
    for n in un:
        ind = [i for i, j in enumerate(name) if j == n]
        m = np.median(mass_no[ind])
        s = np.median(ssfr_no[ind])
        i_r = in_ratio_no[ind]
        o_r = out_ratio_no[ind]

        if m < 10 and s <= -8.5:
            r1 = (np.log10(i_r/o_r))
        if m < 10 and s > -8.5 and s <= -7.75:
            r2 = (np.log10(i_r/o_r))
        if m < 10 and s > -7.75 and s <= -7:
            r3 = (np.log10(i_r/o_r))
        if m < 11 and m >= 10 and s <= -8.5:
            r4 = (np.log10(i_r/o_r))
        if m < 11 and m >= 10 and s > -8.5 and s <= -7.75:
            r5 = (np.log10(i_r/o_r))
        if m < 11 and m >= 10 and s > -7.75 and s <= -7:
            r6 = (np.log10(i_r/o_r))
        if m >= 11 and s <= -8.5:
            r7 = (np.log10(i_r/o_r))
        if m >= 11 and s > -8.5 and s <= -7.75:
            r8 = (np.log10(i_r/o_r))
        if m >= 11 and s > -7.75 and s <= -7:
            r9 = (np.log10(i_r/o_r))
        
    values, base = np.histogram(r1, bins = 70)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[0,0].step(base[:-1], (len(r1) - arr)/(1.0*len(r1)), c = 'green', lw=1.5)
    ax[0,0].set_title(r'log(M$_*$/M$_{\odot}$) < 10', fontsize=11)

    values, base = np.histogram(r2, bins = 70)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[1,0].step(base[:-1], (len(r2) - arr)/(1.0*len(r2)), c = 'green', lw=1.5)
    
    values, base = np.histogram(r3, bins = 70)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[2,0].step(base[:-1], (len(r3) - arr)/(1.0*len(r3)), c = 'green', lw=1.5)
    
    values, base = np.histogram(r4, bins = 70)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[0,1].step(base[:-1], (len(r4) - arr)/(1.0*len(r4)), c = 'green', lw=1.5)
    ax[0,1].set_title(r'10 $\leq$ log(M$_*$/M$_{\odot}$) < 11', fontsize=11)

    values, base = np.histogram(r5, bins = 70)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[1,1].step(base[:-1], (len(r5) - arr)/(1.0*len(r5)), c = 'green', lw=1.5)

    values, base = np.histogram(r6, bins = 70)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[2,1].step(base[:-1], (len(r6) - arr)/(1.0*len(r6)), c = 'green', lw=1.5)

    values, base = np.histogram(r7, bins = 70)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[0,2].step(base[:-1], (len(r7) - arr)/(1.0*len(r7)), c = 'green', lw=1.5)
    ax[0,2].set_title(r'log(M$_*$/M$_{\odot}$) $\geq$ 11', fontsize=11)
    ax[0,2].yaxis.set_label_position('right')
    ax[0,2].set_ylabel(r'log(sSFR) $\leq$ -8.5', fontsize=9)

    values, base = np.histogram(r8, bins = 70)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[1,2].step(base[:-1], (len(r8) - arr)/(1.0*len(r8)), c = 'green', lw=1.5)
    ax[1,2].yaxis.set_label_position('right')
    ax[1,2].set_ylabel(r'-8.5 < log(sSFR) $\leq$ -7.75', fontsize=9)
 
    values, base = np.histogram(r9, bins = 70)
    cumulative = np.cumsum(values)
    arr = cumulative[::-1]
    ax[2,2].step(base[:-1], (len(r9) - arr)/(1.0*len(r9)), c = 'green', lw=1.5)
    ax[2,2].yaxis.set_label_position('right')
    ax[2,2].set_ylabel(r'-7.75 < log(sSFR) $\leq$ -7', fontsize=9)

    fig.subplots_adjust(wspace = 0.0, hspace = 0.0)
    fig.text(0.5, 0.02, r'log([O III]/H$\beta$)$_{in}$ - log([O III]/H$\beta$)$_{out}$', ha='center')
    fig.text(0.04, 0.5, 'Completeness', rotation='vertical', va='center')
        
    fig.savefig('../other_plots/cumsum')
    plt.close(fig)
    
    return


def complete(cut):

    z = 1.8
    mass, ssfr, lum, in_ratio, in_err, out_ratio, out_err = np.loadtxt('../gradients.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    ID = np.genfromtxt('../gradients.dat', unpack=True, usecols=(0), dtype='str')
    blah, rawr = np.loadtxt('../medians.dat', unpack=True, usecols=(0,2), dtype=[('one', '|S42'),('two',float)])

    name = []
    for i, n in enumerate(ID):
        try:
            int(n[-3:])
            name.append(n[:-4])
        except ValueError:
            try:
                new = n[:-3]
                bad = n[-2:]
                int(bad)
                name.append(new)
            except ValueError:
                new = n[:-2]
                name.append(new)
    
    p = []
    f = open('../completeness.dat', 'w')
    un = np.unique(name)  # Find the unique IDs in name
    for n in un:
        ind = [i for i, j in enumerate(name) if j == n]
        m = np.median(mass[ind])
        s = np.median(ssfr[ind])
        l = np.median(lum[ind])
        i_r = in_ratio[ind]
        o_r = out_ratio[ind]
        i_e = in_err[ind]
        o_e = out_err[ind]
        io_err = (i_r/o_r)*np.sqrt(((i_e/i_r)**2 + (o_e/o_r)**2))
        r = np.log10(i_r/o_r)
        e = io_err/(r*np.log(10))
        
        a = 0
        for b,x in enumerate(r):
            if x > cut:
                a += 1.
        perc_det = (a/len(ind))
        p.append(perc_det)
        f.write(n)
        f.write('    '+str(z))
        f.write('    '+str(m))
        f.write('    '+str(s))
        f.write('    '+str(l))
        f.write('   '+str(perc_det))
        f.write('   '+str(np.median(r))+'\n')

    f.close()

    return p


def many_cuts():

    a,b = [],[]
    cuts = np.arange(0, 0.6, 0.01)
    
    for c in cuts:
        a.append(complete(c))
        b.append(false_pos(c))

    plt.step(cuts, a, color='blue', lw=2)
    plt.step(cuts, 1-np.array(b), color='green', lw = 2)
    plt.savefig('what')
    plt.close()
    return a, b


def many_cuts_many_plots():

    z = 1.8
    mass, ssfr, lum, in_ratio, in_err, out_ratio, out_err = np.loadtxt('../gradients.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    ID = np.genfromtxt('../gradients.dat', unpack=True, usecols=(0), dtype='str')
    blah, rawr = np.loadtxt('../medians.dat', unpack=True, usecols=(0,2), dtype=[('one', '|S42'),('two',float)])

    r11, r22, r33, r44, r55, r66, r77, r88, r99 = [],[],[],[],[],[],[],[],[]
    cuts = np.arange(0, 0.6, 0.01)
    for cut in cuts:
        name = []
        for i, n in enumerate(ID):
            try:
                int(n[-3:])
                name.append(n[:-4])
            except ValueError:
                try:
                    new = n[:-3]
                    bad = n[-2:]
                    int(bad)
                    name.append(new)
                except ValueError:
                    new = n[:-2]
                    name.append(new)
                
        un = np.unique(name)  # Find the unique IDs in name
        r1, r2, r3, r4, r5, r6, r7, r8, r9 = [],[],[],[],[],[],[],[],[]
        
        for n in un:
            ind = [i for i, j in enumerate(name) if j == n]
            m = np.median(mass[ind])
            s = np.median(ssfr[ind])
            l = np.median(lum[ind])
            i_r = in_ratio[ind]
            o_r = out_ratio[ind]
            r = np.log10(i_r/o_r)
            r = r.clip(min=0)
        
            a=b=c=d=e=f=g=h=i=0
            for x in r:
                if x > cut and m < 10 and s <= -8.5:
                    a += 1.
                
                if x > cut and m < 10 and s > -8.5 and s <= -7.75:
                    b += 1.
                
                if x > cut and m < 10 and s > -7.75 and s <= -7:
                    c += 1.
                
                if x > cut and m < 11 and m >= 10 and s <= -8.5:
                    d += 1.
                
                if x > cut and m < 11 and m >= 10 and s > -8.5 and s <= -7.75:
                    e += 1.
                
                if x > cut and m < 11 and m >= 10 and s > -7.75 and s <= -7:
                    f += 1.
                
                if x > cut and m >= 11 and s <= -8.5:
                    g += 1.
                
                if x > cut and m >= 11 and s > -8.5 and s <= -7.75:
                    h += 1.
                
                if x > cut and m >= 11 and s > -7.75 and s <= -7:
                    i += 1.
            r9.append(i/len(ind))
            r1.append(a/len(ind))
            r2.append(b/len(ind))
            r3.append(c/len(ind))
            r4.append(d/len(ind))
            r5.append(e/len(ind))
            r6.append(f/len(ind))
            r7.append(g/len(ind))
            r8.append(h/len(ind))

        r11.append(np.sum(r1)/len(r1))
        r22.append(np.sum(r2)/len(r2))
        r33.append(np.sum(r3)/len(r3))
        r44.append(np.sum(r4)/len(r4))
        r55.append(np.sum(r5)/len(r5))
        r66.append(np.sum(r6)/len(r6))
        r77.append(np.sum(r7)/len(r7))
        r88.append(np.sum(r8)/len(r8))
        r99.append(np.sum(r9)/len(r9))
       
    fig, ax = plt.subplots(3,3, sharex=True, sharey=True)

    ax[0,0].step(cuts, r11, c = 'blue', lw=1.5)
    ax[0,0].set_xlim(-0.1, 0.6)
    ax[1,0].step(cuts, r22, c = 'blue', lw=1.5)
    ax[1,0].set_xlim(-0.1, 0.6)
    #ax[2,0].step(cuts, r33, c = 'blue', lw=1.5)
    ax[2,1].set_xticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
    ax[2,0].set_xlim(-0.1, 0.6)
    ax[0,1].step(cuts, r44, c = 'blue', lw=1.5)
    ax[0,1].set_xlim(-0.1, 0.6)
    ax[1,1].step(cuts, r55, c = 'blue', lw=1.5)
    ax[1,1].set_xlim(-0.1, 0.6)
    ax[2,1].step(cuts, r66, c = 'blue', lw=1.5)
    #ax[2,1].set_xticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
    ax[2,1].set_xlim(-0.1, 0.6)
    ax[0,2].step(cuts, r77, c = 'blue', lw=1.5)
    ax[0,2].set_xlim(-0.1, 0.6)
    ax[1,2].step(cuts, r88, c = 'blue', lw=1.5)
    ax[1,2].set_xlim(-0.1, 0.6)
    ax[2,2].step(cuts, r99, c = 'blue', lw=1.5)
    #ax[2,2].set_xticklabels(['', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
    ax[2,2].set_xlim(-0.1, 0.6)

    mass_no, ssfr_no, in_ratio_no, in_err_no, out_ratio_no, out_err_no = np.loadtxt('../gradients_noAGN.dat', unpack=True, usecols=(2,3,4,5,6,7))
    ID_no = np.genfromtxt('../gradients_noAGN.dat', unpack=True, usecols=(0), dtype='str')

    r11, r22, r33, r44, r55, r66, r77, r88, r99 = [],[],[],[],[],[],[],[],[]
    for cut in cuts:
        name = []
        for i, n in enumerate(ID_no):
            try:
                int(n[-3:])
                name.append(n[:-4])
            except ValueError:
                try:
                    new = n[:-3]
                    bad = n[-2:]
                    int(bad)
                    name.append(new)
                except ValueError:
                    new = n[:-2]
                    name.append(new)
                
        un = np.unique(name)  # Find the unique IDs in name
        r1, r2, r3, r4, r5, r6, r7, r8, r9 = [],[],[],[],[],[],[],[],[]
        
        for n in un:
            ind = [i for i, j in enumerate(name) if j == n]
            m = np.median(mass_no[ind])
            s = np.median(ssfr_no[ind])
            i_r = in_ratio_no[ind]
            o_r = out_ratio_no[ind]
            r = np.log10(i_r/o_r)
            r = r.clip(min=0)
        
            a = 0
            for b,x in enumerate(r):
                if x > cut and m < 10 and s <= -8.5:
                    a += 1.
                r1.append(a/len(ind))
            for b,x in enumerate(r):
                if x > cut and m < 10 and s > -8.5 and s <= -7.75:
                    a += 1.
                r2.append(a/len(ind))
            for b,x in enumerate(r):
                if x > cut and m < 10 and s > -7.75 and s <= -7:
                    a += 1.
                r3.append(a/len(ind))
            for b,x in enumerate(r):
                if x > cut and m < 11 and m >= 10 and s <= -8.5:
                    a += 1.
                r4.append(a/len(ind))
            for b,x in enumerate(r):
                if x > cut and m < 11 and m >= 10 and s > -8.5 and s <= -7.75:
                    a += 1.
                r5.append(a/len(ind))
            for b,x in enumerate(r):
                if x > cut and m < 11 and m >= 10 and s > -7.75 and s <= -7:
                    a += 1.
                r6.append(a/len(ind))
            for b,x in enumerate(r):
                if x > cut and m >= 11 and s <= -8.5:
                    a += 1.
                r7.append(a/len(ind))
            for b,x in enumerate(r):
                if x > cut and m >= 11 and s > -8.5 and s <= -7.75:
                    a += 1.
                r8.append(a/len(ind))
            for b,x in enumerate(r):
                if x > cut and m >= 11 and s > -7.75 and s <= -7:
                    a += 1.
                r9.append(a/len(ind))

        r11.append(np.sum(r1)/len(r1))
        r22.append(np.sum(r2)/len(r2))
        r33.append(np.sum(r3)/len(r3))
        r44.append(np.sum(r4)/len(r4))
        r55.append(np.sum(r5)/len(r5))
        r66.append(np.sum(r6)/len(r6))
        r77.append(np.sum(r7)/len(r7))
        r88.append(np.sum(r8)/len(r8))
        r99.append(np.sum(r9)/len(r9))

    ax[0,0].step(cuts, 1-np.array(r11), c = 'green', lw=1.5)
    ax[0,0].set_title(r'log(M$_*$/M$_{\odot}$) < 10', fontsize=11)
    ax[1,0].step(cuts, 1 - np.array(r22), c = 'green', lw=1.5)
    ax[2,0].step(cuts, 1 - np.array(r33), c = 'green', lw=1.5)
    ax[0,1].step(cuts, 1 - np.array(r44), c = 'green', lw=1.5)
    ax[0,1].set_title(r'10 $\leq$ log(M$_*$/M$_{\odot}$) < 11', fontsize=11)
    ax[1,1].step(cuts, 1 - np.array(r55), c = 'green', lw=1.5)
    ax[2,1].step(cuts, 1 - np.array(r66), c = 'green', lw=1.5)
    ax[0,2].step(cuts, 1 - np.array(r77), c = 'green', lw=1.5)
    ax[0,2].set_title(r'log(M$_*$/M$_{\odot}$) $\geq$ 11', fontsize=11)
    ax[0,2].yaxis.set_label_position('right')
    ax[0,2].set_ylabel(r'log(sSFR) $\leq$ -8.5', fontsize=9)
    ax[1,2].step(cuts, 1 - np.array(r88), c = 'green', lw=1.5)
    ax[1,2].yaxis.set_label_position('right')
    ax[1,2].set_ylabel(r'-8.5 < log(sSFR) $\leq$ -7.75', fontsize=9)
    ax[2,2].step(cuts, 1 - np.array(r99), c = 'green', lw=1.5)
    ax[2,2].yaxis.set_label_position('right')
    ax[2,2].set_ylabel(r'-7.75 < log(sSFR) $\leq$ -7', fontsize=9)
    
    
    return 


def many_cuts_many_plots2():

    z = 1.8
    mass, ssfr, lum, in_ratio, in_err, out_ratio, out_err = np.loadtxt('../gradients.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    ID = np.genfromtxt('../gradients.dat', unpack=True, usecols=(0), dtype='str')
    blah, rawr = np.loadtxt('../medians.dat', unpack=True, usecols=(0,2), dtype=[('one', '|S42'),('two',float)])
    
    name = []
    for i, n in enumerate(ID):
        try:
            int(n[-3:])
            name.append(n[:-4])
        except ValueError:
            try:
                new = n[:-3]
                bad = n[-2:]
                int(bad)
                name.append(new)
            except ValueError:
                new = n[:-2]
                name.append(new)
                
    un = np.unique(name)  # Find the unique IDs in name
    r11, r22, r33, r44, r55, r66, r77, r88, r99 = [],[],[],[],[],[],[],[],[]
    cuts = np.arange(0, 0.6, 0.01)

    for cut in cuts:
        
        r1, r2, r3, r4, r5, r6, r7, r8, r9 = [],[],[],[],[],[],[],[],[]
        a=b=c=d=e=f=g=h=k = 0
        aa=bb=cc=dd=ee=ff=gg=hh=kk=0
        for n in un:
            ind = [i for i, j in enumerate(name) if j == n]
            m = np.median(mass[ind])
            s = np.median(ssfr[ind])
            l = np.median(lum[ind])
            i_r = in_ratio[ind]
            o_r = out_ratio[ind]
            r = np.log10(i_r/o_r)
            r = r.clip(min=0)
            
            for x in r:
                if x > cut and m < 10 and s <= -8.5:
                    a += 1.
                if m < 10 and s <= -8.5:
                    aa += 1.
                if x > cut and m < 10 and s > -8.5 and s <= -7.75:
                    b += 1
                if m < 10 and s > -8.5 and s <= -7.75:
                    bb += 1.
                if x > cut and m < 10 and s > -7.75 and s <= -7:
                    c += 1
                if m < 10 and s > -7.75 and s <= -7:
                    cc += 1.
                if x > cut and m < 11 and m >= 10 and s <= -8.5:
                    d += 1
                if m < 11 and m >= 10 and s <= -8.5:
                    dd += 1.
                if x > cut and m < 11 and m >= 10 and s > -8.5 and s <= -7.75:
                    e += 1
                if m < 11 and m >= 10 and s > -8.5 and s <= -7.75:
                    ee += 1.
                if x > cut and m < 11 and m >= 10 and s > -7.75 and s <= -7:
                    f += 1
                if m < 11 and m >= 10 and s > -7.75 and s <= -7:
                    ff += 1.
                if x > cut and m >= 11 and s <= -8.5:
                    g += 1
                if m >= 11 and s <= -8.5:
                    gg += 1.
                if x > cut and m >= 11 and s > -8.5 and s <= -7.75:
                    h += 1
                if m >= 11 and s > -8.5 and s <= -7.75:
                    hh += 1.
                if x > cut and m >= 11 and s > -7.75 and s <= -7:
                    k += 1
                if m >= 11 and s > -7.75 and s <= -7:
                    kk += 1.
    
        r11.append(a/aa)
        r22.append(b/bb)
        r33.append(c/cc)
        r44.append(d/dd)
        r55.append(e/ee)
        r66.append(f/ff)
        r77.append(g/gg)
        r88.append(h/hh)
        r99.append(k/kk)

        #r11.append(np.sum(r1)/len(r1))
        #r22.append(np.sum(r2)/len(r2))
        #r33.append(np.sum(r3)/len(r3))
        #r44.append(np.sum(r4)/len(r4))
        #r55.append(np.sum(r5)/len(r5))
        #r66.append(np.sum(r6)/len(r6))
        #r77.append(np.sum(r7)/len(r7))
        #r88.append(np.sum(r8)/len(r8))
        #r99.append(np.sum(r9)/len(r9))

    fig, ax = plt.subplots(3,3, sharex=True, sharey=True)

    ax[0,0].step(cuts, r11, c = 'blue', lw=1.5)
    ax[0,0].set_yticklabels(['', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax[0,0].set_xlim(0, 0.6)
    ax[1,0].step(cuts, r22, c = 'blue', lw=1.5)
    ax[1,0].set_xlim(0, 0.6)
    ax[2,0].step(cuts, r33, c = 'blue', lw=1.5)
    ax[2,1].set_xticklabels(['', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
    ax[2,0].set_xlim(0, 0.6)
    ax[0,1].step(cuts, r44, c = 'blue', lw=1.5)
    ax[0,1].set_xlim(0, 0.6)
    ax[1,1].step(cuts, r55, c = 'blue', lw=1.5)
    ax[1,1].set_xlim(0, 0.6)
    ax[2,1].step(cuts, r66, c = 'blue', lw=1.5)
    ax[2,1].set_xticklabels(['', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
    ax[2,1].set_xlim(0, 0.6)
    ax[0,2].step(cuts, r77, c = 'blue', lw=1.5)
    ax[0,2].set_xlim(0, 0.6)
    ax[1,2].step(cuts, r88, c = 'blue', lw=1.5)
    ax[1,2].set_xlim(0, 0.6)
    ax[2,2].step(cuts, r99, c = 'blue', lw=1.5)
    ax[2,2].set_xticklabels(['', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
    ax[2,2].set_xlim(0, 0.6)

    mass_no, ssfr_no, in_ratio_no, in_err_no, out_ratio_no, out_err_no = np.loadtxt('../gradients_noAGN.dat', unpack=True, usecols=(2,3,4,5,6,7))
    ID_no = np.genfromtxt('../gradients_noAGN.dat', unpack=True, usecols=(0), dtype='str')
    
    name = []
    for i, n in enumerate(ID_no):
        try:
            int(n[-3:])
            name.append(n[:-4])
        except ValueError:
            try:
                new = n[:-3]
                bad = n[-2:]
                int(bad)
                name.append(new)
            except ValueError:
                new = n[:-2]
                name.append(new)
            
    un = np.unique(name)  # Find the unique IDs in name
    r11, r22, r33, r44, r55, r66, r77, r88, r99 = [],[],[],[],[],[],[],[],[]
    
    for cut in cuts:
        a=b=c=d=e=f=g=h=k = 0
        aa=bb=cc=dd=ee=ff=gg=hh=kk=0
        r1, r2, r3, r4, r5, r6, r7, r8, r9 = [],[],[],[],[],[],[],[],[]
        for n in un:
            ind = [i for i, j in enumerate(name) if j == n]
            m = np.median(mass_no[ind])
            s = np.median(ssfr_no[ind])
            i_r = in_ratio_no[ind]
            o_r = out_ratio_no[ind]
            r = np.log10(i_r/o_r)
            r = r.clip(min=0)
        
            for x in r:
                if x > cut and m < 10 and s <= -8.5:
                    a += 1.
                if m < 10 and s <= -8.5:
                    aa += 1.
                if x > cut and m < 10 and s > -8.5 and s <= -7.75:
                    b += 1
                if m < 10 and s > -8.5 and s <= -7.75:
                    bb += 1.
                if x > cut and m < 10 and s > -7.75 and s <= -7:
                    c += 1
                if m < 10 and s > -7.75 and s <= -7:
                    cc += 1.
                if x > cut and m < 11 and m >= 10 and s <= -8.5:
                    d += 1
                if m < 11 and m >= 10 and s <= -8.5:
                    dd += 1.
                if x > cut and m < 11 and m >= 10 and s > -8.5 and s <= -7.75:
                    e += 1
                if m < 11 and m >= 10 and s > -8.5 and s <= -7.75:
                    ee += 1.
                if x > cut and m < 11 and m >= 10 and s > -7.75 and s <= -7:
                    f += 1
                if m < 11 and m >= 10 and s > -7.75 and s <= -7:
                    ff += 1.
                if x > cut and m >= 11 and s <= -8.5:
                    g += 1
                if m >= 11 and s <= -8.5:
                    gg += 1.
                if x > cut and m >= 11 and s > -8.5 and s <= -7.75:
                    h += 1
                if m >= 11 and s > -8.5 and s <= -7.75:
                    hh += 1.
                if x > cut and m >= 11 and s > -7.75 and s <= -7:
                    k += 1
                if m >= 11 and s > -7.75 and s <= -7:
                    kk += 1.

        r11.append(a/aa)
        r22.append(b/bb)
        r33.append(c/cc)
        r44.append(d/dd)
        r55.append(e/ee)
        r66.append(f/ff)
        r77.append(g/gg)
        r88.append(h/hh)
        r99.append(k/kk)

        #r11.append(np.sum(r1)/len(r1))
        #r22.append(np.sum(r2)/len(r2))
        #r33.append(np.sum(r3)/len(r3))
        #r44.append(np.sum(r4)/len(r4))
        #r55.append(np.sum(r5)/len(r5))
        #r66.append(np.sum(r6)/len(r6))
        #r77.append(np.sum(r7)/len(r7))
        #r88.append(np.sum(r8)/len(r8))
        #r99.append(np.sum(r9)/len(r9))

    ax[0,0].step(cuts, 1-np.array(r11), c = 'green', lw=1.5)
    ax[0,0].set_title(r'log(M$_*$/M$_{\odot}$) $<$ 10', fontsize=11)
    ax[1,0].step(cuts, 1 - np.array(r22), c = 'green', lw=1.5)
    ax[2,0].step(cuts, 1 - np.array(r33), c = 'green', lw=1.5)
    ax[0,1].step(cuts, 1 - np.array(r44), c = 'green', lw=1.5)
    ax[0,1].set_title(r'10 $\leq$ log(M$_*$/M$_{\odot}$) $<$ 11', fontsize=11)
    ax[1,1].step(cuts, 1 - np.array(r55), c = 'green', lw=1.5)
    ax[2,1].step(cuts, 1 - np.array(r66), c = 'green', lw=1.5)
    ax[0,2].step(cuts, 1 - np.array(r77), c = 'green', lw=1.5)
    ax[0,2].set_title(r'log(M$_*$/M$_{\odot}$) $\geq$ 11', fontsize=11)
    ax[0,2].yaxis.set_label_position('right')
    ax[0,2].set_ylabel(r'log(sSFR) $\leq$ $-8.5$', fontsize=9)
    ax[1,2].step(cuts, 1 - np.array(r88), c = 'green', lw=1.5)
    ax[1,2].yaxis.set_label_position('right')
    ax[1,2].set_ylabel(r'$-8.5$ $<$ log(sSFR) $\leq$ $-7.75$', fontsize=9)
    ax[2,2].step(cuts, 1 - np.array(r99), c = 'green', lw=1.5)
    ax[2,2].yaxis.set_label_position('right')
    ax[2,2].set_ylabel(r'$-7.75$ $<$ log(sSFR) $\leq$ $-7$', fontsize=9)
    #ax[2.2].legend(['Completeness', 'Reliability'], color=['blue', 'green'])
    
    fig.subplots_adjust(wspace = 0.0, hspace = 0.0)
    fig.text(0.5, 0.02, r'$\Delta$log([O III]/H$\beta$) Threshold', ha='center')
    fig.text(0.04, 0.5, 'Completeness', rotation='vertical', va='center')
    
    fig.text(0.72, 0.3, 'Completeness', color='blue', fontsize=14)
    fig.text(0.72, 0.26, 'Reliability', color='green', fontsize=14)
        
    fig.savefig('../other_plots/cumsum')
    plt.close(fig)
    
    return 


def complete_total():

    z = 1.8
    mass, ssfr, lum = np.loadtxt('../gradients_all.dat', unpack=True, usecols=(2,3,4))
    Hb, tot_ratio = np.loadtxt('../integrated.dat', unpack=True, usecols=(1,3))
    ID = np.genfromtxt('../gradients_all.dat', unpack=True, usecols=(0), dtype='str')
    blah, err = np.loadtxt('../int_medians.dat', unpack=True, usecols=(0,2), dtype=[('one', '|S42'),('two',float)])

    m1 = np.arange(7, 10, 0.01)
    m2 = np.arange(10, 14, 0.01)
    y1 = 0.375/(m1-10.5)+1.14
    y2 = 410.24 - 109.333*m2 + 9.71731*m2**2 - 0.288244*m2**3

    name = []
    for i, n in enumerate(ID):
        try:
            int(n[-3:])
            name.append(n[:-4])
        except ValueError:
            try:
                new = n[:-3]
                bad = n[-2:]
                int(bad)
                name.append(new)
            except ValueError:
                new = n[:-2]
                name.append(new)
                
    f = open('../completeness_total.dat', 'w')
    un = np.unique(name)  # Find the unique IDs in name
    p = []
    for n in un:
        ind = [i for i, j in enumerate(name) if j == n]
        m = np.median(mass[ind])
        s = np.median(ssfr[ind])
        l = np.median(lum[ind])
        r = np.log10(tot_ratio[ind])
        #print n
        #print blah
        e = (err[np.where(blah == n)])/((tot_ratio[ind])*np.log(10))
        
        a = 0
        for j,x in enumerate(r):
            if m <= 10:
                cut = y1[(np.abs(m1-m)).argmin()]
            else:
                cut = y2[(np.abs(m2-m)).argmin()]
            if (x-e[j]) > cut:      
                a += 1.
        perc_det = a/len(ind)
        p.append(perc_det)
        f.write(n)
        f.write('    '+str(z))
        f.write('    '+str(m))
        f.write('    '+str(s))
        f.write('    '+str(l))
        f.write('   '+str(perc_det)+'\n')

    f.close()

    return p


def plot_comp():

    mass, ssfr, lum, comp = np.loadtxt('../completeness.dat', unpack=True, usecols=(2,3,4,5))
    
    fig, ax = plt.subplots()
    im = ax.hexbin(ssfr, lum, gridsize = 6, C = comp, cmap = plt.cm.YlGnBu)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Completeness')
    ax.set_xlabel(r'sSFR (yr$^{-1}$)')
    ax.set_ylabel(r'Eddington Ratio (log $\lambda_{\rm Edd}$)')
    ax.set_xlim(-10, -6.5)
    ax.set_ylim(-4, 0)
    im.set_clim(vmin =0,vmax=1)
    plt.savefig('../hex_plots/hex_diff_ssfr')
    plt.close()

    fig, ax = plt.subplots()
    im = ax.hexbin(mass, lum, gridsize = 6, C = comp, cmap = plt.cm.YlGnBu)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Completeness')
    ax.set_xlabel(r'Mass (log(M$_*$/M$_{\odot}$)')
    ax.set_ylabel(r'Eddington Ratio (log $\lambda_{\rm Edd}$)')
    ax.set_xlim(8.5, 12)
    ax.set_ylim(-4, 0)
    im.set_clim(vmin =0,vmax=1)
    fig.savefig('../hex_plots/hex_diff_mass')
    plt.close(fig)

    fig, ax = plt.subplots()
    im = ax.hexbin(mass, ssfr, gridsize = 6, C = comp, cmap = plt.cm.YlGnBu)
    ax.set_xlabel(r'Mass (log(M$_*$/M$_{\odot}$)')
    ax.set_ylabel(r'sSFR (yr$^{-1}$)')
    ax.set_xlim(8.5, 12)
    ax.set_ylim(-10, -6.5)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Completeness')
    im.set_clim(vmin =0,vmax=1)
    fig.savefig('../hex_plots/hex_diff_mass_ssfr')
    plt.close(fig)


    mass, ssfr, lum, comp = np.loadtxt('../completeness_total.dat', unpack=True, usecols=(2,3,4,5))

    fig, ax = plt.subplots()
    im = ax.hexbin(ssfr, lum, gridsize = 6, C = comp, cmap = plt.cm.YlGnBu)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Completeness')
    ax.set_xlabel(r'sSFR (yr$^{-1}$)')
    ax.set_ylabel(r'Eddington Ratio (log $\lambda_{\rm Edd}$)')
    ax.set_xlim(-10, -6.5)
    ax.set_ylim(-4, 0)
    im.set_clim(vmin =0,vmax=1)
    fig.savefig('../hex_plots/hex_diff_tot_ssfr')
    plt.close(fig)

    fig, ax = plt.subplots()
    im = ax.hexbin(mass, lum, gridsize = 6, C = comp, cmap = plt.cm.YlGnBu)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Completeness')
    ax.set_xlabel(r'Mass (log(M$_*$/M$_{\odot}$)')
    ax.set_ylabel(r'Eddington Ratio (log $\lambda_{\rm Edd}$)')
    ax.set_xlim(8.5, 12)
    ax.set_ylim(-4, 0)
    im.set_clim(vmin =0,vmax=1)
    fig.savefig('../hex_plots/hex_diff_tot_mass')
    plt.close(fig)

    fig, ax = plt.subplots()
    im = ax.hexbin(mass, ssfr, gridsize = 6, C = comp, cmap = plt.cm.YlGnBu)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Completeness')
    ax.set_xlabel(r'Mass (log(M$_*$/M$_{\odot}$)')
    ax.set_ylabel(r'sSFR (yr$^{-1}$)')
    ax.set_xlim(8.5, 12)
    ax.set_ylim(-10, -6.5)
    #ax.set_xlim(8.5, 11.5)
    #ax.set_ylim(-9.5, -6.5)
    im.set_clim(vmin =0,vmax=1)
    fig.savefig('../hex_plots/hex_diff_tot_mass_ssfr')
    plt.close(fig)
    
    return 


def perc_gain():

    mass, ssfr, lum, comp = np.loadtxt('../completeness.dat', unpack=True, usecols=(2,3,4,5))
    mass_tot, ssfr_tot, lum_tot, comp_tot = np.loadtxt('../completeness_total.dat', unpack=True, usecols=(2,3,4,5))

    gain = comp - comp_tot
    
    fig, ax = plt.subplots()
    im = ax.hexbin(ssfr, lum, gridsize = 6, C = gain, cmap = plt.cm.YlGnBu)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Fractional Gain')
    ax.set_xlabel(r'sSFR (yr$^{-1}$)')
    ax.set_ylabel(r'Eddington Ratio ($\lambda_{Edd}$)')
    ax.set_xlim(-10, -6.5)
    ax.set_ylim(-4, 0)
    im.set_clim(vmin =0,vmax=1)
    fig.savefig('../hex_plots/gain_ssfr')
    plt.close(fig)

    fig, ax = plt.subplots()
    im = ax.hexbin(mass, lum, gridsize = 6, C = gain, cmap = plt.cm.YlGnBu)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Fractional Gain')
    ax.set_xlabel(r'Mass (log(M$_*$/M$_{sol}$)')
    ax.set_ylabel(r'Eddington Ratio ($\lambda_{Edd}$)')
    ax.set_xlim(8.5, 12)
    ax.set_ylim(-4, 0)
    im.set_clim(vmin =0, vmax=1)
    fig.savefig('../hex_plots/gain_mass')
    plt.close(fig)

    fig, ax = plt.subplots()
    im = ax.hexbin(mass, ssfr, gridsize = 6, C = gain, cmap = plt.cm.YlGnBu)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Fractional Gain')
    ax.set_xlabel(r'Mass (log(M$_*$/M$_{sol}$)')
    ax.set_ylabel(r'sSFR (yr$^{-1}$)')
    ax.set_xlim(8.5, 12)
    ax.set_ylim(-10, -6.5)
    im.set_clim(vmin =0,vmax=1)
    fig.savefig('../hex_plots/gain_mass_ssfr')
    plt.close(fig)

    return


def line_fit():

    files = glob('../collapse2d/*spectrum.dat')
    x = open('../integrated.dat', 'w')
    x.write('#     ID            Hbeta           OIII5007        ratio            err\n')

    for file in files:

        wave, flux, err = np.loadtxt(file, unpack=True)
        flux = flux[np.logical_and(wave > 4000, wave < 5700)]
        wave = wave[np.logical_and(wave > 4000, wave < 5700)]
        err = err[np.logical_and(wave > 4000, wave < 5700)]
    
        flux_new = flux
        r = 0
        while r < 10:

            sort = sorted(flux_new)
            size = np.size(flux_new)
            bad = np.logical_and(flux_new > sort[int(0.15*size)], (flux_new < sort[int(0.85*size)]))
            fit = np.poly1d(np.polyfit(np.array(wave)[bad], np.array(flux_new)[bad], 2))
            flux_new = flux_new - fit(wave)
            r += 1
        flux = flux_new

        global N
        N = 1
        global line
        line = np.array([4861, 4959, 5007])

        name = file.strip('../collapse2d/')
        name = name.strip('_spectrum.dat')
        
        p0 = init(line, flux)
        try:
            coeff, var_mat = scipy.optimize.curve_fit(func, wave, flux, p0 = p0, sigma=err, absolute_sigma=True, maxfev=10000)
        except RuntimeError:
            print name
            x.write(name+'  \n')
            continue
        mean_Hb = coeff[N+1]
        mean_OIII5007 = coeff[N+2]
        mean_OIII4959 = coeff[N+2]/3
        sigma = coeff[-1]
        err = np.sqrt(np.diag(var_mat))
        err_Hb = err[N+1]
        err_OIII5007 = err[N+2]
    
        fit_Hb = gaussian(wave, mean_Hb, line[0], sigma)
        fit_OIII4959 = gaussian(wave, mean_OIII4959, line[1], sigma)
        fit_OIII5007 = gaussian(wave, mean_OIII5007, line[2], sigma)
        area_Hb = (np.trapz(fit_Hb, wave))
        area_OIII4959 = (np.trapz(fit_OIII4959, wave))
        area_OIII5007 = (np.trapz(fit_OIII5007, wave))
        fit_result = func(wave, *coeff)

        err_ratio = (area_OIII5007/area_Hb)*np.sqrt((err_Hb/area_Hb)**2 + (err_OIII5007/area_OIII5007)**2)
        
        #print 'Output flux:'
        #print 'Hbeta:', area_Hb
        #print 'OIII4959:', area_OIII4959
        #print 'OIII5007:', area_OIII5007

        x.write(name+'  ')
        x.write(str(area_Hb)+'   ')
        #x.write(str(area_OIII4959)+'   ')
        x.write(str(area_OIII5007)+'   ')
        x.write(str(area_OIII5007/area_Hb)+'   ')
        x.write(str(err_ratio)+'\n')

        plt.step(wave, flux, 'b', lw = 1.5)
        plt.plot(wave, fit_result, 'r-', linewidth = 2.5)
        plt.plot(wave, fit_OIII4959+np.median(flux), 'g-')
        plt.plot(wave, fit_Hb+np.median(flux), 'g-')
        plt.plot(wave, fit_OIII5007+np.median(flux), 'g-')
        plt.xlim(4300, 5500)
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Flux (erg/cm$^2$/sec/A) x 10$^{-16}$')
        plt.minorticks_on()
        plt.savefig('../1Dspectra_retry/'+name+'_fit.pdf', format='pdf')
        plt.close()

    x.close()

    return


def plot_correlation():

    outratio, err = np.loadtxt('../integrated.dat', unpack=True, usecols=(3,4))
    blah, inratio = np.loadtxt('../noAGNratio.dat', unpack=True, usecols=(1,3))

    plt.plot(inratio, outratio, 'bo')
    plt.xlabel(r'Input [O III]/H$\beta$')
    plt.ylabel(r'Output [O III]/H$\beta$')
    plt.savefig('../input_output')
    plt.close()

    plt.plot(inratio, (outratio-inratio)/err, 'bo')
    plt.xlabel(r'Input [O III]/H$\beta$')
    plt.ylabel('Residuals')
    plt.savefig('../input_residuals')
    plt.close()

    return


def func(x, *p):

    n = N

    poly = 0 * x
    for i in xrange(n + 1):
        poly = poly + p[i] * x**i

    gauss1 = p[1+n]*np.exp(-(x-line[0])**2/(2*(p[-1]**2)))      # Hbeta
    gauss2 = (p[2+n]/3)*np.exp(-(x-line[1])**2/(2*(p[-1]**2)))  # OIII4959
    gauss3 = (p[2+n])*np.exp(-(x-line[2])**2/(2*(p[-1]**2)))    # OIII5007

    return gauss1 + gauss2 + gauss3 + poly


def init(line, flux):
    
    n = N
    p0 = np.zeros(shape = (n + len(line) + 1))
    p0[0] = np.mean(flux)
    p0[-1] = 20
    
    return p0


def distribution(mean, err):
    
    return np.random.normal(mean, err, 1000)


def MeX():

    mass, ssfr, lum = np.loadtxt('../int_medians.dat', unpack=True, usecols=(2,3,4))
    tot_ratio, err = np.loadtxt('../int_medians.dat', unpack=True, usecols=(5,6))
    #err = err/(tot_ratio*np.log(10))
    #in_rat, out_rat = np.loadtxt('../medians.dat', unpack=True, usecols=(5,7))

    m1 = np.arange(7, 10, 0.01)
    m2 = np.arange(10, 14, 0.01)
    m3 = np.arange(7, 9.6, 0.01)
    m4 = np.arange(9.6, 14, 0.01)
    y1 = 0.375/(m1-10.5)+1.14
    y2 = 410.24 - 109.333*m2 + 9.71731*m2**2 - 0.288244*m2**3
    y3 = 0.375/(m3-10.5)+1.14
    y4 = 352.066 - 93.8249*m4 + 8.32651*m4**2 - 0.246416*m4**3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(m1, y1, 'k-', lw=3)
    ax.plot(m2, y2, 'k-', lw=3)
    ax.plot(m3, y3, 'k-', lw=3)
    ax.plot(m4, y4, 'k-', lw=3)
    good, bad, good_s, bad_s = [],[],[],[]
    good_m, bad_m = [],[]
    good_l, bad_l = [],[]
    for j, e in enumerate(err):
        if tot_ratio[j] >=e:
            good.append(tot_ratio[j])
            good_s.append(ssfr[j])
            good_m.append(mass[j])
            good_l.append(lum[j])
        if tot_ratio[j] < e:
            bad.append(tot_ratio[j])
            bad_s.append(ssfr[j])
            bad_m.append(mass[j])
            bad_l.append(lum[j])
    for i, l in enumerate(good_l):
        if l == -0.5:
            im = ax.scatter(good_m[i]-0.135, np.log10(good[i]), c = good_s[i], cmap = plt.cm.jet, s = 50, vmin=-9.5, vmax = -7)
        if l == -1.0 or l == -1.5:
            im = ax.scatter(good_m[i]-0.045, np.log10(good[i]), c = good_s[i], cmap = plt.cm.jet, s = 50, vmin=-9.5, vmax = -7)
        if l == -2.0 or l == -2.5:
            im = ax.scatter(good_m[i]+0.045, np.log10(good[i]), c = good_s[i], cmap = plt.cm.jet, s = 50, vmin=-9.5, vmax = -7)
        if l == -3.0 or l == -3.5:
            im = ax.scatter(good_m[i]+0.135, np.log10(good[i]), c = good_s[i], cmap = plt.cm.jet, s = 50, vmin=-9.5, vmax = -7)
    for i, l in enumerate(bad_l):
        if l == -0.5:
            im = ax.scatter(bad_m[i]-0.135, np.log10(bad[i]), c = bad_s[i], cmap = plt.cm.jet, s = 50, vmin=-9.5, vmax = -7, facecolors='None')
        if l == -1.0 or l == -1.5:
            im = ax.scatter(bad_m[i]-0.045, np.log10(bad[i]), c = bad_s[i], cmap = plt.cm.jet, s = 50, vmin=-9.5, vmax = -7, facecolors='None')
        if l == -2.0 or l == -2.5:
            im = ax.scatter(bad_m[i]+0.045, np.log10(bad[i]), c = bad_s[i], cmap = plt.cm.jet, s = 50, vmin=-9.5, vmax = -7, facecolors='None')
        if l == -3.0 or l == -3.5:
            im = ax.scatter(bad_m[i]+0.135, np.log10(bad[i]), c = bad_s[i], cmap = plt.cm.jet, s = 50, vmin=-9.5, vmax = -7, facecolors='None')
            
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'sSFR (yr$^{-1}$)')
    #plt.plot(mass+0.09, np.log10(in_rat), 'ro', ms=8)
    #plt.plot(mass+0.18, np.log10(out_rat), 'bo', ms=8)
    #plt.plot(mass+0.27, np.log10(in_rat-out_rat), 'ko', ms=8)
    ax.set_xlim(8.5, 12)
    ax.set_ylim(-0.5, 1.3)
    #plt.plot([11.75, 11.75], [-1.2-avg_err, -1.2+avg_err], 'k-', lw=2)
    ax.set_xlabel(r'Mass [log(M$_*$/M$_{\odot}$)]')
    ax.set_ylabel(r'log([O III]/H$\beta$)$_{\rm total}$')
    plt.savefig('../other_plots/MEX_tot')
    plt.close(fig)

    return


def check_intlum():

    z = 1.8
    mass = [9.5, 10, 10.5, 11, 11.5]
    sSFR = [-9, -8.5, -8, -7.5, -7]
    mass_str = ['9.5', '10.0', '10.5', '11.0', '11.5']

    f = open('noAGNratio.dat', 'w')
    f.write('#        ID               Hbeta    OIII     ratio\n')

    for i, m in enumerate(mass):
        for s in sSFR:
            cont_mag = mass2mag(m)                              # Convert masses to continuum mags
            EW_disk = ssfr2EWgal(s, m, cont_mag, z)             # Find star formation line EW
            r_disk, met = SFRandMtoR(s, m)

            edit_onespec_test(z, cont_mag)                       # Edit the onespec file for running aXeSIM
                    
            Hbeta, OIII = spectrum_disk('disk.dat', cont_mag, EW_disk, r_disk) # Make the disk spectrum
            print Hbeta, OIII

            file_ext = '1.8z_'+mass_str[i]+'M_'+str(s)+'sSFR'

            #os.mkdir('../FITS/'+file_ext)
            
            f.write(file_ext+'   ')
            f.write(str(Hbeta)+'   ')
            f.write(str(OIII)+'   ')
            f.write(str(OIII/Hbeta)+'\n')

            #axesim_wfc3_new.axe(file_ext)                     # Run aXeSIM
            #interp(file_ext, file_ext)                        # Interpolate the resulting spectrum
            #edit_testlist(file_ext, file_ext, z)              # Edit testlist.dat for to use for Jon's code
    f.close()

    return
            

def edit_onespec_noAGN(z, cont_mag, kpc, pix):      # spectemp is the line number of the fake galaxy in input_spectra.lis

    # read in the one_spec file
    fname = './save/one_spec_G141.lis'
    f = open(fname, 'w')
    f.write('# 1  NUMBER\n# 2  X_IMAGE\n# 3  Y_IMAGE\n# 4  A_IMAGE\n# 5  B_IMAGE\n# 6  THETA_IMAGE\n# 7  MAG_F475W\n# 8  Z\n# 9  SPECTEMP\n# 10 MODIMAGE\n')

    x = [0, 200, 400, 600, 800]
    y = [20, 60, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500, 540, 580, 620, 660, 700, 740, 780]
    
    i=0
    n=0
    for k in x:
        for u, l in enumerate(y):
            
            blah = np.arange(0.5, 10, 0.1)
            ind = [j for j, a in enumerate(blah) if "{0:.1f}".format(a) == "{0:.1f}".format(kpc[n])]
            if not ind:
                ind = [0]
                pix[n] = 0.5
            
            f.write(str(i+1))
            f.write(' '+str(k)+' '+str(l))
            f.write(' '+"{0:.2f}".format(2*pix[n])+' '+"{0:.2f}".format(2*pix[n]))
            f.write(' 90.0 ')
            f.write(str(cont_mag))
            f.write(' '+str(z)+' ')
            f.write('2 '+str(2+ind[0])+'\n')

            i += 2
            n += 1
    return 


def threeDplot():

    mass, ssfr, lum = np.loadtxt('../medians.dat', unpack=True, usecols=(2,3,4))
    #tot_ratio, err = np.loadtxt('../int_medians.dat', unpack=True, usecols=(1,2))
    in_rat, out_rat = np.loadtxt('../medians.dat', unpack=True, usecols=(5,7))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(mass, ssfr, lum, c = np.log10(in_rat/out_rat), cmap = plt.cm.jet, s = 50, vmin=-0.15, vmax = 0.35)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'$\Delta$log([O III]/H$\beta$)')
    ax.set_xlabel(r'Mass [log(M$_*$/M$_{\odot}$)]')
    ax.set_ylabel(r'sSFR (yr$^{-1}$)')
    ax.set_zlabel(r'Eddington Ratio ($\lambda_{Edd}$)')
    ax.yaxis._axinfo['label']['space_factor'] = 2
    ax.xaxis._axinfo['label']['space_factor'] = 2
    ax.zaxis._axinfo['label']['space_factor'] = 2
    ax.elev = 25
    ax.azim = -70
    

    fig.savefig('../other_plots/scatter3D')
    plt.close(fig)

    return


def multi_fit_mcmc():

    mass, ssfr, lum, in_ratio, in_err, out_ratio, out_err = np.loadtxt('../medians.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    ID = np.genfromtxt('../medians.dat', unpack=True, usecols=(0), dtype='str')
    print mass
    print ssfr

    x = np.array(zip(mass-9, ssfr+10, np.log10(in_ratio/out_ratio)))
    y = lum
    yerr = np.array(len(mass))
    yerr.fill(0.1)
    merr = yerr
    serr = yerr
    err = np.log10(in_ratio/out_ratio)*np.sqrt((in_err/in_ratio)**2 + (out_err/out_err)**2)
    rerr = err/(np.log10(in_ratio/out_ratio)*np.log(10))

    a, b, c, const, sigma = multi_regression_mcmc_fit.regression(x, merr, serr, rerr, y, yerr)

    return a, b, c, const, sigma



def comparison():

    mass, ssfr, lum, in_ratio, in_err, out_ratio, out_err = np.loadtxt('../medians.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    mass_d, ssfr_d, lum_d, in_ratio_d, in_err_d, out_ratio_d, out_err_d = np.loadtxt('../medians_devauc.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    mass_e, ssfr_e, lum_e, in_ratio_e, in_err_e, out_ratio_e, out_err_e = np.loadtxt('../medians_ell.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    mass_1, ssfr_1, lum_1, in_ratio_1, in_err_1, out_ratio_1, out_err_1 = np.loadtxt('../medians_1.4z.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    mass_2, ssfr_2, lum_2, in_ratio_2, in_err_2, out_ratio_2, out_err_2 = np.loadtxt('../medians_2.2z.dat', unpack=True, usecols=(2,3,4,5,6,7,8))

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(1,1,1)
    #ax2 = fig.add_subplot(2,1,2)
    #ax3 = fig.add_subplot(3,1,3,)
    
    l = lum == -.5
    l_d = lum_d == -.5
    l_e = lum_e == -.5
    l_1 = lum_1 == -.5
    l_2 = lum_2 == -.5

    mass = mass[l]
    mass_d = mass_d[l_d]
    mass_e = mass_e[l_e]
    mass_1 = mass_1[l_1]
    mass_2 = mass_2[l_2]
    
    #print mass_d
    #print mass_e
    #print mass_1
    #print mass_2

    in_ratio = in_ratio[l]
    in_ratio_d = in_ratio_d[l_d]
    in_ratio_e = in_ratio_e[l_e]
    in_ratio_1 = in_ratio_1[l_1]
    in_ratio_2 = in_ratio_2[l_2]
    out_ratio = out_ratio[l]
    out_ratio_d = out_ratio_d[l_d]
    out_ratio_e = out_ratio_e[l_e]
    out_ratio_1 = out_ratio_1[l_1]
    out_ratio_2 = out_ratio_2[l_2]
    in_err = in_err[l]
    in_err_d = in_err_d[l_d]
    in_err_e = in_err_e[l_e]
    in_err_1 = in_err_1[l_1]
    in_err_2 = in_err_2[l_2]
    out_err = out_err[l]
    out_err_d = out_err_d[l_d]
    out_err_e = out_err_e[l_e]
    out_err_1 = out_err_1[l_1]
    out_err_2 = out_err_2[l_2]
    
    # Plot const mass bins with delta ratio 
    mopt = [9.5, 10, 10.5, 11, 11.5]
    
    inr_constm = np.empty(len(mopt))
    inerr_constm = np.empty((len(mopt)))
    outr_constm = np.empty((len(mopt)))
    outerr_constm = np.empty((len(mopt)))
    inr_constm_d = np.empty((len(mopt)))
    inerr_constm_d = np.empty((len(mopt)))
    outr_constm_d = np.empty((len(mopt)))
    outerr_constm_d = np.empty((len(mopt)))
    inr_constm_e = np.empty((len(mopt)))
    inerr_constm_e = np.empty((len(mopt)))
    outr_constm_e = np.empty((len(mopt)))
    outerr_constm_e = np.empty((len(mopt)))
    inr_constm_1 = np.empty((len(mopt)))
    inerr_constm_1 = np.empty((len(mopt)))
    outr_constm_1 = np.empty((len(mopt)))
    outerr_constm_1 = np.empty((len(mopt)))
    inr_constm_2 = np.empty((len(mopt)))
    inerr_constm_2 = np.empty((len(mopt)))
    outr_constm_2 = np.empty((len(mopt)))
    outerr_constm_2 = np.empty((len(mopt)))
    
    for i, m in enumerate(mopt):

        inr_constm[i] = np.median(in_ratio[np.where(mass == m)])
        inerr_constm[i] = np.median(in_err[np.where(mass == m)])
        outr_constm[i] = np.median(out_ratio[np.where(mass == m)])
        outerr_constm[i] = np.median(out_err[np.where(mass == m)])
        inr_constm_d[i] = np.median(in_ratio_d[np.where(mass_d == m)])
        inerr_constm_d[i] = np.median(in_err_d[np.where(mass_d == m)])
        outr_constm_d[i] = np.median(out_ratio_d[np.where(mass_d == m)])
        outerr_constm_d[i] = np.median(out_err_d[np.where(mass_d == m)])
        inr_constm_e[i] = np.median(in_ratio_e[np.where(mass_e == m)])
        inerr_constm_e[i] = np.median(in_err_e[np.where(mass_e == m)])
        outr_constm_e[i] = np.median(out_ratio_e[np.where(mass_e == m)])
        outerr_constm_e[i] = np.median(out_err_e[np.where(mass_e == m)])
        inr_constm_1[i] = np.median(in_ratio_1[np.where(mass_1 == m)])
        inerr_constm_1[i] = np.median(in_err_1[np.where(mass_1 == m)])
        outr_constm_1[i] = np.median(out_ratio_1[np.where(mass_1 == m)])
        outerr_constm_1[i] = np.median(out_err_1[np.where(mass_1 == m)])
        inr_constm_2[i] = np.median(in_ratio_2[np.where(mass_2 == m)])
        inerr_constm_2[i] = np.median(in_err_2[np.where(mass_2 == m)])
        outr_constm_2[i] = np.median(out_ratio_2[np.where(mass_2 == m)])
        outerr_constm_2[i] = np.median(out_err_2[np.where(mass_2 == m)])

    r = np.log10(inr_constm/outr_constm)
    err_constm = r*np.sqrt((inerr_constm/inr_constm)**2 + (outerr_constm/outr_constm)**2)
    err_r = err_constm/(r*np.log(10))
    r_d = np.log10(inr_constm_d/outr_constm_d)
    err_constm_d = r_d*np.sqrt((inerr_constm_d/inr_constm_d)**2 + (outerr_constm_d/outr_constm_d)**2)
    err_r_d = err_constm_d/(r_d*np.log(10))
    r_e = np.log10(inr_constm_e/outr_constm_e)
    err_constm_e = r_e*np.sqrt((inerr_constm_e/inr_constm_e)**2 + (outerr_constm_e/outr_constm_e)**2)
    err_r_e = err_constm_e/(r_e*np.log(10))
    r_1 = np.log10(inr_constm_1/outr_constm_1)
    err_constm_1 = r_1*np.sqrt((inerr_constm_1/inr_constm_1)**2 + (outerr_constm_1/outr_constm_1)**2)
    err_r_1 = err_constm_1/(r_1*np.log(10))
    r_2 = np.log10(inr_constm_2/outr_constm_2)
    err_constm_2 = r_2*np.sqrt((inerr_constm_2/inr_constm_2)**2 + (outerr_constm_2/outr_constm_2)**2)
    err_r_2 = err_constm_2/(r_2*np.log(10))

    plt.hlines(0.1, 8, 12, linestyle = '--')
    a=ax1.errorbar(mopt, r, yerr = err_r, color='black', linewidth = 3, marker='o', label='z = 1.8')
    #d=ax1.errorbar(mopt, r_d, yerr = err_r_d, color= 'blue', lw = 3, marker='o', label = 'n = 4 profile')
    e=ax1.errorbar(mopt, r_e, yerr = err_r_e, color='blue', lw = 3, marker='o', label='e = 0.2 profile')
    b=ax1.errorbar(mopt, r_1,yerr=err_r_1, color='yellow', lw = 3, marker='o', label='z = 1.4')
    c=ax1.errorbar(mopt, r_2,yerr=err_r_1, color='red', lw = 3, marker='o', label='z = 2.2')
    ax1.set_ylim(0, 0.4)
    ax1.set_xlim(9.4, 11.6)
    ax1.set_xlabel(r'Mass (log(M$_*$/M$_{\odot}$)')
    ax1.set_ylabel(r'log([O III]/H$\beta$)$_{in}$ - log([O III]/H$\beta$)$_{out}$')
    ax1.legend([a,b,c,e], ['z = 1.8','z = 1.4','z = 2.2', 'e = 0.2 profile'], loc = 0, prop={'s\
ize':11})     
    #ax1.legend([a,b,c,d,e], ['z = 1.8','z = 1.4','z = 2.2', 'n = 4 profile','e = 0.2 profile'], loc = 0, prop={'size':11})

    plt.savefig('../other_plots/comparison_lum-0p5')
    plt.close(fig)

    mass, ssfr, lum, in_ratio, in_err, out_ratio, out_err = np.loadtxt('../medians.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    mass_d, ssfr_d, lum_d, in_ratio_d, in_err_d, out_ratio_d, out_err_d = np.loadtxt('../medians_devauc.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    mass_e, ssfr_e, lum_e, in_ratio_e, in_err_e, out_ratio_e, out_err_e = np.loadtxt('../medians_ell.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    mass_1, ssfr_1, lum_1, in_ratio_1, in_err_1, out_ratio_1, out_err_1 = np.loadtxt('../medians_1.4z.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    mass_2, ssfr_2, lum_2, in_ratio_2, in_err_2, out_ratio_2, out_err_2 = np.loadtxt('../medians_2.2z.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(1,1,1)
    #ax2 = fig.add_subplot(2,1,2)
    #ax3 = fig.add_subplot(3,1,3,)
    
    m = mass > 11
    m_d = mass_d > 11
    m_e = mass_e > 11
    m_1 = mass_1 > 11
    m_2 = mass_2 > 11

    lum = lum[m]
    lum_d = lum_d[m_d]
    lum_e = lum_e[m_e]
    lum_1 = lum_1[m_1]
    lum_2 = lum_2[m_2]
    
    in_ratio = in_ratio[m]
    in_ratio_d = in_ratio_d[m_d]
    in_ratio_e = in_ratio_e[m_e]
    in_ratio_1 = in_ratio_1[m_1]
    in_ratio_2 = in_ratio_2[m_2]
    out_ratio = out_ratio[m]
    out_ratio_d = out_ratio_d[m_d]
    out_ratio_e = out_ratio_e[m_e]
    out_ratio_1 = out_ratio_1[m_1]
    out_ratio_2 = out_ratio_2[m_2]
    in_err = in_err[m]
    in_err_d = in_err_d[m_d]
    in_err_e = in_err_e[m_e]
    in_err_1 = in_err_1[m_1]
    in_err_2 = in_err_2[m_2]
    out_err = out_err[m]
    out_err_d = out_err_d[m_d]
    out_err_e = out_err_e[m_e]
    out_err_1 = out_err_1[m_1]
    out_err_2 = out_err_2[m_2]
    
    # Plot const mass bins with delta ratio 
    lopt = [-0.5, -1.5, -2.5, -3.5]
    
    inr_constm = np.empty(len(lopt))
    inerr_constm = np.empty((len(lopt)))
    outr_constm = np.empty((len(lopt)))
    outerr_constm = np.empty((len(lopt)))
    inr_constm_d = np.empty((len(lopt)))
    inerr_constm_d = np.empty((len(lopt)))
    outr_constm_d = np.empty((len(lopt)))
    outerr_constm_d = np.empty((len(lopt)))
    inr_constm_e = np.empty((len(lopt)))
    inerr_constm_e = np.empty((len(lopt)))
    outr_constm_e = np.empty((len(lopt)))
    outerr_constm_e = np.empty((len(lopt)))
    inr_constm_1 = np.empty((len(lopt)))
    inerr_constm_1 = np.empty((len(lopt)))
    outr_constm_1 = np.empty((len(lopt)))
    outerr_constm_1 = np.empty((len(lopt)))
    inr_constm_2 = np.empty((len(lopt)))
    inerr_constm_2 = np.empty((len(lopt)))
    outr_constm_2 = np.empty((len(lopt)))
    outerr_constm_2 = np.empty((len(lopt)))
    
    for i, l in enumerate(lopt):

        inr_constm[i] = np.median(in_ratio[np.where(lum == l)])
        inerr_constm[i] = np.median(in_err[np.where(lum == l)])
        outr_constm[i] = np.median(out_ratio[np.where(lum == l)])
        outerr_constm[i] = np.median(out_err[np.where(lum == l)])
        inr_constm_d[i] = np.median(in_ratio_d[np.where(lum_d == l)])
        inerr_constm_d[i] = np.median(in_err_d[np.where(lum_d == l)])
        outr_constm_d[i] = np.median(out_ratio_d[np.where(lum_d == l)])
        outerr_constm_d[i] = np.median(out_err_d[np.where(lum_d == l)])
        inr_constm_e[i] = np.median(in_ratio_e[np.where(lum_e == l)])
        inerr_constm_e[i] = np.median(in_err_e[np.where(lum_e == l)])
        outr_constm_e[i] = np.median(out_ratio_e[np.where(lum_e == l)])
        outerr_constm_e[i] = np.median(out_err_e[np.where(lum_e == l)])
        inr_constm_1[i] = np.median(in_ratio_1[np.where(lum_1 == l)])
        inerr_constm_1[i] = np.median(in_err_1[np.where(lum_1 == l)])
        outr_constm_1[i] = np.median(out_ratio_1[np.where(lum_1 == l)])
        outerr_constm_1[i] = np.median(out_err_1[np.where(lum_1 == l)])
        inr_constm_2[i] = np.median(in_ratio_2[np.where(lum_2 == l)])
        inerr_constm_2[i] = np.median(in_err_2[np.where(lum_2 == l)])
        outr_constm_2[i] = np.median(out_ratio_2[np.where(lum_2 == l)])
        outerr_constm_2[i] = np.median(out_err_2[np.where(lum_2 == l)])

    r = np.log10(inr_constm/outr_constm)
    err_constm = r*np.sqrt((inerr_constm/inr_constm)**2 + (outerr_constm/outr_constm)**2)
    err_r = err_constm/(r*np.log(10))
    r_d = np.log10(inr_constm_d/outr_constm_d)
    err_constm_d = r_d*np.sqrt((inerr_constm_d/inr_constm_d)**2 + (outerr_constm_d/outr_constm_d)**2)
    err_r_d = err_constm_d/(r_d*np.log(10))
    r_e = np.log10(inr_constm_e/outr_constm_e)
    err_constm_e = r_e*np.sqrt((inerr_constm_e/inr_constm_e)**2 + (outerr_constm_e/outr_constm_e)**2)
    err_r_e = err_constm_e/(r_e*np.log(10))
    r_1 = np.log10(inr_constm_1/outr_constm_1)
    err_constm_1 = r_1*np.sqrt((inerr_constm_1/inr_constm_1)**2 + (outerr_constm_1/outr_constm_1)**2)
    err_r_1 = err_constm_1/(r_1*np.log(10))
    r_2 = np.log10(inr_constm_2/outr_constm_2)
    err_constm_2 = r_2*np.sqrt((inerr_constm_2/inr_constm_2)**2 + (outerr_constm_2/outr_constm_2)**2)
    err_r_2 = err_constm_2/(r_2*np.log(10))

    plt.hlines(0.1, -4, 0, linestyle = '--')
    a=ax1.errorbar(lopt, r, yerr = err_r, color='black', linewidth = 3, marker='o', label='z = 1.8')
    #d=ax1.errorbar(lopt, r_d, yerr = err_r_d, color= 'blue', lw = 3, marker='o', label = 'n = 4 profile')
    e=ax1.errorbar(lopt, r_e, yerr = err_r_e, color='green', lw = 3, marker='o', label='e = 0.2 profile')
    b=ax1.errorbar(lopt, r_1,yerr=err_r_1, color='yellow', lw = 3, marker='o', label='z = 1.4')
    c=ax1.errorbar(lopt, r_2,yerr=err_r_1, color='red', lw = 3, marker='o', label='z = 2.2')
    #ax1.set_ylim(-0.05, 0.5)
    ax1.set_xlim(-3.6, -0.4)
    ax1.set_xlabel(r'Eddington Ratio (log $\lambda_{\rm Edd}$)')
    ax1.set_ylabel(r'log([O III]/H$\beta$)$_{in}$ - log([O III]/H$\beta$)$_{out}$')
    ax1.legend([a,b,c,e], ['z = 1.8','z = 1.4','z = 2.2', 'e = 0.2 profile'], loc = 0, prop={'s\
ize':11})     
    #ax1.legend([a,b,c,d,e], ['z = 1.8','z = 1.4','z = 2.2', 'n = 4 profile','e = 0.2 profile'], loc = 0, prop={'size':11})

    plt.savefig('../other_plots/comparison_lum')
    plt.close(fig)

    mass, ssfr, lum, in_ratio, in_err, out_ratio, out_err = np.loadtxt('../medians.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    mass_d, ssfr_d, lum_d, in_ratio_d, in_err_d, out_ratio_d, out_err_d = np.loadtxt('../medians_devauc.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    mass_e, ssfr_e, lum_e, in_ratio_e, in_err_e, out_ratio_e, out_err_e = np.loadtxt('../medians_ell.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    mass_1, ssfr_1, lum_1, in_ratio_1, in_err_1, out_ratio_1, out_err_1 = np.loadtxt('../medians_1.4z.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    mass_2, ssfr_2, lum_2, in_ratio_2, in_err_2, out_ratio_2, out_err_2 = np.loadtxt('../medians_2.2z.dat', unpack=True, usecols=(2,3,4,5,6,7,8))
    
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(1,1,1)
    #ax2 = fig.add_subplot(2,1,2)
    #ax3 = fig.add_subplot(3,1,3,)
    
    s = ssfr == -8.5
    s_d = ssfr_d == -8.5
    s_e = ssfr_e == -8.5
    s_1 = ssfr_1 == -8.5
    s_2 = ssfr_2 == -8.5

    mass = mass[s]
    mass_d = mass_d[s_d]
    mass_e = mass_e[s_e]
    mass_1 = mass_1[s_1]
    mass_2 = mass_2[s_2]
    
    in_ratio = in_ratio[s]
    in_ratio_d = in_ratio_d[s_d]
    in_ratio_e = in_ratio_e[s_e]
    in_ratio_1 = in_ratio_1[s_1]
    in_ratio_2 = in_ratio_2[s_2]
    out_ratio = out_ratio[s]
    out_ratio_d = out_ratio_d[s_d]
    out_ratio_e = out_ratio_e[s_e]
    out_ratio_1 = out_ratio_1[s_1]
    out_ratio_2 = out_ratio_2[s_2]
    in_err = in_err[s]
    in_err_d = in_err_d[s_d]
    in_err_e = in_err_e[s_e]
    in_err_1 = in_err_1[s_1]
    in_err_2 = in_err_2[s_2]
    out_err = out_err[s]
    out_err_d = out_err_d[s_d]
    out_err_e = out_err_e[s_e]
    out_err_1 = out_err_1[s_1]
    out_err_2 = out_err_2[s_2]
    
    # Plot const mass bins with delta ratio 
    mopt = [9.5, 10, 10.5, 11, 11.5]
    
    inr_constm = np.empty(len(mopt))
    inerr_constm = np.empty((len(mopt)))
    outr_constm = np.empty((len(mopt)))
    outerr_constm = np.empty((len(mopt)))
    inr_constm_d = np.empty((len(mopt)))
    inerr_constm_d = np.empty((len(mopt)))
    outr_constm_d = np.empty((len(mopt)))
    outerr_constm_d = np.empty((len(mopt)))
    inr_constm_e = np.empty((len(mopt)))
    inerr_constm_e = np.empty((len(mopt)))
    outr_constm_e = np.empty((len(mopt)))
    outerr_constm_e = np.empty((len(mopt)))
    inr_constm_1 = np.empty((len(mopt)))
    inerr_constm_1 = np.empty((len(mopt)))
    outr_constm_1 = np.empty((len(mopt)))
    outerr_constm_1 = np.empty((len(mopt)))
    inr_constm_2 = np.empty((len(mopt)))
    inerr_constm_2 = np.empty((len(mopt)))
    outr_constm_2 = np.empty((len(mopt)))
    outerr_constm_2 = np.empty((len(mopt)))
    
    for i, m in enumerate(mopt):

        inr_constm[i] = np.median(in_ratio[np.where(mass == m)])
        inerr_constm[i] = np.median(in_err[np.where(mass == m)])
        outr_constm[i] = np.median(out_ratio[np.where(mass == m)])
        outerr_constm[i] = np.median(out_err[np.where(mass == m)])
        inr_constm_d[i] = np.median(in_ratio_d[np.where(mass_d == m)])
        inerr_constm_d[i] = np.median(in_err_d[np.where(mass_d == m)])
        outr_constm_d[i] = np.median(out_ratio_d[np.where(mass_d == m)])
        outerr_constm_d[i] = np.median(out_err_d[np.where(mass_d == m)])
        inr_constm_e[i] = np.median(in_ratio_e[np.where(mass_e == m)])
        inerr_constm_e[i] = np.median(in_err_e[np.where(mass_e == m)])
        outr_constm_e[i] = np.median(out_ratio_e[np.where(mass_e == m)])
        outerr_constm_e[i] = np.median(out_err_e[np.where(mass_e == m)])
        inr_constm_1[i] = np.median(in_ratio_1[np.where(mass_1 == m)])
        inerr_constm_1[i] = np.median(in_err_1[np.where(mass_1 == m)])
        outr_constm_1[i] = np.median(out_ratio_1[np.where(mass_1 == m)])
        outerr_constm_1[i] = np.median(out_err_1[np.where(mass_1 == m)])
        inr_constm_2[i] = np.median(in_ratio_2[np.where(mass_2 == m)])
        inerr_constm_2[i] = np.median(in_err_2[np.where(mass_2 == m)])
        outr_constm_2[i] = np.median(out_ratio_2[np.where(mass_2 == m)])
        outerr_constm_2[i] = np.median(out_err_2[np.where(mass_2 == m)])

    r = np.log10(inr_constm/outr_constm)
    err_constm = r*np.sqrt((inerr_constm/inr_constm)**2 + (outerr_constm/outr_constm)**2)
    err_r = err_constm/(r*np.log(10))
    r_d = np.log10(inr_constm_d/outr_constm_d)
    err_constm_d = r_d*np.sqrt((inerr_constm_d/inr_constm_d)**2 + (outerr_constm_d/outr_constm_d)**2)
    err_r_d = err_constm_d/(r_d*np.log(10))
    r_e = np.log10(inr_constm_e/outr_constm_e)
    err_constm_e = r_e*np.sqrt((inerr_constm_e/inr_constm_e)**2 + (outerr_constm_e/outr_constm_e)**2)
    err_r_e = err_constm_e/(r_e*np.log(10))
    r_1 = np.log10(inr_constm_1/outr_constm_1)
    err_constm_1 = r_1*np.sqrt((inerr_constm_1/inr_constm_1)**2 + (outerr_constm_1/outr_constm_1)**2)
    err_r_1 = err_constm_1/(r_1*np.log(10))
    r_2 = np.log10(inr_constm_2/outr_constm_2)
    err_constm_2 = r_2*np.sqrt((inerr_constm_2/inr_constm_2)**2 + (outerr_constm_2/outr_constm_2)**2)
    err_r_2 = err_constm_2/(r_2*np.log(10))

    plt.hlines(0.1, 8, 12, linestyle = '--')
    a=ax1.errorbar(mopt, r, yerr = err_r, color='black', linewidth = 3, marker='o', label='z = 1.8')
    #d=ax1.errorbar(mopt, r_d, yerr = err_r_d, color= 'blue', lw = 3, marker='o', label = 'n = 4 profile')
    e=ax1.errorbar(mopt, r_e, yerr = err_r_e, color='blue', lw = 3, marker='o', label='e = 0.2 profile')
    b=ax1.errorbar(mopt, r_1,yerr=err_r_1, color='yellow', lw = 3, marker='o', label='z = 1.4')
    c=ax1.errorbar(mopt, r_2,yerr=err_r_1, color='red', lw = 3, marker='o', label='z = 2.2')
    ax1.set_ylim(0, 0.5)
    ax1.set_xlim(9.4, 11.6)
    ax1.set_ylabel(r'log([O III]/H$\beta$)$_{in}$ - log([O III]/H$\beta$)$_{out}$')
    ax1.legend([a,b,c,e], ['z = 1.8','z = 1.4','z = 2.2', 'e = 0.2 profile'], loc = 0, prop={'s\
ize':11})     
    #ax1.legend([a,b,c,d,e], ['z = 1.8','z = 1.4','z = 2.2', 'n = 4 profile','e = 0.2 profile'], loc = 0, prop={'size':11})
    ax1.set_xlabel(r'Mass (log(M$_*$/M$_{\odot}$)')
    plt.savefig('../other_plots/comparison_ssfr-8')
    plt.close(fig)

    return
