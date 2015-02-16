###########################################################################################
## This script and all it's functions should run all the simulations through aXeSIM      ##
##                                                                                       ##
## Joanna Bridge, 2/2015                                                                 ##
##                                                                                       ##
## Notes: 1.) All runs of aXeSIM use the 'exponential_disk.fits' and the 1400nm PSF      ##
##            fits files, i.e., no need to use mkobjects code                            ##
##        2.) Assumes a set AGN line ratio of [O III]/Hbeta = 0.6                        ##
##                                                                                       ##
###########################################################################################

import numpy as np
import axesim_wfc3_new
import scipy.interpolate
import pyfits
import matplotlib.pyplot as plt
import os
from astropy.cosmology import FlatLambdaCDM
from glob import glob


def simulate():

    redshift = [1.8]                            # Mid-grism for now, low and hi z allowed by G141 grism later, maybe
    mass = [8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5] # log solar masses
    sSFR = [-9.3, -9, -8.7, -8.4, -8.1, -7.8]   # log yr^-1
    logLbolLedd = [-3.5, -2.5, -1.5, -0.5]      # erg/s
    r_AGN = 0.6                                 # Fixed logOIII/Hb for AGN
    
    for z in redshift:
        for m in mass:
            for s in sSFR:
                for t in logLbolLedd:
                    cont_mag = mass2mag(m)                            # Convert masses to continuum mags
                    logLHbMstar = Lbol2Hbetalum(m, t)
                    EW_disk = ssfr2EWgal(s, m, cont_mag, z)
                    EW_AGN = LHbetaM2EWagn(logLHbMstar, m, cont_mag, z)
                    edit_onespec(z, cont_mag)                         # Edit the onespec file for running aXeSIM
                    r_disk = SFRandMtoR(s, m)
                    
                    spectrum('disk.dat', cont_mag, EW_disk, r_disk)   # Make the disk spectrum
                    spectrum('AGN.dat', 99, EW_AGN, r_AGN)            # Make the AGN spectrum

                    r = str(round(r_disk,3))
                    file_ext = str(z)+'z_'+str(m)+'M_'+str(s)+'sSFR_'+str(t)+'logLfrac'   
                    print file_ext                                    # Naming scheme, probably shit
                    axesim_wfc3_new.axe(file_ext)                     # Run aXeSIM
                    interp(file_ext)                                  # Interpolate the resulting spectrum
                    edit_testlist(file_ext, z)                        # Edit testlist.dat for to use for Jon's code
                    
    return


def interp(file):   #STP fits file from aXeSIM
    
    if os.path.exists('driz_'+file) == True:     # If the file exists already, delete it
        os.remove('driz_'+file)
        print 'Deleted old driz_'+file+'_slitless_2.STP.fits first'
        print 'Deleted old driz_'+file+'.fits first'
    
    p = pyfits.open('OUTSIM/'+file+'_slitless_2.STP.fits')
    
    d = p[1].data
    for line in d:
        line[0] = 0.001
        line[-1] = 0.001

    x = np.arange(0, d.shape[1])
    y = np.arange(0, d.shape[0])
    f = scipy.interpolate.interp2d(x, y, d, kind = 'cubic')
    xnew = np.arange(0, d.shape[1]*2)/2.
    ynew = np.arange(0, d.shape[0]*2)/2.

    dnew = f(xnew, ynew)
    pyfits.writeto('../FITS/driz_'+file+'.fits', (), header=p[0].header)
    pyfits.append('../FITS/driz_'+file+'.fits', dnew)

    # Need to grab error from big slitless fits file and make it right size for stamp
    p = pyfits.open('OUTSIM/'+file+'_slitless.fits')
    
    err = p[2].data
    e = err[490:513, 513:696]
    x = np.arange(0, e.shape[1])
    y = np.arange(0, e.shape[0])
    f = scipy.interpolate.interp2d(x, y, e, kind = 'cubic')
    xnew = np.arange(0, e.shape[1]*2)/2.
    ynew = np.arange(0, e.shape[0]*2)/2.
    enew = f(xnew, ynew)
    pyfits.append('../FITS/driz_'+file+'.fits', enew)

    return


def edit_testlist(file_ext, z):

    # read in the one_spec file
    fname = '../testlist.dat'
    f = open(fname, 'a')
    f.write(file_ext+' '+str(z)+' FITS/driz_'+file_ext+'.fits\n')
    
    f.close()

    return


def Lbol2Hbetalum(mass, logLbolLedd):

    # logLbol/Ledd = logLbol - logMbh - 38.1
    # (Lbol/10^40) = 112(LOIII/10^40)^1.2
    # logLbol = 1.2LogOIII - 5.95
    # logMbh = logM*-3
    # ...math...
    # logLOIII/M* = (5/6)logLbol/Ledd - (1/6)logM* + 34.2
    # lobHb/M* = logLOIII/M* + 0.6

    logLhbMstar = (5*logLbolLedd/6.) - (mass/6.) + 34.8

    return logLhbMstar


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


def LHbetaM2EWagn(thing, mass, cont_mag, z):       # Thing = log LHbeta/M*

    LHbeta = 10**(thing + mass)
    cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)
    d = (cosmo.luminosity_distance(z)).value       # Mpc, must change to cm
    d = d * 3.0857e24
    lineflux = LHbeta/(4 * np.pi * d**2)
    lineflux = lineflux/2.97                       
    cont = 10**(-(cont_mag-23.9)/2.5) * 10**(-29)  # This is in microJanskys, the 10^-29 puts in ergs/s/cm^2/Hz
    cont = cont * (3e18)/(4830**2)                 # Multiply by c/lambda^2 where wavelength is in Angtsroms, then ergs/s/cm^2/A
    EW = lineflux/cont

    return EW


def gaussian(x,a,x0,sigma):
    
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def spectrum(filename, cont_mag, EW, ratio):    # Makes the trio o' Gaussians

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

    return wave, Hbeta, OIII4959, OIII5007, cont


def edit_onespec(z, cont_mag):      # spectemp is the line number of the fake galaxy in input_spectra.lis
    
    # read in the one_spec file
    fname = './save/one_spec_G141.lis'
    f = open(fname, 'r')
    lines = f.readlines()
    
    # replace the data with what I choose
    line = lines[-1].split(' ')
    line[6] = str(cont_mag)
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

    #From Mannucci+ 2011, use SFR and mass to get 12+log(O/H)
    s = sSFR+mass
    mu = mass - 0.32*s
    m = mass - 10
    if mu >= 9.5:
        met = 8.9 + 0.37*m - 0.14*s - 0.19*m**2 + 0.12*m*s - 0.054*s**2
    if mu < 9.5:
        met = 8.93 + 0.51*(mu - 10)

    # From Maiolino+ 2008
    # x = 12 + logO/H - 8.69
    c0 = 0.1549
    c1 = -1.5031
    c2 = -0.9790
    c3 = -0.0297
    c4 = 0  
    x = met - 8.69
    logOIIIHb = c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4
    
    return logOIIIHb


def gradient():

    files = glob('../extract1dnew/*_rout.dat')
    x = open('../gradients.dat', 'w')
    x.write('#             ID                    in_ratio          out_ratio\n')

    for f in files:

        file_ext = f.rstrip('_rout.dat')
    
        wave_in, flux_in = np.loadtxt(file_ext+'_rinn.dat', unpack=True)
        wave_in, err_in = np.loadtxt(file_ext+'_rinn_err.dat', unpack=True)
        wave_out, flux_out = np.loadtxt(file_ext+'_rout.dat', unpack=True)
        wave_out, err_out = np.loadtxt(file_ext+'_rout_err.dat', unpack=True)

        flux_new_in, flux_new_out = flux_in, flux_out
        r = 0
        while r < 6:

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
        #plt.show()
    
        lines = [4861, 4959, 5007]

        ind = []
        for l in lines:
            ind.append((np.abs(wave_in-l)).argmin())
        
        Hbeta_in = flux_in[ind[0]]
        Hbeta_out = flux_out[ind[0]]
        OIII4_in = flux_in[ind[1]]
        OIII4_out = flux_out[ind[1]]
        OIII5_in = flux_in[ind[2]]
        OIII5_out = flux_out[ind[2]]

        ratio_in = OIII5_in/Hbeta_in
        ratio_out = OIII5_out/Hbeta_out

        x.write(str(1)+'.'+file_ext.lstrip('../extract1dnew'))
        x.write('    '+str(ratio_in))
        x.write('    '+str(ratio_out)+'\n') 

    x.close()
    
    return 
