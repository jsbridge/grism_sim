;; Re-extract 1D grism spectra in nuclear / extended components along 
;; the cross-dispersion direction only.  Designed for use with outputs
;; of collapse2d.pro.
;;
;; Inputs
;;    infile: includes columns of ID, redshift, and 2D filename
;;            (including path).  Defaults to "testlist.dat"
;;
;; Outputs
;;    new 1D spectrum files in the directory defined by out1ddir.
;;    Outputs have names like "id_inn.dat" and "id_out.dat", for inner
;;    and outer extractions.
;; 
;; The inner and outer extraction regions are defined by the variables
;; "inner" & "outer".
;; 
;; --------------------------------------------------------------------

pro radialextract, objnames, plot=plot

specpath = 'collapse2d_no'
out1ddir = 'extract1dnew_no'

inner = 3     ;Default: inner region is 3 pixels wide (-1,0,1)
outer1= 3    ; outer is pixels 3-6 (and also -3 to -6)
outer2 = 6    ;For 3D-HST resolution, 1 pix = 0.064" ~ 0.6 kpc

if not(keyword_set(infile)) then $
   infile = 'testlist_no.dat'

readcol,infile, id,zz,specfiles, format='a,f,a'
sens = mrdfits('WFC3.IR.G141.1st.sens.2.fits', 1, senshdr, /silent)
wavele = sens.wavelength
sensit = sens.sensitivity

for ii=0, n_elements(id)-1 do begin

 ;Read in 2D file
   spec2dfile = specpath + '/' + id[ii] + '_collapse2d.fits'
   null = mrdfits(specfiles[ii],0,hdr,/silent)
   fl2d = mrdfits(spec2dfile,1,fhdr,/silent)
   err2d = mrdfits(spec2dfile,2,ehdr,/silent)

   ivar = 1/err2d^2
   ivar[where(err2d eq 0)] = 1/(10*max(err2d))^2

   ypix = n_elements(fl2d[0,*])
   center = round(ypix/2)-2
   ;print, center
   wl = mrdfits(spec2dfile,3,whdr,/silent)

   wave = interpol(wavele/(1d + zz[ii]), 7751)
   ind = intarr(n_elements(wl))
   for i = 0, n_elements(wl)-1 do begin
         a = findel(wave, wl[i])
         if a ne a + 2 then begin
            ind[i] = a
         endif
   endfor
   
   sens = sensit[ind]
   
   fl2d_new = dblarr((size(fl2d))[1], (size(fl2d))[2])
   err2d_new = dblarr((size(err2d))[1], (size(err2d))[2])
   for i = 0, (size(fl2d))[2]-1 do begin
      fl2d_new[*,i] = fl2d[*,i]/sens
      err2d_new[*,i] = err2d[*,i]/sens
   endfor

   err2d = err2d_new
   fl2d = fl2d_new

   fracinn = fltarr(ypix)
   fracinn[*] = 0
   fracinn[center-(inner-1)/2:center+(inner-1)/2] = 1
   fracout = fltarr(ypix)
   fracout[*] = 0
   fracout[center+outer1:center+outer2] = 1
   fracout[center-outer2:center-outer1] = 1

   fl1dinn = fltarr(n_elements(fl2d[*,center]))
   fl1dout = fltarr(n_elements(fl2d[*,center]))
   openw,1,out1ddir+'/'+id[ii]+'_rinn.dat'
   openw,2,out1ddir+'/'+id[ii]+'_rout.dat'
   openw,3,out1ddir+'/'+id[ii]+'_rinn_err.dat'
   openw,4,out1ddir+'/'+id[ii]+'_rout_err.dat'

   for ww=0, n_elements(fl1dinn)-1 do begin
      fl1dinn[ww] = total(fracinn * fl2d[ww,*] * ivar[ww,*]) $
                    / (total(fracinn * ivar[ww,*]) / total(fracinn))
      fl1dout[ww] = total(fracout * fl2d[ww,*] * ivar[ww,*]) $
                    / (total(fracout * ivar[ww,*]) / total(fracout))
      printf,1,wl[ww],fl1dinn[ww]
      printf,2,wl[ww],fl1dout[ww]
      printf,3,wl[ww],sqrt(total(fracinn*err2d[ww,*]^2) / total(fracinn))
      printf,4,wl[ww],sqrt(total(fracout*err2d[ww,*]^2) / total(fracout))

   endfor
   
   close,1,2,3,4
   angstrom = '!3' + STRING(197B) + '!X'
   if keyword_set(plot) then begin
      loadct, 38
      ;openpps, out1ddir+'/'+id[ii]
      openpps, 'weird/'+id[ii]
      plot,wl,fl1dout*1d16,psym=10, xstyle=1, $
               xtitle=textoidl('Wavelength ('+angstrom+')'), ytitle=textoidl('Flux (10^{-16} erg/s/cm^{2}/'+angstrom+')'), $
               yrange=[0,max(fl1dinn[where(finite(fl1dinn))]*1d16)>max(fl1dout[where(finite(fl1dout))]*1d16)] , xrange=[4600,5300]
               ;yrange = [0, 50], xrange = [4600, 5300]
      oplot, wl, fl1dinn*1d16, psym = 10, color = 10
      oplot,wl,fl1dout*1d16, color=90, psym=10
      legend, ['Nuclear', 'Extended'], colors = [10, 90], psym = [0,0]
      closepps
   endif

endfor



end
