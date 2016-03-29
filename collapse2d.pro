;; Collapse absorption & emission lines in 2D grism spectra into the
;; cross-dispersion direction.  This assumes that the size of a line
;; in the dispersion direction is purely spatial, rather than a
;; velocity width.  The outputs should be appropriate for radial
;; extractions with the program radialextract.pro.
;;
;; Inputs
;;    infile: includes columns of ID, redshift, and 2D filename
;;            (including path).  Defaults to "testlist.dat"
;;    outdir: output directory (default "collapse2d")
;;
;; Outputs
;;    new 2D spectrum files, with similar names as the input list
;;    (ending in "_collapse.fits") in the outdir directory
;;
;; Details

;; 1) Collapses radially onto the y (cross-dispersion) axis, assigning
;;    fractional pixels based on pixel radius.  The center in y is
;;    defined as a single row, with half collapsed below and half
;;    collapsed above.
;; 2) Defines a galaxy's size as r=0.5" (r~5 kpc at 1<z<2). This is
;;    small enough to avoid the worst of the Hb/O3 blending (the two
;;    lines are 0.8" apart at z=1.86).
;; 3) Treats blended [OIII]4959+5007 a bit differently than other
;;    lines: collapses from left onto 4959, and from right only 5007,
;;    but doesn't touch 4959 < wl0 < 5007.  Also treats the [OII]
;;    and [SII] doublets and Ha+[NII] as single lines.
;; 
;; **************
;; Copyright 2015 Jonathan R. Trump, Penn State University
;;
;; This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
;;
;; This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details, available at <http://www.gnu.org/licenses/>.
;; **************
;; --------------------------------------------------------------------

pro collapse2d, infile, outdir=outdir

if not(keyword_set(outdir)) then $
   outdir = 'collapse2d_no'
if not(keyword_set(infile)) then $
   infile = 'testlist_no.dat' 


lines = [4861,4990]

readcol,infile, id,zz,specfiles, format='a,f,a'

for ii=0, n_elements(specfiles)-1 do begin
   
;   print, specfiles[ii]

   null = mrdfits(specfiles[ii],0,hdr,/silent)

   ;Read flux
   fl2d = mrdfits(specfiles[ii],1,fhdr,/silent)
   err2d = mrdfits(specfiles[ii],2,ehdr,/silent)
   ;err2d = fl2d * 0 + 0.1  ; Made up errors for now

   ypix = n_elements(fl2d[0,*])
   center = round(ypix/2.)-5
   ;print, center
   wld = 46.5/2
   ;wl = 9280 + findgen(n_elements(fl2d[*,0]))*wld - 15*wld     ;1.4z
   ;wl = 9215 + findgen(n_elements(fl2d[*,0]))*wld - 15*wld     ;1.8z 
   ;wl = 9150 + findgen(n_elements(fl2d[*,0]))*wld - 15*wld     ;2.2z
   ;wl = 9210 + findgen(n_elements(fl2d[*,0]))*wld - 15*wld     ;1.8z ell
   ;wl = 9220 + findgen(n_elements(fl2d[*,0]))*wld - 15*wld     ;1.8z devauc 
   wl = 9820 + findgen(n_elements(fl2d[*,0]))*wld - 15*wld     ;1.8z with simulate_one
   wl0 = wl/(1+zz[ii])

   err2d[where(err2d eq 0)] = 10*max(err2d)
   ivar = 1/err2d^2
   ivar[where(err2d lt median(err2d)/5)] = 1/(3*max(err2d))^2

   resolution = 0.064125  ;0.12825 ;sxpar(fhdr,'cdelt2')   ;read spatial resolution from header 
   size = 0.5  ;Set by Hb+O3 blend limit
   sizepix = round(size/resolution) < (center-1)
   dwl = wl0[1]-wl0[0]

   ;Subtract continuum before collapsing
   cont = fl2d * 0
   wlsize = 2*sizepix * dwl
   ;print, wlsize
   nolines = where((wl0 lt lines[0]-wlsize or wl0 gt lines[0]+wlsize) $
                   and (wl0 lt lines[1]-wlsize or wl0 gt lines[1]+wlsize) $
                   and (wl0 gt 4100 and wl0 lt 5800))
                 
   for yy=0, ypix-1 do begin
      ;cont[*,yy] = interpol(fl2d[nolines,yy],wl0[nolines],wl0)
      pfit = poly_fit(wl0[nolines],fl2d[nolines,yy],3)
      for pp=0, n_elements(pfit)-1 do $
         cont[*,yy] += pfit[pp]*wl0^pp
   endfor
      ;;cont[*,yy] = interpol(fl2d[nolines,yy],wl0[nolines],wl0)
      ;;cont[*,yy] = interpol(smooth(fl2d[nolines,yy],5),wl0[nolines],wl0)

   new2d = double(fl2d - cont)
   newerr = double(err2d)

   for ll=0, n_elements(lines)-1 do begin
      if lines[ll]-sizepix*dwl lt min(wl0) $   ;skip lines not in spectrum
         or lines[ll]+sizepix*dwl gt max(wl0) then continue
      linepix = where(abs(wl0-lines[ll]) lt dwl)
      if lines[ll] eq 4990 then begin   ;collapse blended O3 a bit differently
         linepix[0] = (where(abs(wl0-4959) lt dwl))[0]
         linepix[1] = (where(abs(wl0-5007) lt dwl))[1]
      endif

      for quadrant=0, 3 do begin   ;loop over four quadrants
         xtarg = (quadrant ge 2)
         xsign = (xtarg eq 0) ? -1 : 1
         ysign = (quadrant mod 2 eq 0) ? 1 : -1

       ;first, properly weight initial pixels in target column
         totalweight = replicate(0d,sizepix)
         for yy=1, sizepix do begin
            totalweight[yy-1] = ivar[linepix[xtarg],center+ysign*yy]
            new2d[linepix[xtarg],center+ysign*yy] *= totalweight[yy-1]
         endfor
         numpixels = replicate(1.0,sizepix)

       ;do y=0 seperately, put half above and half below
         frac = 0.5
         for xx=1, sizepix do begin
            weight = ivar[linepix[xtarg]+xsign*xx,center]
            new2d[linepix[xtarg],center+ysign*xx] += $
               weight * frac * new2d[linepix[xtarg]+xsign*xx,center]
            totalweight[xx-1] += frac * weight
            numpixels[xx-1] += frac
            if ysign eq -1 then new2d[linepix[xtarg]+xsign*xx,center] = 0
         endfor

       ;now go over grid in x, y
         for xx=1, sizepix do begin
            for yy=1, sizepix do begin

               rr = sqrt(xx^2+yy^2)
               rpixlo = round(rr)
               rpixhi = round(rr+1)
               fraclo = 1-abs(rr-rpixlo)
               frachi = 1-fraclo

               if rpixlo gt sizepix then continue
               if rpixhi gt sizepix then frachi=0

               weight = ivar[linepix[xtarg]+xsign*xx,center+ysign*yy]

               new2d[linepix[xtarg],center+ysign*rpixlo] += weight * fraclo $
                                           * new2d[linepix[xtarg]+xsign*xx,center+ysign*yy]
               totalweight[rpixlo-1] += fraclo*weight
               numpixels[rpixlo-1] += fraclo
               if frachi ne 0 then begin
                  new2d[linepix[xtarg],center+ysign*rpixhi] += weight * frachi $
                                           * new2d[linepix[xtarg]+xsign*xx,center+ysign*yy]
                  totalweight[rpixhi-1] += frachi*weight
                  numpixels[rpixhi-1] += frachi
               endif

               new2d[linepix[xtarg]+xsign*xx,center+ysign*yy] -= $
                  (fraclo*new2d[linepix[xtarg]+xsign*xx,center+ysign*yy] $
                   + frachi*new2d[linepix[xtarg]+xsign*xx,center+ysign*yy])

;;                print,xx,yy,rr
;;                print,rpixlo,fraclo,rpixhi,frachi
            endfor
         endfor

         for yy=1, sizepix do begin  ;normalize to total flux
            new2d[linepix[xtarg],center+ysign*yy] *= numpixels[yy-1] / totalweight[yy-1]
            newerr[linepix[xtarg],center+ysign*yy] = numpixels[yy-1] / sqrt(totalweight[yy-1])
         endfor
      endfor

   endfor

   new2d += cont

 ;;Test output by extracting 1D
   fl1dold = fltarr(n_elements(fl2d[*,0]))
   err1dold = fltarr(n_elements(fl2d[*,0]))
   new1d = fl1dold
   cont1d = fl1dold
   for xx=0, n_elements(new1d)-1 do begin
      fl1dold[xx] = total(fl2d[xx,8:ypix-9])
      err1dold[xx] = total(err2d[xx,8:ypix-9])
      new1d[xx] = total(new2d[xx,8:ypix-9])
      cont1d[xx] = total(cont[xx,8:ypix-9])
   endfor

;   djs_plot,wl0,fl1dold,psym=10, yrange=[-5,650], ystyle = 1
;   djs_oplot,wl0,new1d,color='red',psym=10
;   djs_oplot,wl0,cont1d,color='green'
;;;
;;; ;;Also test O3 EQW
;;;   o3_1 = (where(abs(wl0-4959) lt dwl))[0]
;;;   o3_2 = (where(abs(wl0-5007) lt dwl))[1]
;;;   o3ewold = ewcalc(wl0,fl1dold,cont1d,wl0[o3_1-sizepix],wl0[o3_2+sizepix])
;;;   o3ew = ewcalc(wl0,new1d,cont1d,wl0[o3_1-sizepix],wl0[o3_2+sizepix])
;;;   print,'O3 EW new/old: ',o3ew/o3ewold

   outfile = outdir + '/' + id[ii] + '_collapse2d.fits'
   writefits,outfile,null
   writefits,outfile,new2d,fhdr,/append
   writefits,outfile,newerr,ehdr,/append
   writefits,outfile,wl0,whdr,/append

   openw,1, outdir+'/'+id[ii]+'_spectrum.dat'
   for xx=0, n_elements(fl1dold)-1 do begin
      printf,1, wl0[xx], fl1dold[xx], err1dold[xx]
   endfor
   free_lun, 1

endfor
stop
end
