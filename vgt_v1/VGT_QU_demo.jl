using PyCall,PyPlot,LsqFit,Images,LinearAlgebra,StatsBase,FFTW,FITSIO,HDF5,Statistics
using LazCore,LazCyvecd,LazInstaller,LazIO,LazType,LazVGT
@pyimport numpy as np
@pyimport glob as glob

#commands for intalling package:
#using Pkg;
#Pkg.add("name of package")

#The path of external packages beginging with Laz... should be included to the LOADPATH
#############################
#open FITS data set:
f=FITS("~/Dropbox/NU/NGC7538/data/fits/13co/jcmth20090921_00014_02_reduced001_obs_000.fits");
mom0 = 
FITS("~/Dropbox/NU/NGC7538/data/fits/13co/jcmth20090921_00014_02_reduced001_obs_000.fits"); 
#assign the 3D PPV cube to d;
d2d=read(f[1]);
nx,ny,nv=size(d2d);
d=zeros(nx,ny,nv);
d=d2d[:,:,:,1];

#read header
header=read_header(f[1]);
#calculate coordinate and line-of-sight velocity
RA,DEC,vz=getcoor(header,nx,ny,nv);
#velocity resolution
dv=(vz[2]-vz[1]); #unit: m/s
#caclulate noise level 
noi=std(d[:,:,1][.~isnan.(d[:,:,1])])

###########################constructing Pseudo Stokes parameters
dn=32;#sub-block size
Ni=Int(round(abs((vz[1]-vz[1])/dv)))+1; #inital velocity
Nf=Int(round(abs((vz[end]-vz[1])/dv)))+1; #end velocity
#calculating Pseudo-Stokes parameters
Qi,Ui, Qie, Uie=VGT_QU_mv_error(d,dn,Ni,Nf,noi);
#Qie and Uie are the uncertainties.
#this function outputs of Q, U cubes, instead of maps.
#pixels with noise larger than three sigma levels are blanked out

Qa=Proj(Qi,3);#project the Q cube into Q map 
Ua=Proj(Ui,3);#project the Q cube into Q map 
Qa[Qa.==0].=NaN;Ua[Ua.==0].=NaN; #Qa=0, Ua=0 means intensity is less then 3 times signal to noise ratio.
ker = 4;
# the smoothing width is FWHM = ker*2.355
Qb=imfilter_gaussian(Qa,ker); #Gaussian smoothing the map: FWHM=ker*2.355 pixels
Ub=imfilter_gaussian(Ua,ker);
psi=0.5.*atan.(Ub,Qb);
# rotate the polarization vectors by 90 degree to indicate the magnetic field direction;
psi.+=pi/2;

# decreasing the resolution of psi for visulization purpose
dnn=4;
phi=avB2dx(cos.(psi),sin.(psi),dnn);

Ii=read(mom0[1])
nx,ny=size(Ii);
Xd,Yd=np.meshgrid(div(dnn,2)+1:dnn:div(dnn,2)+ny,div(dnn,2)+1:dnn:div(dnn,2)+nx);

# supuerimpose the gradient and magnetic field vectors on the same intensity map;
# note Julia uses the Cartesian conventation for definiing angle, i.e., the angle is from right to top anticlockwisely 
figure()
imshow(Ii'.*abs(dv),origin="lower", cmap="Greys")
cb=colorbar(pad=0)
cb.ax.tick_params(labelsize=20,direction="in")
clim(0,80)
cb.set_label(label="Intensity [km/s]",size=20);
xlabel("R.A.(J2000) [degree]",size=20);
ylabel("Dec.(J2000) [degree]",size=20);
tick_params(direction="in",labelsize=16);
xticks([0,15,30,45],["349.183","349.085","349.987","349.889"],size=20);
yticks([0,15,30,45],["61.351","61.448","61.546","61.644"],size=20);
psi_resized = psi[1:dnn:end, 1:dnn:end]  # Уменьшаем psi до сетки dnn

# Проверяем размеры
println("Xd size: ", size(Xd))
println("Yd size: ", size(Yd))
println("psi_resized size: ", size(psi_resized))

# Теперь quiver() будет работать
quiver(Yd, Xd, cos.(psi_resized), sin.(psi_resized), headwidth=0, scale=1, 
color="r")

#quiver(Yd[1:end-1,1:end-1],Xd[1:end-1,1:end-1],cos.(psi),sin.(psi),headwidth=0,scale=20,color="r")
#quiver(Yd[1:end-1,1:end-1],Xd[1:end-1,1:end-1],-cos.(psi),-sin.(psi),headwidth=0,scale=20,color="r")
title("VGT_demo")

#calculating uncertainties psie
Qe = Qie
Ue = Uie

Qe[isnan.(Qe)].=0;Ue[isnan.(Ue)].=0;
Qex=sqrt.(Proj(Qe.^2,3));
Uex=sqrt.(Proj(Ue.^2,3));

UQ=Ub./Qb;
UQe=abs.(UQ).*sqrt.((Qex./Qb).^2.0.+(Uex./Ub).^2.0);
psie=0.5.*UQe./(1.0.+UQ.^2.0);
