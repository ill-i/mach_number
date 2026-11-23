using PyCall,PyPlot,LsqFit,Images,LinearAlgebra,StatsBase,FFTW,FITSIO,HDF5,Statistics
using LazCore,LazCyvecd,LazInstaller,LazIO,LazType,LazVGT

fnoise = FITS("ngc7538_13co_try1.fits");
d2d_noise=read(fnoise[1]);
nxn,nyn,nvn=size(d2d_noise);
dnoise=zeros(nxn,nyn,nvn);
dnoise=d2d_noise[:,:,:,1];
noise=std(dnoise[:,:,1][.~isnan.(dnoise[:,:,1])]);
println(noise);

#f=FITS("/Users/aluamukhash/astroproject/LMC/LMC_range_-50_90.fits"); 
f= FITS("ngc7538_13co_try1.fits");
#assign the 3D PPV cube to d;
d2d=read(f[1]);
nx,ny,nv=size(d2d);
d=zeros(nx,ny,nv);
d=d2d[:,:,:,1];

header=read_header(f[1]);
RA,DEC,vz=getcoor(header,nx,ny,nv);
dv=(vz[2]-vz[1]);
#noise=std(d[:,:,1][.~isnan.(d[:,:,1])]);
println(noise);

dn=20;#sub-block size
Ni=Int(round(abs((vz[1]-vz[1])/dv)))+1; #inital velocity
Nf=Int(round(abs((vz[end]-vz[1])/dv)))+1; #end velocity
#println(Nf)
#Qi, Ui, Qie, Uie=VGT_QU_mv_error(d,dn,Ni,Nf,noise);
Qi, Ui, Qie, Uie=VGT_QU_mv_error(d,dn,Ni,Nf,2);
Qa=Proj(Qi,3);#project the Q cube into Q map 
Ua=Proj(Ui,3);#project the Q cube into Q map 
Qa[Qa.==0].=NaN;
Ua[Ua.==0].=NaN; 

ker = 4;
Qb=imfilter_gaussian(Qa,ker); 
Ub=imfilter_gaussian(Ua,ker);
Ii = sum(d2d;dims=3)[:,:,1]
Qb[isnan.(Ii)].=NaN;
Ub[isnan.(Ii)].=NaN;
psi=0.5.*atan.(Ub,Qb);
psi.+=3pi/2;

imshow(psi',origin="lower",cmap="jet")

f = FITS("try1.fits", "w")
write(f,psi)
close(f)
