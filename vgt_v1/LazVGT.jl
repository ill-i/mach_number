module LazVGT

using PyCall,PyPlot,LsqFit,Images,LinearAlgebra,StatsBase,FFTW,FITSIO,HDF5,Statistics
using LazCore,LazCyvecd,LazInstaller,LazIO,LazType

export getcoor,de_resolution,getpeak,imfilter_gaussian,getmap
export sban2d_mv,sban2d_SNR,sban2d_SNR_mv,linspace,avB2dx
export VGT_QU_mv_error,VGT_QU,VGT_QU_mv


function avB2dx(Ax::Mat,Ay::Mat,dn)
	nx,ny=size(Ax)
	Ana=zeros(div(nx,dn),div(ny,dn));
	for  j in 1:div(ny,dn),i in 1:div(nx,dn)
	    is=(i-1)*dn+1;
	    ie=i*dn;
	    js=(j-1)*dn+1;
	    je=j*dn;
	    Axx=Ax[is:ie,js:je];
	    Ayy=Ay[is:ie,js:je];
	    Ana[i,j]=atan.(mean(Ayy[.~isnan.(Ayy)])/mean(Axx[.~isnan.(Ayy)]));
	end
	return Ana
end

function VGT_QU_mv_error(ppvi::Cube,dn,Ni,Nf,noise)
    nx,ny,nv=size(ppvi);
	Q=zeros(nx,ny,nv);
	U=zeros(nx,ny,nv);
    Qe=zeros(nx,ny,nv);
	Ue=zeros(nx,ny,nv);
	for k in Ni:Nf
        ds=zeros(nx,ny);   
        ds=ppvi[:,:,k];    
        #dsx,dsy=sobel_conv_2d(ds);
        dsx=circshift(ds,(1,0)).-ds
        dsy=circshift(ds,(0,1)).-ds
        dna,dns=sban2d_SNR_mv(dsx,dsy,dn,ds,noise);
        xx=div.(dns,pi);
        for i in 1:size(xx)[1],j in 1:size(xx)[2]
            dns[i,j]-=xx[i,j]*pi;
        end
        for ii in 1:nx,jj in 1:ny
			if(isnan(dna[ii,jj]))
				Q[ii,jj,k]=0;U[ii,jj,k]=0;
                Qe[ii,jj,k]=0;Ue[ii,jj,k]=0;
			else
                Q[ii,jj,k]=ds[ii,jj].*cos.(2.0.*dna[ii,jj]);
				U[ii,jj,k]=ds[ii,jj].*sin.(2.0.*dna[ii,jj]);
                ce=abs(2*sin.(2.0.*dna[ii,jj])*dns[ii,jj])
                se=abs(2*cos.(2.0.*dna[ii,jj])*dns[ii,jj])
                Qe[ii,jj,k]=abs(Q[ii,jj,k])*sqrt((noise/ds[ii,jj])^2+(ce/cos.(2.0.*dna[ii,jj]))^2);
                Ue[ii,jj,k]=abs(U[ii,jj,k])*sqrt((noise/ds[ii,jj])^2+(se/sin.(2.0.*dna[ii,jj]))^2);
			end
        end
        println(string(k)*"/"*string(Nf))
    end
    return Q,U,Qe,Ue
end


function hist_new(data,range)
    # Linear histogram function
    # Wrapper for the `hist` function for functions written for julia v0.5-
    h=fit(Histogram,data,range)
    ax=h.edges[1];
    ac=h.weights;
    return ax,ac
end

function sban2d_mv(Ax::Mat,Ay::Mat,dn)
	nx,ny=size(Ax)
	Ana=zeros(nx,ny);
	Ans=zeros(nx,ny);
    for  j in dn/2:ny-dn/2,i in dn/2:nx-dn/2
        i=Int(i);
        j=Int(j);
        is=Int(i-dn/2+1);
        ie=Int(i+dn/2);
        js=Int(j-dn/2+1);
        je=Int(j+dn/2);
        try
            Axx=Ax[is:ie,js:je];
            Ayy=Ay[is:ie,js:je];
            binsize=100;
            Apeak,Adisp=fit_gaussian_2d(Axx,Ayy,binsize);
            Ana[Int(i),Int(j)]=Apeak;
            Ans[Int(i),Int(j)]=Adisp;
        catch
            Ana[Int(i),Int(j)]=NaN;
            Ans[Int(i),Int(j)]=NaN;
        end
        #println("j="*string(j)*";i="*string(i))
    end
    Ana[1:Int(dn/2-1),:].=NaN;
    Ana[:,1:Int(dn/2-1)].=NaN;
    Ana[Int(nx-dn/2+1):nx,:].=NaN;
    Ana[:,Int(ny-dn/2+1):ny].=NaN;
    Ans[isnan.(Ana)].=NaN; 
	return Ana,Ans
end

function getmap(ppv::Cube,vz,peak)
    nx,ny,nv=size(ppv);
	I=zeros(nx,ny);
	C=zeros(nx,ny);
	Ch=zeros(nx,ny);
	for i in 1:nx,j in 1:ny,k in 1:nv
        I[i,j]+=ppv[i,j,k]
        C[i,j]+=ppv[i,j,k]*vz[k]
	end
	C=C./I;
    C[C.>maximum(vz)].=NaN
    C[C.< minimum(vz)].=NaN;
    dC=std(C[.~isnan.(C)]);
	for i in 1:nx,j in 1:ny,k in 1:nv
        vchp=vz[peak];
        vch=vz[k];
     if (abs(vch-vchp)<0.5*dC)
        Ch[i,j]+=ppv[i,j,k]
	 end
	end
    return I,C,Ch
end

function getcoor(header,nx,ny,nv)
    CRPIX1=header["CRPIX1"];
    CRVAL1=header["CRVAL1"];
    CDELT1=header["CDELT1"];
    CRPIX2=header["CRPIX2"];
    CRVAL2=header["CRVAL2"];
    CDELT2=header["CDELT2"];
    CRPIX3=header["CRPIX3"];
    CRVAL3=header["CRVAL3"];
    CDELT3=header["CDELT3"];
    RA=zeros(nx);
    DEC=zeros(ny);
    vz=zeros(nv);
    for i in 1:nx
        RA[i]=CRVAL1+(i-CRPIX1)*CDELT1
    end
    for j in 1:ny
        DEC[j]=CRVAL2+(j-CRPIX2)*CDELT2
    end
    for k in 1:nv
        vz[k]=CRVAL3+(k-CRPIX3)*CDELT3
    end
    xi=CRVAL1+(0-CRPIX1)*CDELT1;
    xe=CRVAL1+(nx-CRPIX1)*CDELT1;
    yi=CRVAL2+(0-CRPIX2)*CDELT2;
    ye=CRVAL2+(ny-CRPIX2)*CDELT2;
    print("xi=",xi,";xe=",xe,";yi=",yi,";ye=",ye)
    return RA,DEC,vz
end

function getpeak(d::Cube)
    nx,ny,nv=size(d);
    peak=zeros(nv);
    for k in 1:nv, j in 1:ny, i in 1:nx
      if(~isnan(d[i,j,k]))
       peak[k]+=d[i,j,k]
      end
    end
    return peak
end

function imfilter_gaussian(d::Mat,p::Number)
    im=imfilter(d,Kernel.gaussian(p),NA())
    return im
end

function de_resolution(b::Mat,dn)
    nx,ny=size(b)
    bx=zeros(div(nx,dn),div(ny,dn));
   for i in 1:div(nx,dn),j in 1:div(ny,dn)
        a=b[i*dn-dn+1:i*dn,j*dn-dn+1:j*dn]
        bx[i,j]=mean(a)
   end
    return bx
end

# """@internal"""
function maxid(ax::Vec)
	return findall(ax.==maximum(ax));
end
   
   # """@internal"""
function maxid(ax::Mat)
	return findall(ax.==maximum(ax));
end

function fit_gaussian_2dx(Ax,Ay,binsize)
	# KH : The subblock Gaussian fitting algorithm
	#      this algorithm was ignoring the periodicity of the angle
	#      and use the Real space defined Gaussian to fit instead.
	#      a fftshift is required to play for the two cases.
	phi=atan.(Ay./Ax);
	phix=phi[.~isnan.(phi)];
	Gauss(x,p)=p[1]*exp.(.-(x.-p[2]).^2 .*p[3]);
	Gauss_x(x,p)=p[1]*exp.(.-(x.-p[2]).^2 .*p[3]).+p[4];
	ax,ac=hist_new(phi[:],linspace(-pi/2,pi/2,binsize+1))
	ax=.5*(ax[1:end-1]+ax[2:end]);
	if (abs.(ax[maxid(ac)])[1]<pi/4)
		y=ac./sum(ac);
		y[isnan.(y)].=0;
		try
			fit1=curve_fit(Gauss_x,ax,y,[maximum(y),0.0,1.0,0])
			if(maximum(y).==0)
				sigma=NaN;
				return fit1.param[2],sigma;
			else
				sigma=margin_error(fit1,0.05);
				return fit1.param[2],sigma[2];
			end
		catch
			fit1=curve_fit(Gauss,ax,y,[maximum(y),0.0,1.0])
			if(maximum(y).==0)
				sigma=NaN;
				return fit1.param[2],sigma;
			else
				sigma=margin_error(fit1,0.05);
				return fit1.param[2],sigma[2];
			end
		end
	else
		ax=ax.-pi./2;
		ac=fftshift(ac);
		y=ac./sum(ac);
		y[isnan.(y)].=0;
		try
			fit1=curve_fit(Gauss_x,ax,y,[maximum(y),-pi/2,1.0,0]);
			if(maximum(y).==0)
				sigma=NaN;
				return fit1.param[2],sigma;
			else
				sigma=margin_error(fit1,0.05);
				return fit1.param[2],sigma[2];
			end
		catch 
			fit1=curve_fit(Gauss,ax,y,[maximum(y),-pi/2,1.0]);
			if(maximum(y).==0)
				sigma=NaN;
				return fit1.param[2],sigma;
			else
				sigma=margin_error(fit1,0.05);
				return fit1.param[2],sigma[2];
			end
		end
	end
end

function sban2d_SNR(Ax::Mat,Ay::Mat,dn,Ch::Mat,noise)
	nx,ny=size(Ax)
	Ana=zeros(div(nx,dn),div(ny,dn));
	Ans=zeros(div(nx,dn),div(ny,dn));
	Ax[Ch.<3*noise].=NaN;
	Ay[Ch.<3*noise].=NaN;
	for  j in 1:div(ny,dn),i in 1:div(nx,dn)
	   	is=(i-1)*dn+1;
	   	ie=i*dn;
	   	js=(j-1)*dn+1;
	   	je=j*dn;
	   	Axx=Ax[is:ie,js:je];
	   	Ayy=Ay[is:ie,js:je];
	   	Axx=Axx[.~isnan.(Axx)];
	   	Ayy=Ayy[.~isnan.(Ayy)];
	   	if(length(Axx)>100)
			binsize=100;
			Apeak,Adisp=fit_gaussian_2dx(Axx,Ayy,binsize);
			Ana[i,j]=Apeak;
			Ans[i,j]=Adisp;
		else
			Ana[i,j]=NaN;
			Ans[i,j]=NaN;
	   #println("j="*string(j)*";i="*string(i))
		end
	end
	return Ana,Ans
end

function sban2d_SNR_mv(Ax::Mat,Ay::Mat,dn,Ch::Mat,noise)
	nx,ny=size(Ax)
	Ana=zeros(nx,ny);
	Ans=zeros(nx,ny);
    Ax[Ch.<3*noise].=NaN;
	Ay[Ch.<3*noise].=NaN;
    for  j in dn/2:ny-dn/2,i in dn/2:nx-dn/2
        i=Int(i);
        j=Int(j);
        is=Int(i-dn/2+1);
        ie=Int(i+dn/2);
        js=Int(j-dn/2+1);
        je=Int(j+dn/2);
        try
            Axx=Ax[is:ie,js:je];
            Ayy=Ay[is:ie,js:je];
            Axx=Axx[.~isnan.(Axx)];
            Ayy=Ayy[.~isnan.(Ayy)];
            if(length(Axx)>100)
                binsize=100;
                Apeak,Adisp=fit_gaussian_2dx(Axx,Ayy,binsize);
                Ana[Int(i),Int(j)]=Apeak;
                Ans[Int(i),Int(j)]=Adisp;
            else
                Ana[Int(i),Int(j)]=NaN;
                Ans[Int(i),Int(j)]=NaN;
            #println("j="*string(j)*";i="*string(i))
            end
        catch
            Ana[Int(i),Int(j)]=NaN;
            Ans[Int(i),Int(j)]=NaN;
        end
        #println("j="*string(j)*";i="*string(i))
    end
    Ana[1:Int(dn/2-1),:].=NaN;
    Ana[:,1:Int(dn/2-1)].=NaN;
    Ana[Int(nx-dn/2+1):nx,:].=NaN;
    Ana[:,Int(ny-dn/2+1):ny].=NaN;
    Ans[isnan.(Ana)].=NaN; 
	return Ana,Ans
end

function VGT_QU(ppvi::Cube,dn,Ni,Nf,noise)
    nx,ny,nv=size(ppvi);
	nxx,nyy=div(nx,dn),div(ny,dn);
	Q=zeros(nxx,nyy,nv);
	U=zeros(nxx,nyy,nv);
	Q2d=zeros(nxx,nyy);
	U2d=zeros(nxx,nyy);
	for k in Ni:Nf
        ds=zeros(nx,ny);   
        ds=ppvi[:,:,k];    
        #dsx,dsy=sobel_conv_2d(ds);
        dsx=circshift(ds,(1,0)).-ds
        dsy=circshift(ds,(0,1)).-ds
        dna,dns=sban2d_SNR(dsx,dsy,dn,ds,noise);
        Ia=de_resolution(ppvi[:,:,k],dn); 
        for ii in 1:div(nx,dn),jj in 1:div(ny,dn)
			if(isnan(dna[ii,jj]))
				Q[ii,jj,k]=0;
				U[ii,jj,k]=0;
			else
				Q[ii,jj,k]=Ia[ii,jj].*cos.(2 .*dna[ii,jj]);
				U[ii,jj,k]=Ia[ii,jj].*sin.(2 .*dna[ii,jj]);
			end
        end
        Q2d.+=Q[:,:,k];
        U2d.+=U[:,:,k];
    end
    return Q2d,U2d
end

function VGT_QU_mv(ppvi::Cube,dn,Ni,Nf,noise)
    nx,ny,nv=size(ppvi);
	Q=zeros(nx,ny,nv);
	U=zeros(nx,ny,nv);
	Q2d=zeros(nx,ny);
	U2d=zeros(nx,ny);
	for k in Ni:Nf
		println(k)
        ds=zeros(nx,ny);   
        ds=ppvi[:,:,k];    
        #dsx,dsy=sobel_conv_2d(ds);
        dsx=circshift(ds,(1,0)).-ds
        dsy=circshift(ds,(0,1)).-ds
        dna,dns=sban2d_SNR_mv(dsx,dsy,dn,ds,noise);
        #Ia=de_resolution(ppvi[:,:,k],dn); 
        for ii in 1:nx,jj in 1:ny
			if(isnan(dna[ii,jj]))
				Q[ii,jj,k]=0;
				U[ii,jj,k]=0;
			else
				Q[ii,jj,k]=ds[ii,jj].*cos.(2.0.*dna[ii,jj]);
				U[ii,jj,k]=ds[ii,jj].*sin.(2.0.*dna[ii,jj]);
			end
        end
        Q2d.+=Q[:,:,k];
        U2d.+=U[:,:,k];
    end
    return Q2d,U2d
end

function linspace(a,b,c)
	# KH: a Lower limit, B Upper limit, C number of points
	# KH : Construct a 1d linspace
	width =(b-a)/c;
	x=zeros(c);
	xx=a:width:b
	for i in 1:c
        x[i]=xx[i]
	end
	return x
end

end
