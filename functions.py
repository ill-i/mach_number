import numpy as np
from astropy.coordinates import SkyCoord

def stokespolparam(I, Q, U):
    '''
    Find polarization degree, p, and angle, psi, given Stokes I, Q, U
    Output: degree of polarization from 0 to 1 and 
    angles in degrees from -90 to 90
    '''
    p = np.sqrt(Q**2+U**2)/I
    psi = 0.5*np.arctan2(U, Q)
    for i in range(len(psi)):
        for j in range(len(psi)):
            if psi[i,j] < - np.pi/2:
                psi[i,j] += np.pi
            elif psi[i,j] > np.pi/2:
                psi[i,j] -= np.pi
    psi = psi*180/np.pi
    return p, psi

def pol_vec_components(p, psi):
    '''
    get vector components given polarization degree, p, and angle, psi
    angles are according to IAU standard, where 0 is at North
    and -90 < psi < 90
    '''
    psi = psi * np.pi/180 + np.pi/2
    vx = p*np.cos(psi)
    vy = p*np.sin(psi)
    return vx, vy

def polang_eqtogal(polang, l, b):
    """
    polang: polarization angle in equatorial system in degrees
    l, b: galactic longitude and latitude
    """
    lncp = 122.93192
    bncp = 27
    o = np.arctan2(np.sin((lncp-l)*np.pi/180), np.tan(bncp*np.pi/180)*np.cos(b*np.pi/180)-np.sin(b*np.pi/180)*np.cos((lncp-l)*np.pi/180))*180/np.pi
    psig = polang + o
    for i in range(len(psig)):
        if psig[i] < -90:
            psig[i] = psig[i] + 180
        if psig[i] > 90:
            psig[i] = 180 - psig[i]
    return psig

def eqtogal(h, m, s, d, mnt, sec, eqnx='J2000'):
    '''
    Convert equatorial coordinates to galactic
    h, m, s: hours, minutes and seconds of RA
    d, min, sec: days, minutes, seconds of Declination
    '''
    l = np.zeros(len(h))
    b = np.zeros(len(h))
    for i in range(len(h)):
        coords = SkyCoord(str(int(h[i]))+'h'+str(int(m[i]))+'m'+str(s[i])+'s', 
        str(int(d[i]))+'d'+str(int(mnt[i]))+'m'+str(sec[i])+'s', equinox=eqnx)
        l[i] = coords.galactic.l.deg
        b[i] = coords.galactic.b.deg
    return l, b

def polang_map(polang, l, b, wcs):
    '''
    Make a 2D map from 1D array
    '''
    s = wcs.array_shape[0]
    ang_map = np.zeros((s,s))
    lpix = np.zeros(len(l))
    bpix = np.zeros(len(l))
    for i in range(len(l)):
        lpix[i], bpix[i] = wcs.all_world2pix(l[i], b[i], 2)
        lpix[i] = np.around(lpix[i])
        bpix[i] = np.around(bpix[i])
        ang_map[int(bpix[i]), int(lpix[i])] = polang[i]
    for r in range(s):
        for c in range(s):
            if ang_map[r,c] == 0:
                ang_map[r,c] = None
    return ang_map

def gaussian(x, res):
    '''
    res = resolution
    '''
    sigma = res/(2*np.sqrt(2*np.log(2)))
    return np.exp(-x**2/(2*sigma**2))

def common_elem(a1,a2):
    '''
    Find a common element of 2 1D arrays
    Returns False if there are no common elements
    '''
    result = False
    for i in range(len(a1)):
        for j in range(len(a2)):
            if a1[i] == a2[j]:
                result = a1[i]
                return result
    return result

def ang_interp(a):
    '''
    interpolate a 2D map of angle values in degrees, while considering
    the distance to the closest neighbours
    a is a 2D square array 
    '''
    count = 0
    anew = np.copy(a)
    a = a*np.pi/180
    l = len(a)
    for r in range(3,l-3):
        for c in range(3,l-3):
            if np.isnan(a[r,c]) == True:
                data1 = np.array([
                    a[r-1,c-1], a[r-1,c], a[r-1, c+1], a[r,c-1], a[r,c+1],
                    a[r+1, c-1], a[r+1,c], a[r+1,c+1]])
                data2 = np.array([
                    a[r-2,c-2], a[r-2,c-1], a[r-2, c], a[r-2,c+1], a[r-2,c+2],
                    a[r-1, c-2], a[r-1,c+2], a[r,c-2], a[r,c+2],a[r+1,c-2], a[r+1,c+2],
                    a[r+2,c-2], a[r+2,c-1], a[r+2, c], a[r+2,c+1], a[r+2,c+2]])
                data3 = np.array([
                    a[r-3,c-2], a[r-3, c-1], a[r-3,c], a[r-3,c+1], a[r-3,c+2],
                    a[r-2, c-3], a[r-2,c+3], a[r-1,c-3], a[r-1,c+3],a[r,c-3], a[r,c+3],
                    a[r+1, c-3], a[r+1,c+3], a[r+2,c-3], a[r+2,c+3],
                    a[r+3,c-2], a[r+3, c-1], a[r+3,c], a[r+3,c+1],a[r+3,c+2]])
                data1 = data1[~np.isnan(data1)]
                data2 = data2[~np.isnan(data2)]
                data3 = data3[~np.isnan(data3)]
                data = np.append(data1,data2)
                data = np.append(data, data3)
                if len(data) > 3:
                    data1y = 0.7528*np.sin(data1)
                    data2y = 0.3212*np.sin(data2)
                    data3y = 0.0777*np.sin(data3)
                    data1x = 0.7528*np.cos(data1)
                    data2x = 0.3212*np.cos(data2)
                    data3x = 0.0777*np.cos(data3)
                    datax = np.append(data1x, data2x)
                    datax = np.append(datax, data3x)
                    datay = np.append(data1y, data2y)
                    datay = np.append(datay, data3y)
                    x = np.sum(datax)
                    y = np.sum(datay)
                    anew[r,c] = np.arctan2(y,x)*180/np.pi
                    count += 1
                    if anew[r,c] > 90:
                        anew[r,c] = anew[r,c] - 180
                    elif anew[r,c] < -90:
                        anew[r,c] = anew[r,c] + 180
    print("+ ", count, " new values")
    return anew

def sig_interp(a, sig):
    '''
    interpolate square array a with consideration of errors of a
    a and sig are in degrees
    '''
    count = 0
    anew = np.copy(a)
    a = a*np.pi/180
    l = len(a)
    for r in range(3,l-3):
        for c in range(3,l-3):
            if np.isnan(a[r,c]) == True:
                data1 = np.array([
                    a[r-1,c-1], a[r-1,c], a[r-1, c+1],
                    a[r,c-1], a[r,c+1],
                    a[r+1, c-1], a[r+1,c], a[r+1,c+1]])
                sigdata1 = np.array([
                    sig[r-1,c-1], sig[r-1,c], sig[r-1, c+1],
                    sig[r,c-1], sig[r,c+1],
                    sig[r+1, c-1], sig[r+1,c], sig[r+1,c+1]])
                data2 = np.array([
                    a[r-2,c-2], a[r-2,c-1], a[r-2, c], a[r-2,c+1], a[r-2,c+2],
                    a[r-1, c-2], a[r-1,c+2],
                    a[r,c-2], a[r,c+2],
                    a[r+1,c-2], a[r+1,c+2],
                    a[r+2,c-2], a[r+2,c-1], a[r+2, c], a[r+2,c+1], a[r+2,c+2]])
                sigdata2 = np.array([
                    sig[r-2,c-2], sig[r-2,c-1], sig[r-2, c], sig[r-2,c+1], sig[r-2,c+2],
                    sig[r-1, c-2], sig[r-1,c+2],
                    sig[r,c-2], sig[r,c+2],
                    sig[r+1,c-2], sig[r+1,c+2],
                    sig[r+2,c-2], sig[r+2,c-1], sig[r+2, c], sig[r+2,c+1], sig[r+2,c+2]])
                data3 = np.array([
                    a[r-3,c-2], a[r-3, c-1], a[r-3,c], a[r-3,c+1], a[r-3,c+2],
                    a[r-2, c-3], a[r-2,c+3], a[r-1,c-3], a[r-1,c+3],a[r,c-3], a[r,c+3],
                    a[r+1, c-3], a[r+1,c+3], a[r+2,c-3], a[r+2,c+3],
                    a[r+3,c-2], a[r+3, c-1], a[r+3,c], a[r+3,c+1],a[r+3,c+2]])
                sigdata3 = np.array([
                    sig[r-3,c-2], sig[r-3, c-1], sig[r-3,c], sig[r-3,c+1],
                    sig[r-3,c+2], sig[r-2, c-3], sig[r-2,c+3], sig[r-1,c-3],
                    sig[r-1,c+3], sig[r,c-3], sig[r,c+3], sig[r+1, c-3], sig[r+1,c+3],
                    sig[r+2,c-3], sig[r+2,c+3], sig[r+3,c-2], sig[r+3, c-1],
                    sig[r+3,c], sig[r+3,c+1], sig[r+3,c+2]])
                data1 = data1[~np.isnan(data1)]
                data2 = data2[~np.isnan(data2)]
                data3 = data3[~np.isnan(data3)]
                data = np.append(data1,data2)
                data = np.append(data, data3)
                sigdata1 = sigdata1[~np.isnan(sigdata1)]
                sigdata2 = sigdata2[~np.isnan(sigdata2)]
                sigdata3 = sigdata3[~np.isnan(sigdata3)]
                if len(data) > 5:
                    data1y = 0.7528*np.sin(data1)*gaussian(sigdata1, 2.4277)
                    data2y = 0.3212*np.sin(data2)*gaussian(sigdata2, 2.4277)
                    data3y = 0.0777*np.sin(data3)*gaussian(sigdata3, 2.4277)
                    data1x = 0.7528*np.cos(data1)*gaussian(sigdata1, 2.4277)
                    data2x = 0.3212*np.cos(data2)*gaussian(sigdata2, 2.4277)
                    data3x = 0.0777*np.cos(data3)*gaussian(sigdata3, 2.4277)
                    datax = np.append(data1x, data2x)
                    datax = np.append(datax, data3x)
                    datay = np.append(data1y, data2y)
                    datay = np.append(datay, data3y)
                    x = np.sum(datax)
                    y = np.sum(datay)
                    anew[r,c] = np.arctan2(y,x)*180/np.pi
                    count += 1
                    if anew[r,c] > 90:
                        anew[r,c] = anew[r,c] - 180
                    elif anew[r,c] < -90:
                        anew[r,c] = anew[r,c] + 180
    print("+ ", count, " new values")
    return anew

def star_interp(a, sig, wcs, stars_pa, stars_spa, stars_l, stars_b):
    '''
    a and sig are angles of RHT and their errors in degrees,
    stars_pa is polarization angle of stars in degrees,
    l and b - their coordiantes
    '''
    count = 0
    l = len(a)
    lpix, bpix = wcs.all_world2pix(stars_l, stars_b, 2)
    lpix = np.around(lpix)
    bpix = np.around(bpix)
    pa_map = np.zeros((l,l))
    pa_map[:,:] = None
    for i in range(len(stars_pa)):
        pa_map[int(bpix[i]), int(lpix[i])] = stars_pa[i]
        a[int(bpix[i]), int(lpix[i])] = stars_pa[i]
        sig[int(bpix[i]), int(lpix[i])] = stars_spa[i]
    anew = np.copy(a)
    a = a*np.pi/180
    for r in range(3,l-3):
        for c in range(3,l-3):
            if np.isnan(a[r,c]) == True:
                data1 = np.array([
                    a[r-1,c-1], a[r-1,c], a[r-1, c+1], a[r,c-1], a[r,c+1],
                    a[r+1, c-1], a[r+1,c], a[r+1,c+1]])
                sigdata1 = np.array([
                    sig[r-1,c-1], sig[r-1,c], sig[r-1, c+1], sig[r,c-1], sig[r,c+1],
                    sig[r+1, c-1], sig[r+1,c], sig[r+1,c+1]])
                data2 = np.array([
                    a[r-2,c-2], a[r-2,c-1], a[r-2, c], a[r-2,c+1], a[r-2,c+2],
                    a[r-1, c-2], a[r-1,c+2], a[r,c-2], a[r,c+2],a[r+1,c-2], a[r+1,c+2],
                    a[r+2,c-2], a[r+2,c-1], a[r+2, c], a[r+2,c+1], a[r+2,c+2]])
                sigdata2 = np.array([
                    sig[r-2,c-2], sig[r-2,c-1], sig[r-2, c], sig[r-2,c+1], sig[r-2,c+2],
                    sig[r-1, c-2], sig[r-1,c+2], sig[r,c-2], sig[r,c+2],sig[r+1,c-2], sig[r+1,c+2],
                    sig[r+2,c-2], sig[r+2,c-1], sig[r+2, c], sig[r+2,c+1], sig[r+2,c+2]])
                data3 = np.array([
                    a[r-3,c-2], a[r-3, c-1], a[r-3,c], a[r-3,c+1], a[r-3,c+2],
                    a[r-2, c-3], a[r-2,c+3], a[r-1,c-3], a[r-1,c+3],a[r,c-3], a[r,c+3],
                    a[r+1, c-3], a[r+1,c+3], a[r+2,c-3], a[r+2,c+3],
                    a[r+3,c-2], a[r+3, c-1], a[r+3,c], a[r+3,c+1],a[r+3,c+2]])
                sigdata3 = np.array([
                    sig[r-3,c-2], sig[r-3, c-1], sig[r-3,c], sig[r-3,c+1],
                    sig[r-3,c+2], sig[r-2, c-3], sig[r-2,c+3], sig[r-1,c-3],
                    sig[r-1,c+3], sig[r,c-3], sig[r,c+3], sig[r+1, c-3], sig[r+1,c+3],
                    sig[r+2,c-3], sig[r+2,c+3], sig[r+3,c-2], sig[r+3, c-1],
                    sig[r+3,c], sig[r+3,c+1], sig[r+3,c+2]])
                data1 = data1[~np.isnan(data1)]
                data2 = data2[~np.isnan(data2)]
                data3 = data3[~np.isnan(data3)]
                data = np.append(data1,data2)
                data = np.append(data, data3)
                sigdata1 = sigdata1[~np.isnan(sigdata1)]
                sigdata2 = sigdata2[~np.isnan(sigdata2)]
                sigdata3 = sigdata3[~np.isnan(sigdata3)]
                if len(data) > 5:
                    data1y = 0.7528*np.sin(data1)*gaussian(sigdata1, 2.4277)
                    data2y = 0.3212*np.sin(data2)*gaussian(sigdata2, 2.4277)
                    data3y = 0.0777*np.sin(data3)*gaussian(sigdata3, 2.4277)
                    data1x = 0.7528*np.cos(data1)*gaussian(sigdata1, 2.4277)
                    data2x = 0.3212*np.cos(data2)*gaussian(sigdata2, 2.4277)
                    data3x = 0.0777*np.cos(data3)*gaussian(sigdata3, 2.4277)
                    datax = np.append(data1x, data2x)
                    datax = np.append(datax, data3x)
                    datay = np.append(data1y, data2y)
                    datay = np.append(datay, data3y)
                    if common_elem(data, stars_pa) != False:
                        st = common_elem(data,stars_pa)
                        stx = 2*np.cos(st*np.pi/180)
                        sty = 2*np.sin(st*np.pi/180)
                        datax = np.append(datax, stx)
                        datay = np.append(datay, sty)
                    x = np.sum(datax)
                    y = np.sum(datay)
                    anew[r,c] = np.arctan2(y,x)*180/np.pi
                    count += 1
                    if anew[r,c] > 90:
                        anew[r,c] = anew[r,c] - 180
                    elif anew[r,c] < -90:
                        anew[r,c] = anew[r,c] + 180
    print("+ ", count, " new values")
    return anew


def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs ):
    if ax is None :
        fig , ax = plt.subplots()
    
    data, x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )
    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    return ax


