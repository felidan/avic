
def levelset_ivc2013(img=None, w=None, ini=None):
    #   Matlad code implementing the paper 
    #       'A new level set method for inhomogeneous image segmentation'
    #       Image and Vision Computing 31 (2013) 809C822
    #   Argin:
    #       img: input 2D gray image;
    #       w: coefficient cordinating local and global forces
    #       ini: initial level set contour
    #   Argout:
    #       imgls: final level set surface
    #       imgrec: evolving level set contours

    #   V1.0 by LI Bing Nan @ HFUT, Nov 2013

    #img = float(img(:, :, 1))
    imgn = 255. * (img - min(min(img))) /eldiv/ (max(max(img)) - min(min(img)))

    # get the size
    [nrow, ncol] = size(img)

    if nargin < 2 or w < 0:
        rho = 3    #Eq(15): adaptive weighting

        rf = rangefilt(imgn, ones(15))
        ct = rf /eldiv/ (max(max(imgn)))

        w = rho *elmul* mean(mean(ct)) *elmul* (1 - ct)
        

    if nargin < 3:
        u = sdf2circle(nrow, ncol, nrow / 2, ncol / 2, min(nrow / 8, ncol / 8))
    elif ini == 0:
        h_im = imshow(img, mcat([]))
        e = imellipse(gca)
        imgbk = createMask(e, h_im)

        u = 2 * (0.5 - imgbk)
    else:
        u = 2 * (0.5 - (ini > 0.5))

    enta = 0.0001        #enta=0.001;
    numIter = 1000        #1000;
    timestep = 2        #1;

    imgfilt = imfilter(img, fspecial(mstring('average'), 25), mstring('symmetric'), mstring('conv'))
    imgfilt = imgfilt - img

    imgrec = u

    # start level set evolution
    ul0 = 0
    tcost = 0
    for k in mslice[1:numIter]:
        # update level set function
        u = EVOL_BGFRLS(img, imgfilt, u, w, timestep)

        ul = sum(sum(u >= 0))
        tcost(k).lvalue = abs(ul - ul0) / ul
        
        if tcost(k) <= enta:
            break
        
        ul0 = ul

    imgls = u
    return [imgls,tcost,imgrec]

def EVOL_BGFRLS(img=None, imgfilt=None, imgini=None, w=None, timestep=None):
    #   This function updates the level set function according to Eq(22) 
    phi = imgini
    phi = NeumannBoundCond(phi)

    phi = (phi > 0) - (phi < 0)

    Hphi = Heaviside(phi)

    [c1, c2] = binaryfit(img, Hphi)
    p1 = (img - (c1 + c2) / 2) /eldiv/ (c1 - c2)

    [m1, m2] = binaryfit(imgfilt, Hphi)
    p2 = (imgfilt - (m1 + m2) / 2) /eldiv/ (m1 - m2)

    # updating the phi function     % Original CV2001 paper
    phi = phi + timestep *elmul* (w *elmul* p1 + (1 - w) *elmul* p2)

    phi = imfilter(phi, fspecial(mstring('gaussian'), 5, 5), mstring('symmetric'))#controlling smoothness
    
    return phi

def binaryfit(Img=None, H_phi=None):
    a = Img *elmul* H_phi
    numer_1 = sum(a(mslice[:]))
    denom_1 = sum(H_phi(mslice[:]))
    C1 = numer_1 / denom_1

    b = (1 - H_phi) *elmul* Img
    numer_2 = sum(b(mslice[:]))
    c = 1 - H_phi
    denom_2 = sum(c(mslice[:]))
    C2 = numer_2 / denom_2
    return [C1,C2]

def NeumannBoundCond(f=None):
    # Make a function satisfy Neumann boundary condition
    [nrow, ncol] = size(f)
    g = f
    g(mcat([1, nrow]), mcat([1, ncol])).lvalue = g(mcat([3, nrow - 2]), mcat([3, ncol - 2]))
    g(mcat([1, nrow]), mslice[2:end - 1]).lvalue = g(mcat([3, nrow - 2]), mslice[2:end - 1])
    g(mslice[2:end - 1], mcat([1, ncol])).lvalue = g(mslice[2:end - 1], mcat([3, ncol - 2]))
    return g

def Heaviside(phi=None):
    H = 0.5 * (1 + phi)
    return H

def sdf2circle(nrow=None, ncol=None, ci=None, cj=None, rd=None):
    # computes the signed distance to a circle
    [X, Y] = meshgrid(mslice[1:ncol], mslice[1:nrow])
    f = sqrt((X - cj) **elpow** 2 + (Y - ci) **elpow** 2) - rd
    return f
