import matplotlib.pyplot as plt
import numpy as np
import pdb
import skimage.draw as draw
from scipy.misc import imresize, imsave

def denormalize(img_orginal_size, coords):
    return (0.5 * ((coords + 1.0) * img_orginal_size))

#imgs is [batch, y, x, f] and locs is [batch, numglimpse, 2] where last dim is
#y by x locs, with center as 0, 0, normalized to -1 to 1
def plotGlimpseTrace(imgs, locs, outdir, targetSize, nameprefix=""):
    (nbatch, ny, nx, nf) = imgs.shape
    #TODO turn this off
    assert(ny == nx)
    #TODO make radius a function of image shape
    radius = 1
    (nbatch_2, nglimpse, nloc) = locs.shape
    assert(nbatch == nbatch_2)
    assert(nloc == 2)
    assert(nf == 3 or nf == 1)

    imgs = imgs.copy()
    locs = locs.copy()

    #Make grayscale into rgb
    if(nf == 1):
        imgs = np.tile(imgs, [1, 1, 1, 3])

    locs = denormalize(60, locs)

    #Convert locs to integer locs
    y_locs = np.round(locs[..., 0]).astype(np.int32)
    x_locs = np.round(locs[..., 1]).astype(np.int32)

    for b in range(nbatch):
        img = imgs[b]
        y_loc = y_locs[b]
        x_loc = x_locs[b]

        #Draw filled circle for first point
        rr, cc = draw.circle(y_loc[0], x_loc[0], radius, shape=(targetSize, targetSize))
        #Draw in green
        img[rr, cc, 1] = 1

        #Draw unfilled circle for last point
        rr, cc = draw.circle_perimeter(y_loc[-1], x_loc[-1], radius, shape=(targetSize, targetSize))
        img[rr, cc, 1] = 1

        #Draw lines following locations
        for i in range(nglimpse - 1):
            rr, cc, val = draw.line_aa(y_loc[i], x_loc[i], y_loc[i+1], x_loc[i+1])
            #Clip values
            rr = np.clip(rr, 0, targetSize-1)
            cc = np.clip(cc, 0, targetSize-1)

            img[rr, cc, 1] = val

        imsave(outdir + "/glimpse_" + nameprefix + "_" + str(b) + ".png", img)
