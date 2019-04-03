'''
preprocessing

Created on May 09 2018 11:48 
#@author: Kevin Le 
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False
from dataset import SPCDataset

def main():
    pass

def aspect_resize(im):
    '''
    Preserve aspect ratio and resizes the image
    :param im: data array of image rescaled from 0->255
    :return: resized image
    '''
    ii = 256
    mm = [int (np.median (im[0, :, :])), int (np.median (im[1, :, :])), int (np.median (im[2, :, :]))]
    cen = np.floor (np.array ((ii, ii)) / 2.0).astype ('int')  # Center of the image
    dim = im.shape[0:2]
    if DEBUG:
        print("median {}".format (mm))
        print("ROC {}".format (cen))
        print("img dim {}".format (dim))
        # exit(0)

    if dim[0] != dim[1]:
        # get the largest dimension
        large_dim = max (dim)

        # ratio between the large dimension and required dimension
        rat = float (ii) / large_dim

        # get the smaller dimension that maintains the aspect ratio
        small_dim = int (min (dim) * rat)

        # get the indicies of the large and small dimensions
        large_ind = dim.index (max (dim))
        small_ind = dim.index (min (dim))
        dim = list (dim)

        # the dimension assigment may seem weird cause of how python indexes images
        dim[small_ind] = ii
        dim[large_ind] = small_dim
        dim = tuple (dim)
        if DEBUG:
            print('before resize {}'.format (im.shape))
        im = cv2.resize (im, dim)
        half = np.floor (np.array (im.shape[0:2]) / 2.0).astype ('int')
        if DEBUG:
            print('after resize {}'.format (im.shape))

        # make an empty array, and place the new image in the middle
        res = np.zeros ((ii, ii, 3), dtype='uint8')
        res[:, :, 0] = mm[0]
        res[:, :, 1] = mm[1]
        res[:, :, 2] = mm[2]

        if large_ind == 1:
            test = res[cen[0] - half[0]:cen[0] + half[0], cen[1] - half[1]:cen[1] + half[1] + 1]
            if test.shape != im.shape:
                res[cen[0] - half[0]:cen[0] + half[0] + 1, cen[1] - half[1]:cen[1] + half[1] + 1] = im
            else:
                res[cen[0] - half[0]:cen[0] + half[0], cen[1] - half[1]:cen[1] + half[1] + 1] = im
        else:
            test = res[cen[0] - half[0]:cen[0] + half[0] + 1, cen[1] - half[1]:cen[1] + half[1]]
            if test.shape != im.shape:
                res[cen[0] - half[0]:cen[0] + half[0] + 1, cen[1] - half[1]:cen[1] + half[1] + 1] = im
            else:
                res[cen[0] - half[0]:cen[0] + half[0] + 1, cen[1] - half[1]:cen[1] + half[1]] = im
    else:
        res = cv2.resize (im, (ii, ii))
        half = np.floor (np.array (im.shape[0:2]) / 2.0).astype ('int')

    if DEBUG:
        print('aspect_resize: {}'.format (res.shape))
    return res


def convert_to_8bit(img, auto_scale=True):
    # Convert to 8 bit and autoscale
    if auto_scale:

        result = np.float32 (img) - np.median (img)
        max_val1 = np.max (img)
        max_val2 = np.max (result)
        result[result < 0] = 0
        result = result / (0.5 * max_val1 + 0.5 * max_val2)

        bch = result[:, :, 0]
        gch = result[:, :, 1]
        rch = result[:, :, 2]
        b_avg = np.mean (bch)
        g_avg = np.mean (gch)
        r_avg = np.mean (rch)
        avg = np.mean (np.array ([b_avg, g_avg, r_avg]))
        # print "R: " + str(r_avg) + ", G: " + str(g_avg) + ", B: " + str(b_avg)
        bch = bch * 1.075
        rch = rch * 0.975
        gch = gch * 0.95
        # bch = bch*avg/b_avg
        # rch = rch*avg/r_avg
        # gch = gch*avg/g_avg
        # b_avg = np.mean(bch)
        # g_avg = np.mean(gch)
        # r_avg = np.mean(rch)
        # print "New R: " + str(r_avg) + ", G: " + str(g_avg) + ", B: " + str(b_avg)
        result[:, :, 0] = bch
        result[:, :, 1] = gch
        result[:, :, 2] = rch

        result = result / np.max (result)
        img_8bit = np.uint8 (255 * result)
    else:
        img_8bit = np.unit8 (img)

    return img_8bit

if __name__ == '__main__':
    root = '/data6/lekevin/cayman'
    img_dir = '/data6/lekevin/cayman/rawdata'
    csv_filename = root + '/data/1/data_{}.csv'

    # Test initialization
    dataset = {phase: SPCDataset (csv_filename=csv_filename.format (phase), img_dir=img_dir, phase=phase) for phase
               in ['train', 'val']}
    for phase in dataset:
        print (dataset[phase])

    # Test file, lbl retrieval
    fns, lbls = dataset['train'].get_fns ()
    testimg = fns[0]

    def aspect_resize_tst(testimg):
        img = cv2.imread (testimg)
        img = (img * 255).astype (np.uint8)
        img = aspect_resize (img)
        plt.savefig('asepct_resize.png', img)

    aspect_resize_tst()

    def convert_8bit_tst(testimg):
        img = cv2.imread (testimg)
        img = convert_8bit_tst(img)
        plt.savefig('8bit_convert.png', img)

    convert_8bit_tst(testimg)

