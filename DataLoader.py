import numpy as np
import struct
import time

def loadData(filename, out=None):
    '''
    :param filename: path to the file
    :param out: choice of representation out of ['vector' , 'covariance', 'coherence']
    :return: returns a matrix representation as specified in the parameter OUT
    Loads the data from a RAT file into a matrix of data points in
    lexicographical vector representation
    '''
    """
    """
    tStart = time.time()
    with open(filename, 'rb') as f:
        data = f.read(4)
        dim = struct.unpack('<I', data)[0]
        dim = (dim >> 24) | ((dim << 8) & 0x00FF0000) | ((dim >> 8) & 0x0000FF00)
        size = np.zeros(dim, dtype=np.int64)

        for i in range(dim):
            data = f.read(4)
            size[i] = struct.unpack('<I', data)[0]
            size[i] = (size[i] >> 24) | ((size[i] << 8) & 0x00FF0000) | ((size[i] >> 8) & 0x0000FF00)
            print('image dimention ' + str(i) + ' :', size[i])

        data = f.read(4)
        var = struct.unpack('<I', data)[0]
        var = (var >> 24) | ((var << 8) & 0x00FF0000) | ((var >> 8) & 0x0000FF00)
        data = f.read(4)
        dtype = struct.unpack('<I', data)[0]
        dtype = (dtype >> 24) | ((dtype << 8) & 0x00FF0000) | ((dtype >> 8) & 0x0000FF00)
        print('dtype: ', dtype)

        data = f.read(4)
        data = f.read(4)
        data = f.read(4)
        data = f.read(4)
        # read info
        info = f.read(80)
        print('info: ', info)

        nchannels = 0
        dsize = 0

        if var == 6:
            nchannels = 2
            dsize = 4
        else:
            print("ERROR: arraytyp not recognized (wrong format?)")

        # image data array initialisation
        img = np.zeros((2, size[2], size[1]))
        if dim == 3:
            real = np.zeros((size[2], size[1]))
            imag = np.zeros((size[2], size[1]))
            img[0] = real
            img[1] = imag
            rData = [img * size[0] for i in range(size[0])]

            for y in range(size[2]):
                for x in range(size[1]):
                    for i in range(size[0]):
                        realVal = 0.0
                        imagVal = 0.0
                        # read real part
                        buf = f.read(dsize)
                        buf = buf[::-1]
                        realVal = struct.unpack('<f', buf)[0]
                        rData[i][0][size[2] - y - 1, x] = realVal
                        # read imaginary part
                        buf = f.read(dsize)
                        buf = buf[::-1]
                        imagVal = struct.unpack('<f', buf)[0]
                        rData[i][1][size[2] - y - 1, x] = imagVal
    tEnd = time.time()
    print('Data Load successfully with time cost: ' + str(tEnd - tStart))

    # further formatting
    if out == 'vector':
        d_formatted = np.empty((size[2], size[1], 3), dtype=complex)
        for xi in range(size[1]):
            for yi in range(size[2]):
                d_pixel = []
                for idp, pol in enumerate(['HH', 'HV', 'VV']):
                    d_pixel.append(complex(rData[idp][0][yi][xi], rData[idp][1][yi][xi]))
                d_formatted[yi][xi] = np.array(d_pixel)
        rData = d_formatted

    elif out == 'covariance':
        d_formatted = np.empty((size[2], size[1], 3, 3), dtype=complex)
        for xi in range(size[1]):
            for yi in range(size[2]):
                d_pixel = []
                for idp, pol in enumerate(['HH', 'HV', 'VV']):
                    d_pixel.append(complex(rData[idp][0][yi][xi], rData[idp][1][yi][xi]))

                d_pixel = np.array(d_pixel).reshape((3, 1))

                d_formatted[yi][xi] = np.dot(np.matrix(d_pixel), np.matrix(d_pixel).getH())
        rData = d_formatted

    elif out == 'coherence':
        d_formatted = np.empty((size[2], size[1], 3, 3), dtype=complex)
        for xi in range(size[1]):
            for yi in range(size[2]):
                d_pixel = []
                for idp, pol in enumerate(['HH', 'HV', 'VV']):
                    d_pixel.append(complex(rData[idp][0][yi][xi], rData[idp][1][yi][xi]))

                d_pixel = np.array(d_pixel).reshape((3, 1))

                k_pixel = np.array([d_pixel[0] + d_pixel[2], d_pixel[0] - d_pixel[2], 2 * d_pixel[1]]) / (2 ** (1 / 2))

                d_formatted[yi][xi] = np.dot(np.matrix(k_pixel), np.matrix(k_pixel).getH())
        rData = d_formatted

    return rData