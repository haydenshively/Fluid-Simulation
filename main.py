"""
https://thecodingtrain.com/CodingChallenges/132-fluid-simulation.html
http://www.dgp.toronto.edu/people/stam/reality/Research/pdf/GDC03.pdf
https://mikeash.com/pyblog/fluid-simulation-for-dummies.html
"""

import numpy as np
import cv2
from random import random
from fluid import Fluid

if __name__ == '__main__':

    fluid = Fluid(128, 0.002, diff = 0.0, visc = 0.0)
    # fluid = Fluid(128, 0.002, diff = 0.0001, visc = 0.000005)

    cx = fluid.size//2
    q1x = cx//2
    q3x = cx + q1x
    w = 4

    t = 0

    fluid.d[:,:] = 127

    while True:
        # fluid.d[cx - w:cx + w, cx - w: cx + w] = 200 + 55*random()
        # fluid.v[cx - w:cx + w, cx - w: cx + w] = np.sin(np.array([t, t + 3.14/2]))*20.0

        # fluid.d[cx - w:cx + w, q1x - w: q1x + w] = 200 + 55*random()
        # fluid.v[cx - w:cx + w, q1x - w: q1x + w] = np.sin(np.array([t, t + 3.14/2]))*20.0
        # fluid.d[cx - w:cx + w, q3x - w: q3x + w] = 200 + 55*random()
        # fluid.v[cx - w:cx + w, q3x - w: q3x + w] = np.sin(np.array([-t, -t - 3.14/2]))*20.0

        fluid.d[cx - w:cx + w, q1x - w: q1x + w] = 200 + 55*random()
        fluid.v[cx - w:cx + w, q1x - w: q1x + w] = [0, 15*random()]
        fluid.d[cx - w:cx + w, q3x - w: q3x + w] = 55*random()
        fluid.v[cx - w:cx + w, q3x - w: q3x + w] = [0, -15*random()]

        fluid.step()

        # fluid.d = np.clip(fluid.d - 1.0, 0, 255)

        t += random()*0.03

        cv2.imshow('dye', cv2.pyrUp(fluid.d.astype('uint8')))
        ch = cv2.waitKey(1)
        if ch == 27: break

    cv2.destroyAllWindows()
