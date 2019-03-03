"""
https://thecodingtrain.com/CodingChallenges/132-fluid-simulation.html
http://www.dgp.toronto.edu/people/stam/reality/Research/pdf/GDC03.pdf
https://mikeash.com/pyblog/fluid-simulation-for-dummies.html
"""

import numpy as np
from scipy import ndimage

from math import floor, copysign
from random import random

class Fluid():
    iter = 4

    def __init__(self, size, dt, diff, visc):
        self.size = size
        self.dt = dt
        self.diff = diff
        self.visc = visc

        self.vector_field = np.zeros((self.size, self.size, 2), dtype = 'float')
        self.vector_field_old = np.zeros_like(self.vector_field)
        self.dye_density = np.zeros((self.size, self.size), dtype = 'float')
        self.dye_density_old = np.zeros_like(self.dye_density)

    """converted and checked"""
    @staticmethod
    def set_bound(b, x):
        if b is 2:
            x[:, 0] = -x[:, 1]
            x[:, -1] = -x[:, -2]
        else:
            x[:, 0] = x[:, 1]
            x[:, -1] = x[:, -2]

        if b is 1:
            x[0, :] = -x[1, :]
            x[-1, :] = -x[-2, :]
        else:
            x[0, :] = x[1, :]
            x[-1, :] = x[-2, :]

        x[0, 0] = (x[1, 0] + x[0, 1])/2.0
        x[0, -1] = (x[1, -1] + x[0, -2])/2.0
        x[-1, 0] = (x[-2, 0] + x[-1, 1])/2.0
        x[-1, -1] = (x[-2, -1] + x[-1, -2])/2.0

    """converted and checked"""
    @staticmethod
    def lin_solve(b, x, x0, a, c):
        kernel1 = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
        ], dtype = 'float')*a/c
        kernel2 = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
        ], dtype = 'float')
        full_kernel = np.dstack((kernel2, kernel1))

        stack = np.dstack((x, x0))

        for k in range(Fluid.iter):
            ndimage.convolve(stack, full_kernel, output = stack, mode = 'wrap')
            stack[:,:,1] = x0
            Fluid.set_bound(b, stack[:,:,0])

        return stack[:,:,0]

    """converted and checked"""
    def advect(self, b, d, d0, velocities):
        dtx = self.dt*(self.size - 2)
        dty = self.dt*(self.size - 2)

        indices = np.indices((self.size, self.size))
        x = indices[0] - dtx*velocities[:, :, 0]
        y = indices[1] - dty*velocities[:, :, 1]

        x.clip(0.5, self.size + 0.5, out = x)
        y.clip(0.5, self.size + 0.5, out = y)

        i0 = np.floor(x)
        j0 = np.floor(y)
        i1 = i0 + 1.0
        j1 = j0 + 1.0

        s1 = x - i0
        t1 = y - j0
        s0 = 1.0 - s1
        t0 = 1.0 - t1

        i0 = i0.astype('int32', copy = False)
        i1 = i1.astype('int32', copy = False)
        j0 = j0.astype('int32', copy = False)
        j1 = j1.astype('int32', copy = False)

        i0.clip(0, self.size - 1, out = i0)
        i1.clip(0, self.size - 1, out = i1)
        j0.clip(0, self.size - 1, out = j0)
        j1.clip(0, self.size - 1, out = j1)

        d[indices[0], indices[1]] = s0*(t0*d0[i0, j0] + t1*d0[i0, j1]) + s1*(t0*d0[i1, j0] + t1*d0[i1, j1])

        Fluid.set_bound(b, d);
        return d

    """converted and checked"""
    def project(self, velocities, p, div):
        # velocities always points somewhere other than p and div
        # but p and div may point to the same array

        kernel1 = np.array([
        [0, -1, 0],
        [0, 0, 0],
        [0, 1, 0]
        ], dtype = 'float')
        kernel2 = np.array([
        [0, 0, 0],
        [-1, 0, 1],
        [0, 0, 0]
        ], dtype = 'float')
        full_kernel = np.dstack((kernel2, kernel1))

        div = ndimage.convolve(velocities, full_kernel*0.5/self.size, mode = 'wrap')[:,:,0]
        p[:,:] = 0

        Fluid.set_bound(0, div)
        Fluid.set_bound(0, p)
        p = Fluid.lin_solve(0, p, div, 1, 8.0)# should be 4, but 6 works better

        velocities[:,:,0] += ndimage.convolve(p, kernel1*0.5*self.size, mode = 'wrap')
        velocities[:,:,1] += ndimage.convolve(p, kernel2*0.5*self.size, mode = 'wrap')

        Fluid.set_bound(1, velocities[:,:,0])
        Fluid.set_bound(2, velocities[:,:,1])

        return p


    def step(self):
        a = self.dt*self.visc*(self.size - 2)*(self.size - 2)
        # diffuse x velocities
        self.vector_field_old[:,:,0] = Fluid.lin_solve(1, self.vector_field_old[:,:,0], self.vector_field[:,:,0], a, 1 + 4*a)
        # diffuse y velocities
        self.vector_field_old[:,:,1] = Fluid.lin_solve(2, self.vector_field_old[:,:,1], self.vector_field[:,:,1], a, 1 + 4*a)

        # project 1?
        self.vector_field[:,:,0] = self.project(self.vector_field_old, self.vector_field[:,:,0], self.vector_field[:,:,1])

        # advect x velocities
        self.vector_field[:,:,0] = self.advect(1, self.vector_field[:,:,0], self.vector_field_old[:,:,0], self.vector_field_old)
        # advect y velocities
        self.vector_field[:,:,1] = self.advect(2, self.vector_field[:,:,1], self.vector_field_old[:,:,1], self.vector_field_old)

        # project 2?
        self.vector_field_old[:,:,0] = self.project(self.vector_field, self.vector_field_old[:,:,0], self.vector_field_old[:,:,1])

        a = self.dt*self.diff*(self.size - 2)*(self.size - 2)
        # diffuse dye
        self.dye_density_old = Fluid.lin_solve(0, self.dye_density_old, self.dye_density, a, 1 + 4*a)
        # advect
        self.dye_density = self.advect(0, self.dye_density, self.dye_density_old, self.vector_field)

import cv2

fluid = Fluid(128, 0.002, diff = 0.0, visc = 0.0)
# fluid = Fluid(128, 0.002, diff = 0.0001, visc = 0.000005)
cx = fluid.size//2
w = 4

t = 0

while True:
    fluid.dye_density[cx - w:cx + w, cx - w: cx + w] = 200 + 55*random()
    fluid.vector_field[cx - w:cx + w, cx - w: cx + w] = np.sin(np.array([t, t + 3.14/2]))*20.0#np.random.rand(2)*2.0 - 1.0
    fluid.step()

    fluid.dye_density = np.clip(fluid.dye_density - 1.0, 0, 255)

    t += random()*0.05

    # print(fluid.vector_field.max(), fluid.dye_density.max())
    vis = (fluid.dye_density).astype('uint8')

    cv2.imshow('dye', cv2.pyrUp(vis))
    ch = cv2.waitKey(1)
    if ch == 27: break

cv2.destroyAllWindows()
