from scipy import ndimage
import numpy as np
import cv2
from math import floor

class Fluid():
    def __init__(self, size, dt = 0.1, diff = 0, visc = 0):
        self.vector_field = np.zeros((size, size, 2), dtype = 'float')
        self.vector_field_old = np.zeros_like(self.vector_field)
        self.dye_density = np.zeros((size, size), dtype = 'float')
        self.dye_density_old = np.zeros_like(self.dye_density)


        self.size = size
        self.dt = dt
        self.diff = diff
        self.visc = visc

        self.iter = 10

    def addDensity(self, pos, amount):
        self.dye_density[pos[1], pos[0]] += amount

    def addVelocity(self, pos, amount):
        elem = self.vector_field[pos[1], pos[0]]
        elem[0] += amount[0]
        elem[1] += amount[1]

    def diffuse(self, b, x, x0, diff):
        a = self.dt*diff*(self.size - 2)*(self.size - 2)
        self.lin_solve(b, x, x0, a, 1 + 6*a)# check that this is actually 6*a

    def lin_solve(self, b, x, x0, a, c):
        c_recip = 1.0/c

        kernel1 = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
        ], dtype = 'float')*a*c_recip

        kernel2 = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
        ], dtype = 'float')

        full_kernel = np.dstack((kernel2, kernel1))

        for k in range(self.iter):
            stack = np.dstack((x, x0)).copy()
            # ndimage.convolve(stack, full_kernel, output = stack, mode = 'wrap')#mode = 'constant', cval = 0)



            for j in range(self.size - 1):
                for i in range(self.size - 1):
                    x[i, j] = x0[i, j] + a*(x[i+1, j] + x[i-1, j] + x[i, j+1] + x[i, j-1])/c

            # temp1 = x.copy()
            #
            # x = stack[:,:,0]
            #
            # temp2 = x.copy()
            #
            # print(np.array_equal(temp1, temp2))
            #
            self.set_bound(b, x)

    def project(self, velocities, p, div, p_is_vector_field):
        for j in range(self.size - 1):
            for i in range(self.size - 1):
                div[i, j] = -0.5*(velocities[i+1, j, 0] - velocities[i-1, j, 0] + velocities[i, j+1, 1] - velocities[i, j-1, 1])/self.size
                p[i, j] = 0

        self.set_bound(0, div)
        self.set_bound(0, p)
        self.lin_solve(0, p, div, 1, 6)

        # kernel = 0.5*self.size*np.array([[1, 0, -1],], dtype = 'float')
        #
        # if p_is_vector_field:
        #     kernel[0, 1] = 1.0
        #     ndimage.convolve(p, kernel, output = velocities[:,:,0], mode = 'nearest')
        #     ndimage.convolve(p, kernel.T, output = velocities[:,:,1], mode = 'nearest')
        # else:
        #     velocities[:,:,0] += ndimage.convolve(p, kernel, mode = 'nearest')
        #     velocities[:,:,1] += ndimage.convolve(p, kernel.T, mode = 'nearest')


        for j in range(self.size - 1):
            for i in range(self.size - 1):
                velocities[i, j, 0] += 0.5*(-p[i+1, j] + p[i-1, j])*self.size
                velocities[i, j, 1] += 0.5*(-p[i, j+1] + p[i, j-1])*self.size

        # self.set_bound(1, velocities)
        self.set_bound(1, velocities[:,:,0])
        self.set_bound(2, velocities[:,:,1])

    def advect(self, b, d, d0, velocities):
        dtx = self.dt*(self.size - 2)
        dty = dtx

        indices = np.indices((self.size, self.size))
        x = indices[0] - dtx*velocities[:, :, 0]
        y = indices[1] - dty*velocities[:, :, 1]

        x = x.clip(0.5, self.size + 0.5)
        y = y.clip(0.5, self.size + 0.5)

        i0 = np.floor(x)
        i1 = i0 + 1.0
        j0 = np.floor(y)
        j1 = j0 + 1.0

        s1 = x - i0
        s0 = 1.0 - s1
        t1 = y - j0
        t0 = 1.0 - t1

        i1 = np.floor(i1).astype('int32')
        j1 = np.floor(j1).astype('int32')

        i0 = i0.clip(max = self.size - 1)
        j0 = j0.clip(max = self.size - 1)
        i1 = i1.clip(max = self.size - 1)
        j1 = j1.clip(max = self.size - 1)

        d[indices[0], indices[1]] = s0*(t0*d0[i0.astype('int32'), j0.astype('int32')] + t1*d0[i0.astype('int32'), j1]) + s1*(t0*d0[i1, j0.astype('int32')] + t1*d0[i1, j1])

        self.set_bound(b, d);

    def set_bound(self, b, x):
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
        # for i in range(self.size - 1):
        #     x[i, 0] = -x[i, 1] if b is 2 else x[i, 1]
        #     x[i, self.size - 1] = -x[i, self.size - 2] if b is 2 else x[i, self.size - 2]
        #
        # for j in range(self.size - 1):
        #     x[0, j] = -x[1, j] if b is 2 else x[1, j]
        #     x[self.size - 1, j] = -x[self.size - 2, j] if b is 2 else x[self.size - 2, j]
        #
        # x[0, 0] = 0.5*(x[1, 0] + x[0, 1])
        # x[0, self.size - 1] = 0.5*(x[1, self.size - 1] + x[0, self.size - 2])
        # x[self.size - 1, 0] = 0.5*(x[self.size - 2, 0] + x[self.size - 1, 1])
        # x[self.size - 1, self.size - 1] = 0.5*(x[self.size - 2, self.size - 1] + x[self.size - 1, self.size - 2])

    def step(self):
        self.diffuse(1, self.vector_field_old[:,:,0], self.vector_field[:,:,0], self.visc)
        self.diffuse(2, self.vector_field_old[:,:,1], self.vector_field[:,:,1], self.visc)

        self.project(self.vector_field_old, self.vector_field[:,:,0], self.vector_field[:,:,1], True)

        self.advect(1, self.vector_field[:,:,0], self.vector_field_old[:,:,0], self.vector_field_old)
        self.advect(2, self.vector_field[:,:,1], self.vector_field_old[:,:,1], self.vector_field_old)

        self.project(self.vector_field, self.vector_field_old[:,:,0], self.vector_field_old[:,:,1], False)

        self.diffuse(0, self.dye_density_old, self.dye_density, self.diff)
        self.advect(0, self.dye_density, self.dye_density_old, self.vector_field)


fluid = Fluid(100, diff = 0.01, visc = 1.0)

fluid.dye_density[50, 50] = 10
fluid.vector_field[50, 50, 1] = 100

while True:
    fluid.step()
    print('running')
    vis = fluid.dye_density*255/fluid.dye_density.max()
    cv2.imshow('dye', cv2.pyrUp(vis.astype('uint8')))
    ch = cv2.waitKey(1)
    if ch == 27: break

cv2.destroyAllWindows()
