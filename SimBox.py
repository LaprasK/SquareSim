import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import aabb
from geometry import test_simple_polygon_overlap
import time
import pickle
import cv2

class Simulation_Box:
    '''
    numParticles: number of particles in the system
    radius: radius of circle wall
    nIteration: number of iteration of simulation
    active_number: number of activer particles in the system
    D0: diffusion constant in v0 direction
    DT: transverse diffusion constant
    DR: rotational diffusion constant
    dim: particle dimension, in our case it's 2
    sidelength: side length of square particle 
    maxdisp: max displacement of a single particle, in unit of side length
    '''
    
    def __init__(self, numParticles, active_number, boxSize, D0, DR, DT, v0, nIteration = 1000, maxdisp = 0.1, 
                 sidelength = 1):
        self.numParticles = numParticles
        self.boxSize = aabb.VectorDouble(2)
        self.boxSize[0] = boxSize
        self.boxSize[1] = boxSize
        self.D0 = D0
        self.DR = DR
        self.DT = DT
        self.v0 = v0
        self.active_number = active_number
        self.nIteration = nIteration
        self.maxdisp = maxdisp
        self.pt_sidelen = sidelength
        self.diagonal_radius = np.sqrt(2)*sidelength*0.5
        self.shuffle_list =  np.arange(numParticles)
        self.posImage = 0.5*boxSize
        self.negImage = -0.5*boxSize
        self.period = aabb.VectorBool(2)
        self.period[0] = True
        self.period[1] = True
        self.dim = 2
        self.success_move = 0
        
        
    def __periodBoundary(self, pos_c):
        for i in range(2):
            if pos_c[i] < 0:
                pos_c[i] += self.boxSize[i]*self.period[i]
            elif pos_c[i] >= self.boxSize[i]:
                pos_c[i] -= self.boxSize[i]*self.period[i]
                
    
    def __printResult(self, filename, positions):
        with open(filename, 'a') as trajectoryFile:
            trajectoryFile.write('%lu\n' % (len(positions)))
            trajectoryFile.write('\n')
            for pos in positions:
                trajectoryFile.write('0 %lf %lf 0\n' % (pos[0], pos[1]))

    
    def init_box(self):
        # create a new AABB Tree
        self.tree = aabb.Tree(2, self.maxdisp, self.period, self.boxSize, self.numParticles)
        self.positions = list()
        self.square_list = list()
        self.orient_list = list()
        self.tags = list()
        self.sq_hist = list()
        active_count = 0
        pos_c = aabb.VectorDouble(self.dim)
        lower_bound = aabb.VectorDouble(self.dim)
        upper_bound = aabb.VectorDouble(self.dim)
        
        for i in range(self.numParticles):
            if i == 0:
                pos_c[0] = self.boxSize[0] * np.random.rand()
                pos_c[1] = self.boxSize[1] * np.random.rand()
                
                orientation = 2*np.pi*np.random.rand()
                
                temp_square = Square(pos_c, orientation, self.pt_sidelen)
            else:
                overlap = True
        
                while overlap:
                    #random position in circle wall
                    pos_c[0] = self.boxSize[0] * np.random.rand()
                    pos_c[1] = self.boxSize[1] * np.random.rand()
                    
                    orientation = 2*np.pi*np.random.rand()
                    
                    temp_square = Square(pos_c, orientation, self.pt_sidelen)
                    
                    #lowerbound and upperbound of a single square
                    for j in range(2):
                        lower_bound[j] = pos_c[j] - self.diagonal_radius
                        upper_bound[j] = pos_c[j] + self.diagonal_radius
                    
                    AABB = aabb.AABB(lower_bound, upper_bound)
                    
                    particles = self.tree.query(AABB)
                    
                    overlap = False
                    
                    for idx in particles:
                        if(test_simple_polygon_overlap(self.square_list[idx], temp_square, self.boxSize[0])):
                            overlap = True
                            break   
            self.tree.insertParticle(i, pos_c, self.diagonal_radius)
            self.square_list.append(temp_square)
            self.orient_list.append(orientation)
            tag = 0 if active_count < self.active_number else 1
            self.tags.append(tag)
            active_count+=1
            self.positions.append([pos_c[0], pos_c[1]])
        self.sq_hist.append(self.square_list)
        return
    
    
    def dynamics(self):
        pos_c = aabb.VectorDouble(2)
        lowerBound = aabb.VectorDouble(2)
        upperBound = aabb.VectorDouble(2)
        for _ in range(self.nIteration):
            np.random.shuffle(self.shuffle_list)
            for pt_idx in self.shuffle_list:
                pt_tag = self.tags[pt_idx]
                curr_orient = self.orient_list[pt_idx]
                
                
                #assign mean value to 
                if pt_tag == 0: #active particles
                    mu = self.v0
                elif pt_tag == 1: #brownian squares
                    mu = 0
                
                rotation_disp = np.random.randn()*self.DR
                disp_0 = (mu + np.random.randn() * self.D0)*self.pt_sidelen
                disp_t = np.random.randn()*self.DT*self.pt_sidelen
                
                disp_x = disp_0 * np.cos(curr_orient) - disp_t * np.sin(curr_orient)
                disp_y = disp_0 * np.sin(curr_orient) + disp_t * np.cos(curr_orient)
                
                pos_c[0] = self.positions[pt_idx][0] + disp_x
                pos_c[1] = self.positions[pt_idx][1] + disp_y

                self.__periodBoundary(pos_c)
                
                temp_square = Square(pos_c, curr_orient+rotation_disp, self.pt_sidelen)
                
                for i in range(2):
                    lowerBound[i] = pos_c[i] - self.diagonal_radius
                    upperBound[i] = pos_c[i] + self.diagonal_radius
                    
                AABB = aabb.AABB(lowerBound, upperBound)
                
                particles = self.tree.query(AABB)
                
                overlap = False
                
                for idx in particles:
                    if idx != pt_idx:
                        if test_simple_polygon_overlap(temp_square, self.square_list[idx], self.boxSize[0]):
                            overlap = True
                            break

                
                if not overlap:
                    self.success_move +=1
                    self.positions[pt_idx] = [pos_c[0], pos_c[1]]
                    self.square_list[pt_idx] = temp_square
                    self.orient_list[pt_idx] = curr_orient + rotation
                    self.tree.updateParticle(int(pt_idx), lowerBound, upperBound)
            
            self.sq_hist.append(self.square_list)
                    
            
    def get_snapshot(self):
        fig, ax = plt.subplots(figsize = (12,12))
        patches = []

        for square in self.square_list:
            polygon = Polygon(square.get_vertices(), True)
            patches.append(polygon)

        p = PatchCollection(patches, alpha=0.4)

        #colors = 100*np.random.rand(len(patches))
        #p.set_array(np.array(colors))

        ax.add_collection(p)
        ax.set_xlim([-1,box.boxSize[0]+1])
        ax.set_ylim([-1,box.boxSize[1]+1])

        plt.show()
        
    
    def save_result(self):
        with open('sq_hist.pkl', 'wb') as output:
            pickle.dump(self.sq_hist, output, pickle.HIGHEST_PROTOCOL)
                
                
    def generate_gif(self, start = 0, step = 1):
        return

                
            

