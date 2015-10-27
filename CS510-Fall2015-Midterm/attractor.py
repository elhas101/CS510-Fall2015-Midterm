# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

class Attractor(object):
    
    def __init__(self, s=10.0, p=28.0, b=8.0/3.0, start=0.0, end=80, points=10000):
        self.params = np.array([s, p, b]);
        self.start = start;
        self.end = end;
        self.points = points;
        self.dt = (self.end - self.start)/(self.points);
        self.solution = [];
    
    
    #=======================SET DIFFERENTIAL EQ FUNCTIONS=======================#
    # function for dx(t)/dt =s[y(t)−x(t)]
    def dx(self, x, y):
        return self.dt*self.params[0]*(y - x)
    
    # function for dy(t)/dt =x(t)[p−z(t)]−y(t)
    def dy(self, x, y, z):
        return self.dt*(x*(self.params[1]-z)-y)
    
    # function for dz(t)/dt =x(t)y(t)−bz(t)
    def dz(self, x, y, z):
        return self.dt*(x*y-self.params[2]*z)
    
    #=======================START RK1/Euler===================================#
    # first order step k1 runge-kutta function k1=y(t,x(t)) (i.e. euler function)
    def euler(self, a = np.array([])):
        
        # initialize values of x, y, z
        k1 = {'x': np.zeros(self.points+1, dtype=np.float), 
              'y': np.zeros(self.points+1, dtype=np.float), 
              'z': np.zeros(self.points+1, dtype=np.float)};
        
        k1['x'][0] = a[0]; k1['y'][0] = a[1]; k1['z'][0] = a[2];
        
        # set values of runge-kutta k1 x, y, z
        for i in range(1, self.points):
            k1['x'][i+1] = k1['x'][i] + self.dx(k1['x'][i],k1['y'][i])
            k1['y'][i+1] = k1['y'][i] + self.dy(k1['x'][i],k1['y'][i],k1['z'][i])
            k1['z'][i+1] = k1['z'][i] + self.dz(k1['x'][i],k1['y'][i],k1['z'][i])

        return k1;
    #=======================END RK1/Euler=============================#
    #
    #
    #=======================START RK2=============================#
    # second order step k2 runge-kutta function k2=y(t+Δt/2,x(t)+k1Δt/2)
    def rk2(self, a = np.array([])):
        
        # initialize values of x, y, z
        k2 = {'x': np.zeros(self.points+1, dtype=np.float), 
              'y': np.zeros(self.points+1, dtype=np.float), 
              'z': np.zeros(self.points+1, dtype=np.float)};
        
        k2['x'][0] = a[0]; k2['y'][0] = a[1]; k2['z'][0] = a[2];
        
        # set values of runge-kutta k2 x, y, z
        for i in xrange( self.points ):
            xk1 = self.dx(k2['x'][i],k2['y'][i])
            yk1 = self.dy(k2['x'][i],k2['y'][i],k2['z'][i])
            zk1 = self.dz(k2['x'][i],k2['y'][i],k2['z'][i])

            xk2 = self.dx(k2['x'][i]+xk1*(self.dt/2),k2['y'][i])
            yk2 = self.dy(k2['x'][i],k2['y'][i]+yk1*(self.dt/2),k2['z'][i])
            zk2 = self.dz(k2['x'][i],k2['y'][i],k2['z'][i]+zk1*(self.dt/2))
            
            k2['x'][i+1] = k2['x'][i] + ( xk1 + xk2 ) / 2.0
            k2['y'][i+1] = k2['y'][i] + ( yk1 + yk2 ) / 2.0
            k2['z'][i+1] = k2['z'][i] + ( zk1 + zk2 ) / 2.0

        return k2;
    
    #=======================END RK2=========================================#
    #
    #
    #=======================START RK4=========================================#
    # fourth order step k4 runge-kutta function k4=y(t+Δt,x(t)+k3*Δt)
    def rk4(self, a = np.array([])):
        
        # initialize values of x, y, z on k4
        k4 = {'x': np.zeros(self.points+1, dtype=np.float), 
              'y': np.zeros(self.points+1, dtype=np.float), 
              'z': np.zeros(self.points+1, dtype=np.float)};
        
        k4['x'][0] = a[0]; k4['y'][0] = a[1]; k4['z'][0] = a[2];

        # set values of runge-kutta k4 x, y, z
        for i in xrange( self.points ):
            xk1 = self.dx(k4['x'][i],k4['y'][i])
            yk1 = self.dy(k4['x'][i],k4['y'][i],k4['z'][i])
            zk1 = self.dz(k4['x'][i],k4['y'][i],k4['z'][i])

            xk2 = self.dx(k4['x'][i]+xk1*(self.dt/2),k4['y'][i])
            yk2 = self.dy(k4['x'][i],k4['y'][i]+yk1*(self.dt/2),k4['z'][i])
            zk2 = self.dz(k4['x'][i],k4['y'][i],k4['z'][i]+zk1*(self.dt/2))
            
            xk3 = self.dx(k4['x'][i]+xk2*(self.dt/2),k4['y'][i])
            yk3 = self.dy(k4['x'][i],k4['y'][i]+yk2*(self.dt/2),k4['z'][i])
            zk3 = self.dz(k4['x'][i],k4['y'][i],k4['z'][i]+zk2*(self.dt/2))
            
            xk4 = self.dx(k4['x'][i]+xk3*(self.dt/2),k4['y'][i])
            yk4 = self.dy(k4['x'][i],k4['y'][i]+yk3*(self.dt/2),k4['z'][i])
            zk4 = self.dz(k4['x'][i],k4['y'][i],k4['z'][i]+zk3*(self.dt/2))
            
            
            k4['x'][i+1] = k4['x'][i] + (self.dt/6)*(xk1 + 2*xk2 + 2*xk3 + xk4)
            k4['y'][i+1] = k4['y'][i] + (self.dt/6)*(yk1 + 2*yk2 + 2*yk3 + yk4)
            k4['z'][i+1] = k4['z'][i] + (self.dt/6)*(zk1 + 2*zk2 + 2*zk3 + zk4)

        return k4;
    #=======================END RK4===================================#
    #
    #
    #=======================START EVOLVE METHOD=======================#
    # The evolve method allows to choose the stepping method of runge-kutta (1: Euler, 2: rk2, 4: rk4)
    def evolve(self, r0 = np.array([0.1, 0.0, 0.0]), order=4):
        df = pd.DataFrame();
        
        # check for selected order (default=4)
        if (order==4):
            data = self.rk4(r0);
        elif (order==2):
            data = self.rk2(r0);
        elif (order==1):
            data = self.euler(r0);
        else:
            return 'Error'
        
        # set column header and data
        df['t'] = np.linspace( self.start, self.end, self.points+1);
        df['x'] = data['x'];
        df['y'] = data['y'];
        df['z'] = data['z'];
        
        self.solution = df;
        
        return df;
    
    #=======================END EVOLVE METHOD=======================#
    #
    #
    #=======================START SAVE METHOD=======================#
    def save(self):
        # save into the file rangekutta.csv, setting separators as ',', and ommiting index
        self.solution.to_csv('rungekutta.csv', sep=',', encoding='utf-8', index=0)
    
    #=======================END SAVE METHOD=======================#
    #
    #
    #=======================START PLOT METHODS=======================#
    # plot x(t) versus time
    def plotx(self):
        plotx = self.solution.plot(x='t', y='x', label='x', title='x(t) versus time');
        plt.figure(); plotx;

    # plot y(t) versus time
    def ploty(self):
        ploty = self.solution.plot(x='t', y='y', label='y', title='y(t) versus time');
        plt.figure(); ploty;

    # plot z(t) versus time
    def plotz(self):
        plotz = self.solution.plot(x='t', y='z', label='z', title='z(t) versus time');
        plt.figure(); plotz;
    
    # plot x(t) versus y(t)
    def plotxy(self):
        xy = plt.figure().gca();
        xy.plot(self.solution['x'], self.solution['y']);
        plt.suptitle('Solution x(t) against solution y(t)', fontsize=20)
        plt.xlabel('x(t)', fontsize=18)
        plt.ylabel('y(t)', fontsize=16)
    
    # plot y(t) versus z(t)
    def plotyz(self):
        yz = plt.figure().gca();
        yz.plot(self.solution['y'], self.solution['z']);
        plt.suptitle('Solution y(t) against solution z(t)', fontsize=20)
        plt.xlabel('y(t)', fontsize=18)
        plt.ylabel('z(t)', fontsize=16)
        
    # plot z(t) versus x(t)
    def plotzx(self):
        zx = plt.figure().gca();
        zx.plot(self.solution['z'], self.solution['x']);
        plt.suptitle('Solution z(t) against solution x(t)', fontsize=20)
        plt.xlabel('z(t)', fontsize=18)
        plt.ylabel('x(t)', fontsize=16)
    
    # 3d plot of the x-y-z solution curves
    def plot3d(self):
        fig = plt.figure()
        xyz = fig.add_subplot(111, projection='3d')
        
        tmp_planes = xyz.zaxis._PLANES 
        xyz.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                     tmp_planes[0], tmp_planes[1], 
                     tmp_planes[4], tmp_planes[5])
        
        view_1 = (50, -135)
        view_2 = (50, -45)
        init_view = view_2
        
        xyz.view_init(*init_view);
        
        xyz.plot(self.solution['x'], self.solution['y'], self.solution['z']);
        
        xyz.set_xlabel('x(t)', linespacing=4);
        xyz.set_ylabel('y(t)');
        xyz.set_zlabel('z(t)');
        
        fig.suptitle('3d plot of the x-y-z solution curves', fontsize=20);