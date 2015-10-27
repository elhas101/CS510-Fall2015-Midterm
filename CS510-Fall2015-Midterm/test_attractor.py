# -*- coding: utf-8 -*-

import unittest
import attractor as att
 
 
class TddInPythonExample(unittest.TestCase):
 
    def setUp(self):
        self.Attractor = att.Attractor();
     
    # test function for dx(t)/dt =s[y(t)−x(t)]
    def test_dx(self):
        result = self.Attractor.dx(2, 1)
        self.assertEqual(self.Attractor.dt*self.Attractor.params[0]*(1-2), result)
        
    # test function for dy(t)/dt =x(t)[p−z(t)]−y(t)
    def test_dy(self):
        result = self.Attractor.dy(2, 1, 2)
        self.assertEqual(self.Attractor.dt*(2*(self.Attractor.params[1]-2)-1), result)
        
    # test function for dz(t)/dt =x(t)y(t)−bz(t)
    def test_dz(self):
        result = self.Attractor.dz(1, 2, 3)
        self.assertEqual(self.Attractor.dt*(1*2-self.Attractor.params[2]*3), result)
 
    # test the length of the returned x-y-z arrays in euler to make sure all arrays have different length
    def test_euler(self):
        result = len(self.Attractor.euler([1, 1, 1])['x']);
        result2 = len(self.Attractor.euler([1, 1, 1])['y']);
        result3 = len(self.Attractor.euler([1, 1, 1])['z']);
        self.assertEqual(self.Attractor.points+1, result);
        self.assertEqual(self.Attractor.points+1, result);
        self.assertEqual(self.Attractor.points+1, result2);
        self.assertEqual(self.Attractor.points+1, result3);

    # test the length of the returned x-y-z arrays in rk2 to make sure all arrays have different length
    def test_rk2(self):
        result = len(self.Attractor.rk2([1, 1, 1])['x']);
        result2 = len(self.Attractor.rk2([1, 1, 1])['y']);
        result3 = len(self.Attractor.rk2([1, 1, 1])['z']);
        self.assertEqual(self.Attractor.points+1, result);
        self.assertEqual(self.Attractor.points+1, result);
        self.assertEqual(self.Attractor.points+1, result2);
        self.assertEqual(self.Attractor.points+1, result3);
    
    # test the length of the returned x-y-z arrays in rk4 to make sure all arrays have different length
    def test_rk4(self):
        result = len(self.Attractor.rk4([1, 1, 1])['x']);
        result2 = len(self.Attractor.rk4([1, 1, 1])['y']);
        result3 = len(self.Attractor.rk4([1, 1, 1])['z']);
        self.assertEqual(self.Attractor.points+1, result);
        self.assertEqual(self.Attractor.points+1, result);
        self.assertEqual(self.Attractor.points+1, result2);
        self.assertEqual(self.Attractor.points+1, result3);
        
if __name__ == '__main__':
    unittest.main()