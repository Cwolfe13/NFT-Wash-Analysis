#!/usr/bin/env python3

import unittest
from ipynb.fs.defs.benford import first_sig_fig, make_eth_clusters

"""
This file serves as a testing script using pythons built in library unittest
A simple example is shown below. 
Note: to import the method definition from another notebook the ipynb library must be installed
'pip3 install ipynb', alternatively pipe the requirements.txt into pip.
"""

class TestCluster(unittest.TestCase):
    def test_9(self):
        """
        Tests numbers with 9 because fomo was missing them
        """
        data = [9.0, 99.0, 9.6, 9.4]
        result, tail = make_eth_clusters(data)
        self.assertEqual(result, [9.0, 99, 10, 9])
        
if __name__ == '__main__':
    unittest.main()