#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:29:33 2019

@author: gabriel
"""

import pygraphviz as pgv

G = pgv.AGraph(directed = True, strict = False)

G.add_node('a', label = "X[0] < 2")
G.add_edge('a', 'b', label = "True")
G.add_edge('a', 'c', label = "False")

G.write('bob.dot')
G.draw('bob.png', prog = 'circo')