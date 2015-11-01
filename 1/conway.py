################################################################################
# conway.py
#
# Author: electronut.in
# 
# Description:
#
# A simple Python/matplotlib implementation of Conway's Game of Life.
################################################################################

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

N = 100
ON = 255
OFF = 0
vals = [ON, OFF]

# populate grid with random on/off - more off than on
grid = np.random.choice(vals, N*N, p=[0.2, 0.8]).reshape(N, N)

def update(data):
  global grid
  # copy grid since we require 8 neighbors for calculation
  # and we go line by line 
  newGrid = grid.copy()
  sums = np.zeros_like(grid)
  for y in (-1,0,1):
    for x in (-1,0,1):
      if not(y==0 and x==0):
        sums =sums+ np.roll(np.roll(grid,y,1),x,0)
  sums=sums/255
  newGrid[np.logical_and((grid==ON), np.logical_or(sums<2, sums>3))]=OFF
  newGrid[np.logical_and(grid==OFF,sums==3)]=ON
  # update data
  mat.set_data(newGrid)
  grid = newGrid
  return [mat]

# set up animation
fig, ax = plt.subplots()
mat = ax.matshow(grid)
ani = animation.FuncAnimation(fig, update, interval=50,
                              save_count=50)
plt.show()
