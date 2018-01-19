import numpy as np
import kernel as kernel

from params import par
for key,val in par.items():
    exec(key + '=val')

def find_density(position,ppositions,smoothingL,kernel=kernel.kernel_gaussian):
    total   =   0
    for j in range(N):
        total   +=  kernel(position-ppositions[j,:],smoothingL)
    return m * total
