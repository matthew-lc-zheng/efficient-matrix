from timeit import default_timer as timer
import numpy as np

d=100000000
n=20
t=[]

# consttruction
for i in range(n):
    t1=timer()        
    m1=np.ones(d)
    t2=timer()
    t.append(1000*(t2-t1))

# print(t)
print('construction with initialization: %f ms' % np.mean(t))

