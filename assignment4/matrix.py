import numpy as np

X = np.array([[1,1,1],[1,4,1],[1,6,0],[1,8,2],[1,10,1]])
XT = np.transpose(X)

y = np.array([52,63,62,91,75])

XT_X = np.matmul(XT,X)
XT_y  = np.matmul(XT,y)

inv = np.linalg.inv(XT_X)

coe = np.matmul(inv,XT_y)

print(coe)