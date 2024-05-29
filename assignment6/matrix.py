import numpy as np

X = np.array([[1,1,3],[1,-2,2],[1,0.3,1],[1,5,-1],[1,3,4],[1,7,3]])
XT = np.transpose(X)
y = np.array([1,0,0,1,1,1])

XT_X = np.matmul(XT,X)
XT_y  = np.matmul(XT,y)

inv = np.linalg.inv(XT_X)
coe = np.matmul(inv,XT_y)

y_ = []
for i in X:
    y_.append(round((coe[0]+coe[1]*i[1]+coe[2]*i[2]),3))
print(y_)