import numpy as np

if __name__ == '__main__':
    tmp = [[1,1.5,1],[1,2.4,1]]
    temp = np.asarray(tmp)
    np.savetxt("temp", temp, fmt='%1.1f',delimiter="\t")