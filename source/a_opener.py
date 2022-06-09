import numpy as np

def a_opener(filename):
    depth, ar, ag, ab = np.genfromtxt(filename, delimiter=",", skip_header=1).T
    return depth, ar, ag, ab

if __name__ == "__main__":
    # Oden
    # depth, ar, ag, ab = a_opener("a_oden.csv")
    # print(np.mean(ar[2:]), np.mean(ag[2:]), np.mean(ab[2:]))
    # 0.1543711501111111 0.08106681833333333 0.08158018577777779

    # Station 2 BDC
    depth, ar, ag, ab = a_opener("a_bdc_st2.csv")
    print(np.mean(ar[4:]), np.mean(ag[4:]), np.mean(ab[4:]))