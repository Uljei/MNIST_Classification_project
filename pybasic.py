import time
from matplotlib.pyplot import *

def fibonacci1(n):
    if n==0:
        return 0
    elif n==1:
        return 1
    else:
        return fibonacci1(n-1)+fibonacci1(n-2)




if __name__ == "__main__":
    start_time = time.time()
    fibonacci1(100)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("elapsed time: ", elapsed_time)
