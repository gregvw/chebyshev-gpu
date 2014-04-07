import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebder
from chebtran_cl import ChebTran

if __name__ == '__main__':

    n = 1024
    e1 = np.zeros(n)
    e1[1] = 1
 
    ct = ChebTran(n)
    s = ct.get_device_name()

    print('Accessing device: ' + s)

    # Chebyshev spectral differentiation

    x = ct.coeff_to_nodal(e1)

    f = np.exp(np.cos(np.pi*x))

    fhat = ct.nodal_to_coeff(f)

    Dfhat = np.zeros(n)
    Dfhat[:-1] = chebder(fhat)
    Df = ct.coeff_to_nodal(Dfhat)

    plt.plot(x,Df,lw=2)
    plt.show()
    
 
     

    
