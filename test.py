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

    fx = ct.nodal_diff(f)

    fig = plt.figure(1,(12,6))
    ax1 = fig.add_subplot(121)
    ax1.plot(x,Df,lw=2)
    ax1.set_title('Modal differentiation')

    ax2 = fig.add_subplot(122)
    ax2.plot(x,fx,lw=2)
    ax2.set_title('Nodal differentiation')
   

    plt.show()
    
 
     

    
