import numpy as np
from utils import calculate_lambda


def test_lambda():
    """Test that lambda fits characteristic equation."""
    x, y, z = 6, 5, 4
    a, b, c = 3, 2, 1
    lmbda = calculate_lambda(x, y, z, a, b, c)

    # test for lambda is within parameters for an ellipsoid, as suppposed to sh$
    assert lmbda > -(c**2)

    # check lambda fits the characteristic equation
    np.testing.assert_allclose(
        x**2 / (a**2 + lmbda) + y**2 / (b**2 + lmbda) + z**2 / (c**2 + lmbda), 1.0
    )

def test_arccos_for_lambda():
    
    a=3
    b=5
    c=5
    x=10
    y=0
    z=0
    
    p_0 = a**2 * b**2 * c**2 - b**2 * c**2 * x**2 - c**2 * a**2 * y**2 - a**2 * b**2 * z**2
    p_1 = a**2 * b**2 + b**2 * c**2 + c**2 * a**2 - (b**2 + c**2) * x**2 - (c**2 + a**2) * y**2 - (a**2 + b**2) * z**2
    p_2 = a**2 + b**2 + c**2 - x**2 - y**2 - z**2

    p = p_1 - (p_2**2)/3 

    q = p_0 - ((p_1*p_2)/3) + 2*(p_2/3)**3

    theta_internal = -q / (2 * np.sqrt((-p/3)**3))
    
    theta_internal_denominator = (2 * np.sqrt((-p/3)**3))
    #clip to remove floating point precision errors
    #theta_internal_1 = np.clip(theta_internal, -1.0, 1.0)
    
    theta = np.arccos(theta_internal) 
    
    lmbda = 2 * np.sqrt((-p/3)) * np.cos(theta/3) - p_2/3 
    
    assert(q < theta_internal_denominator) #?
    
    return p, q, p_1, p_2, theta_internal

p, q, p_1, p_2, theta_internal = test_arccos_for_lambda()

print(theta_internal)
print('p=', p)
print('q=', q)
print('p_1=', p_1)
print('p_2=', p_2)
print(2 * np.sqrt((-p/3)**3))

# q and the sqrt p**3 term are incredibly similar to the smallest term 
# of precision.