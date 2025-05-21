def test_lambda():
    
    """Test that lambda fits characteristic equation."""
    lmbda = calculate_lambda(6, 5, 4, 3, 2, 1)
    
    # test for lambda is within parameters for an ellipsoid, as suppposed to sh$
    if not (lmbda > -c**2):
        raise ValueError(f"Lambda value is invalid: it should be true that lamb$
         f"but instead lambda = {lmbda} and -c^2 = {-c**2}")
    
    # check lambda fits the characteristic equation
    assert np.round(x**2/(a**2 + lmbda) + y**2/(b**2 + lmbda) + z**2/(c**2 + lm$

