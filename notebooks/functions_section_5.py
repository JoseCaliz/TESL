import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted        
from scipy.optimize import minimize
from collections.abc import Iterable

def validate_X_for_spline(X) -> np.ndarray:
    '''
    Validates if X and cast (if necessary) X as numpy array
    '''
    
    if not isinstance(X, Iterable):
        X = np.array([X])   
    elif not isinstance(X, np.ndarray) :
        X = np.array(X)
    else:
        pass
    return X

def piece_linear_basis(X, knot) -> np.ndarray:
    '''
    Function to return Picewise-linear basis to the power of 3
    '''
    
    return np.where(X >= knot, (X-knot)**3, 0)


class global_linear_base():
    '''
    Global Linear Base that returns basis matrix
    where each row is composed of [1, X]
    '''
    
    def generate_basis_matrix(self, X) -> np.ndarray:
        X = validate_X_for_spline(X)
        results = [np.array([1]*X.shape[0]), X]
        return np.array(results).T


class global_polinomial_base():
    '''
    Global Polinomial Base that returns basis matrix
    where each row is composed of [1, X, X**2, X**3]
    '''
    
    def generate_basis_matrix(self, X) -> np.ndarray:
        X = validate_X_for_spline(X)
        results = [np.array([1]*X.shape[0]), X, X**2, X**3]
        return np.array(results).T

class cubic_spline_base():
    '''
    Cubic Spline Base that returns basis matrix
    where each row is composed of:
    [1, X, X**2, X**3, piece_linear_basis(knot_0), ..., piece_linear_basis(knot_k)]
    '''
    
    knots = list()
    def __init__(self, knots:np.array):
        self.knots = knots
            
    def generate_basis_matrix(self, X) -> np.ndarray:
        X = validate_X_for_spline(X)
        
        results = [np.array([1]*X.shape[0]), X, X**2, X**3]
        for knot in self.knots:
            results.append(piece_linear_basis(X, knot))
            
        return np.array(results).T

class natural_cubic_spline_base():
    '''
    Cubic Spline Base that returns basis matrix
    where each row is composed of:
    [1, X, X**2, X**3, d_{k_0}(X) - d_{k_{K-1}}(X), ..., d_{k_{K-2}}(X) - d_{k_{K-1}}(X)]
    '''
    
    def integral_result(self, zeta_i, zeta_j, zeta_K_1, zeta_K):
        A = max(zeta_j, zeta_i)
        return(
            -6*(
                2*A**3 - 3*A**2*zeta_i - 3*A**2*zeta_j +
                6*A*zeta_i*zeta_j - 2*zeta_K*zeta_i*zeta_j +
                2*zeta_K*zeta_i*zeta_K_1 + 2*zeta_K*zeta_j*zeta_K_1 -
                2*zeta_K*zeta_K_1**2 - 4*zeta_i*zeta_j*zeta_K_1+
                zeta_i*zeta_K_1**2 + zeta_j*zeta_K_1**2
            )/(
                (zeta_K - zeta_i)*(zeta_K-zeta_j)
            )
        )
    
    def __init__(self, knots:np.array):
        self.knots = knots
        self.basis_matrix = None
        self.omega = None
        
    def d_k(self, X, counter):
        return  (
                piece_linear_basis(X, self.knots[counter]) -
                piece_linear_basis(X, self.knots[-1])
            ) / (self.knots[-1] - self.knots[counter])
        
    def generate_basis_matrix(self, X) -> np.ndarray:
        X = validate_X_for_spline(X)
        
        results = [np.array([1]*X.shape[0]), X]
        knot_shape = self.knots.shape[0]
        for i, knot in enumerate(self.knots[:-2]):
            results.append(self.d_k(X, i) - self.d_k(X, -2))
            
        self.basis_matrix = np.array(results).T
        return np.array(results).T
    
    def generate_omega_matrix(self, X) -> np.ndarray:
        
        omega = np.zeros((len(X), len(X)))
        omega[:2, :] = 0
        omega[:, :2] = 0
        
        for i in range(len(X)-2):
            for j in range(len(X)-2):
                omega[i+2, j+2] = self.integral_result(
                    self.knots[i], self.knots[j], self.knots[-2], self.knots[-1]
                )
                
        self.omega = omega
        return omega