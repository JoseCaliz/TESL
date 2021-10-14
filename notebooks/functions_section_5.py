import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted        
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
    
    knots = list()
    def __init__(self, knots:np.array):
        self.knots = knots
        
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
            
        return np.array(results).T