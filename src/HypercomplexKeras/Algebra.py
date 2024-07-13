#! /usr/bin/env python

import numpy as np

class StructureConstants:
    """Class that creates structure constants based on dictionary that represents
    multiplication table of an algebra. 
    The multiplication table e1 x e2 = e3 is converted to the dictionary entry:
    d = {(1,2):(3,1)}. Implicit assumption is that the unit of the multiplication 
    occurs at index 0 although it may be modified.
    """
    
    def __init__(self, multiplication_dict, has_unit_at_0_index = True, dim = 0):
        """Creates multiplication table for algebra based on multipliplication dictionary.
        Inputs: 
            multiplication_dict - dictionary representing multiplication table of the algebra:
            The multiplication table e1 x e2 = e3 is converted to the dictionary entry:
            d = {(1,2):(3,1)}.
            
            has_unit_at_0_index - if there is the multiplication unit at index 0, i.e., e0 is the unit
            By default it is True and then you do not have to provide 0th index 
            part of the multiplication table. If set to False you have to provide 
            all multiplication table.
          
            dim - dimension of the algebra. If it is set to 0 then the dimenison
            is deduced from the maximal value of indexes in multiplication_dict
        """
    
        #estimate dimension of algebra
        assert(dim >= 0)
        # if dim =0 then we must ourselves find dimension of algebra
        self.dim = dim
        if dim == 0:
            for i,j in multiplication_dict.keys():
                if i > self.dim:
                    self.dim = i
                if j > self.dim:
                    self.dim = j
            self.dim = self.dim + 1 # we start from index 0
        
        
        # create array
        self.A = np.zeros((self.dim, self.dim, self.dim))
        
        #initialize multiplication by the unit 
        if has_unit_at_0_index == True:
            for i in range(self.dim):
                self.A[0,i,i] = 1
                self.A[i,0,i] = 1
                
        # rest is initialized from dictionary
        for i,j in multiplication_dict.keys():
            c = multiplication_dict[(i,j)]
            self.A[i,j,c[0]] = c[1]
            
    def getA(self):
        """ Returns multiplication table. If ei x ej = c ek, then A[i,j,k]==c
        """
        return(self.A.copy())
    
    def Mult(self,q1, q2):
        """Returns multiplication of the vectors q1 x q2"""
        return( q1.dot(q2.dot(self.A)) )
    
    def getDim(self):
        """Returns dimension of of algebra"""
        return self.dim
    
    

#Complex numbers

Complex_dict = {(1,1):(0,-1)}
Complex = StructureConstants(Complex_dict )

Q_dict={(1,1):(0,-1),(1,2):(3,1),(1,3):(2,-1),(2,1):(3,-1),(2,2):(0,-1),(2,3):(1,1),(3,1):(2,1),(3,2):(1,-1),(3,3):(0,-1)}
Quaternions = StructureConstants(Q_dict)

Klein4_dict={(1,1):(0,1),(1,2):(3,1),(1,3):(2,1),(2,1):(3,1),(2,2):(0,1),(2,3):(1,1),(3,1):(2,1),(3,2):(1,1),(3,3):(0,1)}
Klein4 = StructureConstants(Klein4_dict)

Cl20_dict={(1,1):(0,1),(1,2):(3,1),(1,3):(2,1),(2,1):(3,-1),(2,2):(0,1),(2,3):(1,-1),(3,1):(2,-1),(3,2):(1,1),(3,3):(0,-1)}
Cl20 = StructureConstants(Cl20_dict)

Coquaternions_dict={(1,1):(0,-1),(1,2):(3,1),(1,3):(2,-1),(2,1):(3,-1),(2,2):(0,1),(2,3):(1,-1),(3,1):(2,1),(3,2):(1,1),(3,3):(0,1)}
Coquaternions = StructureConstants(Coquaternions_dict)

Cl11_dict={(1,1):(0,1),(1,2):(3,1),(1,3):(2,1),(2,1):(3,-1),(2,2):(0,-1),(2,3):(1,1),(3,1):(2,-1),(3,2):(1,-1),(3,3):(0,1)}
Cl11 = StructureConstants(Cl11_dict)


Bicomplex_dict={(1,1):(0,-1),(1,2):(3,1),(1,3):(2,-1),(2,1):(3,1),(2,2):(0,-1),(2,3):(1,-1),(3,1):(2,-1),(3,2):(1,-1),(3,3):(0,1)}
Bicomplex = StructureConstants(Bicomplex_dict)

Tessarines_dict={(1,1):(0,-1),(1,2):(3,1),(1,3):(2,-1),(2,1):(3,1),(2,2):(0,1),(2,3):(1,1),(3,1):(2,-1),(3,2):(1,1),(3,3):(0,-1)}
Tessarines = StructureConstants(Tessarines_dict)

Octonions_dict={(1,1):(0,-1),(1,2):(3,1),(1,3):(2,-1),(1,4):(5,1),(1,5):(4,-1),(1,6):(7,-1),(1,7):(6,1),
                (2,1):(3,-1),(2,2):(0,-1),(2,3):(1,1),(2,4):(6,1),(2,5):(7,1),(2,6):(4,-1),(2,7):(5,-1),
                (3,1):(2,1),(3,2):(1,-1),(3,3):(0,-1),(3,4):(7,1),(3,5):(6,-1),(3,6):(5,1),(3,7):(4,-1),
                (4,1):(5,-1),(4,2):(6,-1),(4,3):(7,-1),(4,4):(0,-1),(4,5):(1,1),(4,6):(2,1),(4,7):(3,1),
                (5,1):(4,1),(5,2):(7,-1),(5,3):(6,1),(5,4):(1,-1),(5,5):(0,-1),(5,6):(3,-1),(5,7):(2,1),
                (6,1):(7,1),(6,2):(4,1),(6,3):(5,-1),(6,4):(2,-1),(6,5):(3,1),(6,6):(0,-1),(6,7):(1,-1),
                (7,1):(6,-1),(7,2):(5,1),(7,3):(4,1),(7,4):(3,-1),(7,5):(2,-1),(7,6):(1,1),(7,7):(0,-1)}
Octonions = StructureConstants(Octonions_dict)


def test_Complex():
    """
    Test simple operations in algebra.

    Returns
    -------
    None.

    """
    assert (Complex.Mult(np.array([0,1]), np.array([0,1])) == [-1,0]).all() #i x i = -1
    assert (Complex.Mult(np.array([1,0]), np.array([1,0])) == [1,0]).all()  #1 x 1 = 1
    assert (Complex.Mult(np.array([1,0]), np.array([0,1])) == [0,1]).all()  #1 x i = i

def test_Quaternins():
    """
    Test simple operations in algebra.

    Returns
    -------
    None.

    """
    assert (Quaternions.Mult(np.array([1,0,0,0]), np.array([1,1,1,1])) == [1,1,1,1]).all()   # 1 x 1+i+j+k
    assert (Quaternions.Mult(np.array([0,1,0,0]), np.array([1,1,1,1])) == [-1,1,-1,1]).all() # i x 1+i+j+k
    assert (Quaternions.Mult(np.array([0,0,1,0]), np.array([1,1,1,1])) == [-1,1,1,-1]).all() # j x 1+i+j+k
    assert (Quaternions.Mult(np.array([0,0,0,1]), np.array([1,1,1,1])) == [-1,-1,1,1]).all() # k x 1+i+j+k
    assert (Quaternions.Mult(np.array([1,0,0,0]), np.array([1,0,0,0])) == [1,0,0,0]).all() # 1 x 1
    assert (Quaternions.Mult(np.array([0,1,0,0]), np.array([0,1,0,0])) == [-1,0,0,0]).all() # i x i
    assert (Quaternions.Mult(np.array([0,0,1,0]), np.array([0,0,1,0])) == [-1,0,0,0]).all() # j x j
    assert (Quaternions.Mult(np.array([0,0,0,1]), np.array([0,0,0,1])) == [-1,0,0,0]).all() # k x k
    assert (Quaternions.Mult(np.array([0,1,0,0]), np.array([0,0,1,0])) == [0,0,0,1]).all() # i x j == k
    assert (Quaternions.Mult(np.array([0,0,1,0]), np.array([0,1,0,0])) == [0,0,0,-1]).all() # jxi == -k

def test_Klein4():
    """
    Test simple operations in algebra.

    Returns
    -------
    None.

    """
    assert (Klein4.Mult(np.array([1,0,0,0]), np.array([1,1,1,1])) == [1,1,1,1]).all()   # 1 x 1+i+j+k
    assert (Klein4.Mult(np.array([0,1,0,0]), np.array([1,1,1,1])) == [1,1,1,1]).all() # i x 1+i+j+k
    assert (Klein4.Mult(np.array([0,0,1,0]), np.array([1,1,1,1])) == [1,1,1,1]).all() # j x 1+i+j+k
    assert (Klein4.Mult(np.array([0,0,0,1]), np.array([1,1,1,1])) == [1,1,1,1]).all() # k x 1+i+j+k
    assert (Klein4.Mult(np.array([1,0,0,0]), np.array([1,0,0,0])) == [1,0,0,0]).all() # 1 x 1
    assert (Klein4.Mult(np.array([0,1,0,0]), np.array([0,1,0,0])) == [1,0,0,0]).all() # i x i
    assert (Klein4.Mult(np.array([0,0,1,0]), np.array([0,0,1,0])) == [1,0,0,0]).all() # j x j
    assert (Klein4.Mult(np.array([0,0,0,1]), np.array([0,0,0,1])) == [1,0,0,0]).all() # k x k
    assert (Klein4.Mult(np.array([0,1,0,0]), np.array([0,0,1,0])) == [0,0,0,1]).all() # i x j 
    assert (Klein4.Mult(np.array([0,0,1,0]), np.array([0,1,0,0])) == [0,0,0,1]).all() # jxi

def test_Cl20():
    """
    Test simple operations in algebra.

    Returns
    -------
    None.

    """
    assert (Cl20.Mult(np.array([1,0,0,0]), np.array([1,1,1,1])) == [1,1,1,1]).all()   # 1 x 1+i+j+k
    assert (Cl20.Mult(np.array([0,1,0,0]), np.array([1,1,1,1])) == [1,1,1,1]).all() # i x 1+i+j+k
    assert (Cl20.Mult(np.array([0,0,1,0]), np.array([1,1,1,1])) == [1,-1,1,-1]).all() # j x 1+i+j+k
    assert (Cl20.Mult(np.array([0,0,0,1]), np.array([1,1,1,1])) == [-1,1,-1,1]).all() # k x 1+i+j+k
    assert (Cl20.Mult(np.array([1,0,0,0]), np.array([1,0,0,0])) == [1,0,0,0]).all() # 1 x 1
    assert (Cl20.Mult(np.array([0,1,0,0]), np.array([0,1,0,0])) == [1,0,0,0]).all() # i x i
    assert (Cl20.Mult(np.array([0,0,1,0]), np.array([0,0,1,0])) == [1,0,0,0]).all() # j x j
    assert (Cl20.Mult(np.array([0,0,0,1]), np.array([0,0,0,1])) == [-1,0,0,0]).all() # k x k
    assert (Cl20.Mult(np.array([0,1,0,0]), np.array([0,0,1,0])) == [0,0,0,1]).all() # i x j 
    assert (Cl20.Mult(np.array([0,0,1,0]), np.array([0,1,0,0])) == [0,0,0,-1]).all() # jxi


def test_Coquaternions():
    """
    Test simple operations in algebra.

    Returns
    -------
    None.

    """
    assert (Coquaternions.Mult(np.array([1,0,0,0]), np.array([1,1,1,1])) == [1,1,1,1]).all()   # 1 x 1+i+j+k
    assert (Coquaternions.Mult(np.array([0,1,0,0]), np.array([1,1,1,1])) == [-1,1,-1,1]).all() # i x 1+i+j+k
    assert (Coquaternions.Mult(np.array([0,0,1,0]), np.array([1,1,1,1])) == [1,-1,1,-1]).all() # j x 1+i+j+k
    assert (Coquaternions.Mult(np.array([0,0,0,1]), np.array([1,1,1,1])) == [1,1,1,1]).all() # k x 1+i+j+k
    assert (Coquaternions.Mult(np.array([1,0,0,0]), np.array([1,0,0,0])) == [1,0,0,0]).all() # 1 x 1
    assert (Coquaternions.Mult(np.array([0,1,0,0]), np.array([0,1,0,0])) == [-1,0,0,0]).all() # i x i
    assert (Coquaternions.Mult(np.array([0,0,1,0]), np.array([0,0,1,0])) == [1,0,0,0]).all() # j x j
    assert (Coquaternions.Mult(np.array([0,0,0,1]), np.array([0,0,0,1])) == [1,0,0,0]).all() # k x k
    assert (Coquaternions.Mult(np.array([0,1,0,0]), np.array([0,0,1,0])) == [0,0,0,1]).all() # i x j 
    assert (Coquaternions.Mult(np.array([0,0,1,0]), np.array([0,1,0,0])) == [0,0,0,-1]).all() # jxi


def test_Cl11():
    """
    Test simple operations in algebra.

    Returns
    -------
    None.

    """
    assert (Cl11.Mult(np.array([1,0,0,0]), np.array([1,1,1,1])) == [1,1,1,1]).all()   # 1 x 1+i+j+k
    assert (Cl11.Mult(np.array([0,1,0,0]), np.array([1,1,1,1])) == [1,1,1,1]).all() # i x 1+i+j+k
    assert (Cl11.Mult(np.array([0,0,1,0]), np.array([1,1,1,1])) == [-1,1,1,-1]).all() # j x 1+i+j+k
    assert (Cl11.Mult(np.array([0,0,0,1]), np.array([1,1,1,1])) == [1,-1,-1,1]).all() # k x 1+i+j+k
    assert (Cl11.Mult(np.array([1,0,0,0]), np.array([1,0,0,0])) == [1,0,0,0]).all() # 1 x 1
    assert (Cl11.Mult(np.array([0,1,0,0]), np.array([0,1,0,0])) == [1,0,0,0]).all() # i x i
    assert (Cl11.Mult(np.array([0,0,1,0]), np.array([0,0,1,0])) == [-1,0,0,0]).all() # j x j
    assert (Cl11.Mult(np.array([0,0,0,1]), np.array([0,0,0,1])) == [1,0,0,0]).all() # k x k
    assert (Cl11.Mult(np.array([0,1,0,0]), np.array([0,0,1,0])) == [0,0,0,1]).all() # i x j 
    assert (Cl11.Mult(np.array([0,0,1,0]), np.array([0,1,0,0])) == [0,0,0,-1]).all() # jxi


def test_Bicomplex():
    """
    Test simple operations in algebra.

    Returns
    -------
    None.

    """
    assert (Bicomplex.Mult(np.array([1,0,0,0]), np.array([1,1,1,1])) == [1,1,1,1]).all()   # 1 x 1+i+j+k
    assert (Bicomplex.Mult(np.array([0,1,0,0]), np.array([1,1,1,1])) == [-1,1,-1,1]).all() # i x 1+i+j+k
    assert (Bicomplex.Mult(np.array([0,0,1,0]), np.array([1,1,1,1])) == [-1,-1,1,1]).all() # j x 1+i+j+k
    assert (Bicomplex.Mult(np.array([0,0,0,1]), np.array([1,1,1,1])) == [1,-1,-1,1]).all() # k x 1+i+j+k
    assert (Bicomplex.Mult(np.array([1,0,0,0]), np.array([1,0,0,0])) == [1,0,0,0]).all() # 1 x 1
    assert (Bicomplex.Mult(np.array([0,1,0,0]), np.array([0,1,0,0])) == [-1,0,0,0]).all() # i x i
    assert (Bicomplex.Mult(np.array([0,0,1,0]), np.array([0,0,1,0])) == [-1,0,0,0]).all() # j x j
    assert (Bicomplex.Mult(np.array([0,1,0,0]), np.array([0,0,1,0])) == [0,0,0,1]).all() # i x j 
    assert (Bicomplex.Mult(np.array([0,0,1,0]), np.array([0,1,0,0])) == [0,0,0,1]).all() # jxi

def test_Tessarines():
    """
    Test simple operations in algebra.

    Returns
    -------
    None.

    """
    assert (Tessarines.Mult(np.array([1,0,0,0]), np.array([1,1,1,1])) == [1,1,1,1]).all()   # 1 x 1+i+j+k
    assert (Tessarines.Mult(np.array([0,1,0,0]), np.array([1,1,1,1])) == [-1,1,-1,1]).all() # i x 1+i+j+k
    assert (Tessarines.Mult(np.array([0,0,1,0]), np.array([1,1,1,1])) == [1,1,1,1]).all() # j x 1+i+j+k
    assert (Tessarines.Mult(np.array([0,0,0,1]), np.array([1,1,1,1])) == [-1,1,-1,1]).all() # k x 1+i+j+k
    assert (Tessarines.Mult(np.array([1,0,0,0]), np.array([1,0,0,0])) == [1,0,0,0]).all() # 1 x 1
    assert (Tessarines.Mult(np.array([0,1,0,0]), np.array([0,1,0,0])) == [-1,0,0,0]).all() # i x i
    assert (Tessarines.Mult(np.array([0,0,1,0]), np.array([0,0,1,0])) == [1,0,0,0]).all() # j x j
    assert (Tessarines.Mult(np.array([0,1,0,0]), np.array([0,0,1,0])) == [0,0,0,1]).all() # i x j 
    assert (Tessarines.Mult(np.array([0,0,1,0]), np.array([0,1,0,0])) == [0,0,0,1]).all() # jxi

def test_Octonions():
    """
    Test simple operations in algebra.

    Returns
    -------
    None.

    """
    assert (Octonions.Mult(np.array([1,0,0,0,0,0,0,0]), np.array([1,1,1,1,1,1,1,1])) == [1,1,1,1,1,1,1,1]).all()       # 1 x 1+i+j+k+l+m+n+o
    assert (Octonions.Mult(np.array([0,1,0,0,0,0,0,0]), np.array([1,1,1,1,1,1,1,1])) == [-1,1,-1,1,-1,1,1,-1]).all()   # i x 1+i+j+k+l+m+n+o
    assert (Octonions.Mult(np.array([0,0,1,0,0,0,0,0]), np.array([1,1,1,1,1,1,1,1])) == [-1,1,1,-1,-1,-1,1,1]).all()   # j x 1+i+j+k+l+m+n+o
    assert (Octonions.Mult(np.array([0,0,0,1,0,0,0,0]), np.array([1,1,1,1,1,1,1,1])) == [-1,-1,1,1,-1,1,-1,1]).all()   # k x 1+i+j+k+l+m+n+o
    assert (Octonions.Mult(np.array([1,0,0,0,0,0,0,0]), np.array([1,1,1,1,1,1,1,1])) == [1,1,1,1,1,1,1,1]).all()       # 1 x 1
    assert (Octonions.Mult(np.array([0,0,0,0,1,0,0,0]), np.array([1,1,1,1,1,1,1,1])) == [-1,1,1,1,1,-1,-1,-1]).all()   # l x 1+i+j+k+l+m+n+o
    assert (Octonions.Mult(np.array([0,0,0,0,0,1,0,0]), np.array([1,1,1,1,1,1,1,1])) == [-1,-1,1,-1,1,1,1,-1]).all()   # m x 1+i+j+k+l+m+n+o
    assert (Octonions.Mult(np.array([0,0,0,0,0,0,1,0]), np.array([1,1,1,1,1,1,1,1])) == [-1,-1,-1,1,1,-1,1,1]).all()   # n x 1+i+j+k+l+m+n+o
    assert (Octonions.Mult(np.array([0,0,0,0,0,0,0,1]), np.array([1,1,1,1,1,1,1,1])) == [-1,1,-1,-1,1,1,-1,1]).all()   # o x 1+i+j+k+l+m+n+o
    assert (Octonions.Mult(np.array([0,1,0,0,0,0,0,0]), np.array([0,0,1,0,0,0,0,0])) == [0,0,0,1,0,0,0,0]).all()       # i x j =k
    assert (Octonions.Mult(np.array([0,0,1,0,0,0,0,0]), np.array([0,1,0,0,0,0,0,0])) == [0,0,0,-1,0,0,0,0]).all()       # j x i =-k



if __name__=="__main__":
    test_Complex()
    test_Quaternins()
    test_Klein4()
    test_Cl20()
    test_Coquaternions()
    test_Cl11()
    test_Bicomplex()
    test_Tessarines()
    test_Octonions()
