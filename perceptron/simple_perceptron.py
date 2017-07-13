import numpy as np

class Perceptron(object):
	
	def __init__(self, weigths = np.array([1]), threshold = 0):
		
		self.weigths = weigths
		self.threshold = threshold
	
	
	def activation(self,inputs):
		
		strength = np.dot(self.weights,inputs)
		
		if strength <= self.threshold:
			self.result = 0
		else:
			self.result = 1
		
		return self.result
	
	
	def update(self,values,train,eta=0.1):
		
		for data_point in range(len(values)):
			
			prediction = self.activation(values[data_point])
			
			error = train[data_point] - prediction
			
			weight_update = error*prediction*train[data]
			
			self.weight += weight_update
	
	

def test():
    """
    A few tests to make sure that the perceptron class performs as expected.
    Nothing should show up in the output if all the assertions pass.
    """
    def sum_almost_equal(array1, array2, tol = 1e-6):
        return sum(abs(array1 - array2)) < tol

    p1 = Perceptron(np.array([1,1,1]),0)
    p1.update(np.array([[2,0,-3]]), np.array([1]))
    assert sum_almost_equal(p1.weights, np.array([1.2, 1, 0.7]))

    p2 = Perceptron(np.array([1,2,3]),0)
    p2.update(np.array([[3,2,1],[4,0,-1]]),np.array([0,0]))
    assert sum_almost_equal(p2.weights, np.array([0.7, 1.8, 2.9]))

    p3 = Perceptron(np.array([3,0,2]),0)
    p3.update(np.array([[2,-2,4],[-1,-3,2],[0,2,1]]),np.array([0,1,0]))
    assert sum_almost_equal(p3.weights, np.array([2.7, -0.3, 1.7]))
	

if __name__ == "__main__":
	test()