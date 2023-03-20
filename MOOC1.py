import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from scipy.stats import multivariate_normal as mvn
import math
from scipy.linalg import sqrtm




# Task 1
# import measurements
file = open("measurement.txt", 'r')

h = []
q = []
for line in file.readlines()[1:-1]:
    h.append(float(line.split(',')[0]))
    q.append(float(line.split(',')[1]))
    
h = np.array(h)
q = np.array(q)


log_q = np.log(q)
log_h = np.log(h)

n = 2 
n_d = h.shape[0]

x = np.random.rand(n,1)

A = np.zeros((n_d, n))
A[:, 0] = 1
A[:, 1] = log_h


print(A)

Ax = np.matmul(A, x)

Gamma = np.zeros((n_d, n_d))
for i in range(n_d):
    Gamma[i, i] = 0.03 * log_q[i]

eta = [x - y for x, y in zip(log_q, Ax)]
    

# Task 2

# Initialize
m_0 = np.array([0, 0])
Sigma_0 = np.array([[10, 0], [0, 10]])


covariance_matrix = np.zeros((n_d, n_d), float)
for i in range(n_d):
    covariance_matrix[i][i] = 0.03 * log_q[i]

# Formulas
inv_sum = np.linalg.inv(Gamma + np.matmul(A, np.matmul(Sigma_0, A.T)))
prod = np.matmul(Sigma_0, np.matmul(A.T, inv_sum))

# Equation 6
m = m_0 + np.matmul(prod,(log_q - A @ m_0))
print(m)

# Equation 7
Sigma = Sigma_0 - np.matmul(prod, np.matmul(A, Sigma_0))
print(Sigma)

# Task 3
def prior(x, m, sigma):
    x_m = np.array([[x[0][0] - m[0]], [x[1][0] - m[1]]])
    return np.exp(-1/2 * np.matmul(x_m.T, np.matmul(np.linalg.inv(sigma),(x_m))))

def likelihood(x, Gamma, A, q):
    diff = np.zeros((len(q), 1))
    for i in range(len(q)):
        diff[i, 0] = np.log(q[i])
    diff -= np.matmul(A, x)
    return np.exp(- 1/2 * np.linalg.norm(np.matmul(np.linalg.inv(sqrtm(Gamma)), diff))**2)


def likelihood_prior_prod(x, q, mean, covariance, Gamma):
    return prior(x, mean, covariance) * likelihood(x, Gamma, A, q)

# Task 4 

N = 5000  # number of desired samples in the chain
x0 = np.ones(n)  # starting point

def dens(x):
    x = np.array([[x[0]], [x[1]]])
    return likelihood_prior_prod(x, q, m_0, Sigma_0, Gamma)


prop_var = 0.25**2  # variance of proposal #TODO

def proposal():
    return rnd.multivariate_normal(mean=np.zeros(n), cov=prop_var*np.eye(n))

samples = np.empty((N, n))  # storage for generated samples
samples[0, :] = x0  # first element in the chain is x0

x = x0  # initialization
dens_x = dens(x)
accptd = 0  # number of accepted proposals
    

for j in range(1, N):
    eps = proposal()
    x_ = x + eps
    dens_x_ = dens(x_)

    accpt_prob = np.min([1, dens_x_/dens_x])

    if accpt_prob >= rnd.random():
        # accept
        x = x_
        dens_x = dens_x_
        accptd += 1

    samples[j, :] = x

    if j % 100 == 0:
        # Print acceptance rate every 100th step
        print("Acceptance rate: %f" % (accptd/j))

    np.savetxt('samples.txt', samples)
    

# Estimate mean and covariance matrix of the samples

print(np.cov(samples.T))
print([np.mean(samples.T[0,:]), np.mean(samples.T[1,:])])

np.savetxt('samples.txt', samples)



plt.figure()
plt.plot(range(1, N+1), samples[:, 0])  # plot first component of all chain elements
plt.plot(range(1, N+1), samples[:, 1])  # plot first component of all chain elements
plt.tight_layout()
plt.show()
