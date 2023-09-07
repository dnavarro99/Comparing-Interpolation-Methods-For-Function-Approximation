import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, make_interp_spline
from scipy.interpolate import lagrange
import time
import zipfile

with zipfile.ZipFile('Math104Afinal_codes.zip', 'w') as myzip:
    myzip.write('main.py')



def neville(x, y, x_new):
    n = len(x)
    Q = np.zeros((n, n))
    Q[:, 0] = y
    for j in range(1, n):
        for i in range(n-j):
            Q[i][j] = ((x_new - x[i+j])*Q[i][j-1] - (x_new - x[i])*Q[i+1][j-1])/(x[i] - x[i+j])
    return Q[0][-1]


# define the data points
x = np.linspace(0, 2*np.pi, 7)
y = np.sin(x)

# define the range of x values to interpolate over
x_new = np.linspace(x.min(), x.max(), 300)

# define the true function
y_true = np.sin(x_new)

# linear interpolation
start_time = time.time()
linear_interp = interp1d(x, y)
y_linear = linear_interp(x_new)
linear_time = time.time() - start_time

# quadratic interpolation
start_time = time.time()
quad_interp = interp1d(x, y, kind='quadratic')
y_quad = quad_interp(x_new)
quad_time = time.time() - start_time


# cubic spline interpolation
start_time = time.time()
spline_interp = make_interp_spline(x, y, bc_type='natural')
y_spline = spline_interp(x_new)
spline_time = time.time() - start_time

# Lagrange polynomial interpolation
start_time = time.time()
lagrange_interp = lagrange(x, y)
y_lagrange = lagrange_interp(x_new)
lagrange_time = time.time() - start_time


# Neville's method interpolation
start_time = time.time()
y_neville = np.array([neville(x, y, xi) for xi in x_new])
neville_time = time.time() - start_time


# compute the RMSE for each interpolation method
rmse_linear = np.sqrt(np.mean((y_linear - y_true)**2))
rmse_quad = np.sqrt(np.mean((y_quad - y_true)**2))
rmse_spline = np.sqrt(np.mean((y_spline - y_true)**2))
rmse_lagrange = np.sqrt(np.mean((y_lagrange - y_true)**2))
# compute the RMSE for Neville's method interpolation
rmse_neville = np.sqrt(np.mean((y_neville - y_true)**2))


# compute the absolute error for each interpolation method
ae_linear = np.mean(np.abs(y_linear - y_true))
ae_quad = np.mean(np.abs(y_quad - y_true))
ae_spline = np.mean(np.abs(y_spline - y_true))
ae_lagrange = np.mean(np.abs(y_lagrange - y_true))
ae_neville = np.mean(np.abs(y_neville - y_true))

'''
# Just to visualize what the coefficients were but was not used when evaluating efficiency so can ignore
coeffs = np.zeros(len(x))
for i in range(len(x)):
    coeffs[i] = neville(x, y, x[i])
'''


# print the time for each method
print(f"Linear interpolation: time = {linear_time:.8f} s")
print(f"Quadratic interpolation: time = {quad_time:.8f} s")
print(f"Cubic spline interpolation: time = {spline_time:.8f} s")
print(f"Lagrange polynomial interpolation: time = {lagrange_time:.8f} s")
print(f"Neville's method interpolation: time = {neville_time:.8f} s")

print("=========================================================================================================")
# Print the values of the variables
print(f"Absolute error for linear interpolation method = {ae_linear:.4f}")
print(f"Absolute error for quadratic interpolation method = {ae_quad:.4f}")
print(f"Absolute error for spline interpolation method= {ae_spline:.4f}")
print(f"Absolute error for Lagrange interpolation method = {ae_lagrange:.4f}")
print(f"Absolute error for Neville interpolation method = {ae_neville:.4f}")
print("=========================================================================================================")

# print the RMSE for each method
print(f"Linear interpolation: RMSE = {rmse_linear:.4f}")
print(f"Quadratic interpolation: RMSE = {rmse_quad:.4f}")
print(f"Cubic spline interpolation: RMSE = {rmse_spline:.4f}")
print(f"Lagrange polynomial interpolation: RMSE = {rmse_lagrange:.4f}")
print(f"Neville's method interpolation: RMSE = {rmse_neville:.4f}")

print("=========================================================================================================")

'''
# print the coefficients for each interpolation method(Again can ignore)

np.set_printoptions(precision=4, suppress=True, floatmode='fixed')
print(f"Linear interpolation coefficients: [{', '.join(map(str, linear_interp.y))}]")
print(f"Quadratic interpolation coefficients:[ {', '.join(map(str, quad_interp.y))}]")
print(f"Cubic spline interpolation coefficients:[ {', '.join(map(str, spline_interp.c.T.flatten()))}]")
print(f"Lagrange polynomial interpolation coefficients: [{', '.join(map(str, lagrange_interp.coefficients))}]")
print(f"Neville's method coefficients: [{', '.join(map(str, coeffs))}]")

print("=========================================================================================================")
'''

# plot the results
plt.plot(x, y, 'o', label='data points')
plt.plot(x_new, y_true, label='true function')
plt.plot(x_new, y_linear, label='linear')
plt.plot(x_new, y_quad, label='quadratic')
plt.plot(x_new, y_spline, label='cubic spline')
plt.plot(x_new, y_lagrange, label='lagrange')
plt.plot(x_new, y_neville, label='neville')
plt.legend()

# plot the predicted and true values
plt.figure()
plt.plot(x_new, y_linear, label='linear')
plt.plot(x_new, y_true, label='true function')
plt.legend()
plt.title('Linear Interpolation')

plt.figure()
plt.plot(x_new, y_quad, label='quadratic')
plt.plot(x_new, y_true, label='true function')
plt.legend()
plt.title('Quadratic Interpolation')

plt.figure()
plt.plot(x_new, y_spline, label='cubic spline')
plt.plot(x_new, y_true, label='true function')
plt.legend()
plt.title('Cubic Spline Interpolation')

# plot the Lagrange polynomial interpolation
plt.figure()
plt.plot(x_new, y_lagrange, label='Lagrange polynomial')
plt.plot(x_new, y_true, label='true function')
plt.legend()
plt.title('Lagrange Polynomial Interpolation')

# plot Neville's method interpolation
plt.figure()
plt.plot(x_new, y_neville, label="Neville's method")
plt.plot(x_new, y_true, label='true function')
plt.legend()
plt.title("Neville's Method Interpolation")

plt.show()
