import numpy as np
from scipy.optimize import minimize

W11 = np.array([0.366, 0.261, 0.369])
W21 = np.array([0.421, 0.243, 0.336])

W12 = np.array([0.289, 0.514, 0.197])
W22 = np.array([0.176, 0.456, 0.368])

W13 = np.array([0.476,0.524])
W23 = np.array([0.441,0.559])

W14 = np.array([0.257,0.232, 0.279, 0.231])
W24 = np.array([0.284,0.188, 0.195, 0.333])

for i in range(1,5):
    W1 = eval(f'W1{i}')
    W2 = eval(f'W2{i}')
    def objective(lambda_vec):
        lambda1, lambda2 = lambda_vec
        W = lambda1 * W1 + lambda2 * W2
        return np.linalg.norm(W - W1)**2 + np.linalg.norm(W - W2)**2


    constraint = {'type': 'eq', 'fun': lambda lambda_vec: lambda_vec[0] + lambda_vec[1] - 1}


    initial_guess = [0, 1]


    result = minimize(objective, initial_guess, constraints=constraint)


    lambda1_opt, lambda2_opt = result.x
    W_opt = lambda1_opt * W1 + lambda2_opt * W2

    print("Optimal lambda1:", lambda1_opt)
    print("Optimal lambda2:", lambda2_opt)
    print("Optimal W:", W_opt)
    print("Minimum objective value:", result.fun)
    print('')