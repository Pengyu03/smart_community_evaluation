import pulp


def calculate_bwm_weights(n, best, worst, bo_vector, ow_vector):
    """
    Calculate weights using the Best-Worst Method (BWM).

    Parameters:
        n (int): Number of criteria.
        best (int): Index of the best criterion.
        worst (int): Index of the worst criterion.
        bo_vector (list): Best-to-Others vector (e.g., [2, 5, 3]).
        ow_vector (list): Others-to-Worst vector (e.g., [4, 3, 2]).

    Returns:
        dict: A dictionary containing the weights and consistency ratio (xi).
    """
    # Validate input
    if len(bo_vector) != n - 1 or len(ow_vector) != n - 1:
        raise ValueError("BO and OW vectors must have length (n - 1).")
    if best == worst:
        raise ValueError("Best and worst criteria cannot be the same.")

    # Define the optimization problem
    prob = pulp.LpProblem("BWM_Weights", pulp.LpMinimize)

    # Define decision variables
    weights = [pulp.LpVariable(f"w_{i}", lowBound=0) for i in range(n)]  # Weights for each criterion
    xi = pulp.LpVariable("xi", lowBound=0)  # Consistency ratio

    # Objective function: Minimize xi
    prob += xi

    # Constraint: Sum of weights equals 1
    prob += pulp.lpSum(weights) == 1

    # Best-to-Others constraints
    others_bo = [i for i in range(n) if i != best]
    for j, a_bj in zip(others_bo, bo_vector):
        prob += weights[best] >= a_bj * weights[j] - xi
        prob += weights[best] <= a_bj * weights[j] + xi

    # Others-to-Worst constraints
    others_ow = [i for i in range(n) if i != worst]
    for i, a_iw in zip(others_ow, ow_vector):
        prob += weights[i] >= a_iw * weights[worst] - xi
        prob += weights[i] <= a_iw * weights[worst] + xi

    # Solve the problem
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if status == pulp.LpStatusOptimal:
        # Extract results
        w_values = [pulp.value(w) for w in weights]
        xi_value = pulp.value(xi)

        return {
            "weights": [round(v, 6) for v in w_values],
            "xi": round(xi_value, 6),
            "status": "Optimal"
        }
    else:
        return {
            "weights": None,
            "xi": None,
            "status": "Infeasible or error"
        }


if __name__ == "__main__":
    # Example usage
    n = 4  # Number of criteria
    best = 3  # Index of the best criterion (0-based)
    worst = 1  # Index of the worst criterion (0-based)
    bo_vector = [2, 5, 3]  # Best-to-Others vector
    ow_vector = [4, 3, 2]  # Others-to-Worst vector

    result = calculate_bwm_weights(n, best, worst, bo_vector, ow_vector)

    # Print results
    print("BWM Calculation Results:")
    print(f"Weights: {result['weights']}")
    print(f"Consistency Ratio (xi): {result['xi']}")
    print(f"Status: {result['status']}")