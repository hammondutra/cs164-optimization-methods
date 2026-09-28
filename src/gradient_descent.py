"""CS164 Session 7: Armijo backtracking, normalized gradient descent, and tests.

Reconstructed from the user's September 28, 2026 PCW PDF screenshot.
Requires: numpy. Run `python src/gradient_descent.py` from the repository root.
"""

# Backtracking: Armijo approximate line search
import numpy as np


def backtracking(f, grad_f, d, x_i, alpha=1.0, p=0.5, c=1e-4):
    """
    Armijo backtracking line search.

    Parameters
    ----------
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient of the objective function.
    d : np.ndarray
        Descent direction.
    x_i : np.ndarray
        Current position.
    alpha : float
        Initial step size (default: 1.0).
    p : float
        Step-size reduction factor (default: 0.5).
    c : float
        Armijo sufficient-decrease parameter (default: 1e-4).

    Returns
    -------
    float
        An acceptable step size satisfying the Armijo condition.
    """
    x_i = np.asarray(x_i, dtype=float)
    d = np.asarray(d, dtype=float)

    # Evaluate the function and gradient at x_i
    f_i = f(x_i)
    grad_i = grad_f(x_i)

    # Calculate the initial directional derivative
    slope = np.dot(grad_i, d)

    # The search direction must be a descent direction
    if slope >= 0:
        raise ValueError("d must be a descent direction.")

    # Reduce alpha until sufficient decrease is achieved
    while f(x_i + alpha * d) > f_i + c * alpha * slope:
        alpha = p * alpha

    # Return the accepted step size
    return alpha


# Gradient descent with central finite differences and normalized directions

def central_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x, dtype=float)

    for j in range(len(x)):
        step = np.zeros_like(x, dtype=float)
        step[j] = h
        grad[j] = (f(x + step) - f(x - step)) / (2 * h)

    return grad


def gradient_descent(f, x0, epsilon=1e-6, max_iter=1000):
    """
    Minimize f using normalized gradient descent.

    Parameters
    ----------
    f : callable
        Objective function.
    x0 : np.ndarray
        Initial position.
    epsilon : float
        Gradient magnitude tolerance.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    x : np.ndarray
        Final position.
    f(x) : float
        Objective function value at the final position.
    """
    x = np.array(x0, dtype=float, copy=True)

    # Gradient function used by both gradient descent and backtracking
    grad_f = lambda x: central_gradient(f, x)

    for i in range(max_iter):
        # Compute the gradient
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)

        # Check the termination condition
        if grad_norm < epsilon:
            return x, f(x)

        # Calculate the normalized descent direction
        d = -grad / grad_norm

        # Find an acceptable step size using backtracking
        alpha = backtracking(f, grad_f, d, x)

        # Update the current position
        x_new = x + alpha * d

        # Check whether floating-point precision prevents progress
        if np.array_equal(x_new, x):
            raise RuntimeError("Step too small to make numerical progress.")

        x = x_new

    print("Maximum number of iterations reached.")
    return x, f(x)


# Optional: analytical exact line search for the saddle function

def exact_saddle_line_search(f, grad_f, d, x_i):
    """
    Analytical exact line search for f(x,y) = x^2 - y^2.
    Assumes d is a strict descent direction.
    """
    slope = np.dot(grad_f(x_i), d)
    curvature = d[0]**2 - d[1]**2

    if curvature <= 0:
        raise ValueError(
            "No finite optimal step: "
            "the function is unbounded below along this direction."
        )

    return -slope / (2 * curvature)




def run_exercises():
    # Three tests for backtracking (as in the submitted PCW)
    # Test 1: Quadratic bowl f(x,y) = x^2 + y^2
    f1 = lambda x: x[0]**2 + x[1]**2
    grad1 = lambda x: 2 * x
    x1 = np.array([1.0, 1.0])
    d1 = -grad1(x1)
    alpha1 = backtracking(f1, grad1, d1, x1)
    print("Test 1 - Quadratic bowl")
    print(f"Step size: {alpha1}, New point: {x1 + alpha1 * d1}")
    print()

    # Test 2: Shifted quadratic f(x,y) = (x-3)^2 + (y+2)^2
    f2 = lambda x: (x[0] - 3)**2 + (x[1] + 2)**2
    grad2 = lambda x: np.array([2 * (x[0] - 3), 2 * (x[1] + 2)])
    x2 = np.array([0.0, 0.0])
    d2 = -grad2(x2)
    alpha2 = backtracking(f2, grad2, d2, x2)
    print("Test 2 - Shifted quadratic")
    print(f"Step size: {alpha2}, New point: {x2 + alpha2 * d2}")
    print()

    # Test 3: One-dimensional quadratic f(x) = x^2
    f3 = lambda x: x[0]**2
    grad3 = lambda x: 2 * x
    x3 = np.array([4.0])
    d3 = -grad3(x3)
    alpha3 = backtracking(f3, grad3, d3, x3)
    print("Test 3 - One-dimensional quadratic")
    print(f"Step size: {alpha3}, New point: {x3 + alpha3 * d3}")

    assert np.isclose(alpha1, 0.5)
    assert np.isclose(alpha2, 0.5)
    assert np.isclose(alpha3, 0.5)
    print("\nAll 3 tests passed!")


    # Test 1: Quadratic bowl and numerical-vs-exact comparison
    def f(x):
        return x[0]**2 + x[1]**2

    x0 = np.array([1.0, 1.0])
    x_min, f_min = gradient_descent(f, x0)
    true_min = np.array([0.0, 0.0])
    true_f_min = 0.0

    print("Test 1 - Quadratic Bowl:")
    print("True minimum position:", true_min)
    print("Computed position:", x_min)
    print("Distance to true minimum:", np.linalg.norm(x_min - true_min))
    print("True minimum value:", true_f_min)
    print("Computed value:", f_min)
    print("Objective error:", abs(f_min - true_f_min))


    # Test 2: Saddle point f(x,y) = x^2 - y^2
    def f(x):
        return x[0]**2 - x[1]**2

    x0 = np.array([1.0, 1.0])

    # Run gradient descent with a 30-iteration limit
    x, f_final = gradient_descent(f, x0, max_iter=30)

    print("\nTest 2 - Saddle Point")
    print("Starting position:", x0)
    print("Starting value:", f(x0))
    print("Final position:", x)
    print("Final value:", f_final)
    print("Change in objective:", f_final - f(x0))

    # The origin is a stationary saddle point, NOT a minimum
    print("Saddle point:", np.array([0.0, 0.0]))
    print("Function value at saddle:", f(np.array([0.0, 0.0])))


    # Exact saddle-point line-search example
    # Define the saddle function and its analytical gradient
    def f_saddle(x):
        return x[0]**2 - x[1]**2


    def grad_saddle(x):
        return np.array([2 * x[0], -2 * x[1]])


    # Same starting point
    x0 = np.array([1.0, 1.0])
    grad = grad_saddle(x0)
    d = -grad / np.linalg.norm(grad)

    try:
        alpha = exact_saddle_line_search(f_saddle, grad_saddle, d, x0)
        x_new = x0 + alpha * d

        print("Exact step size:", alpha)
        print("New position:", x_new)
        print("New objective value:", f_saddle(x_new))
    except ValueError as error:
        print(error)


if __name__ == "__main__":
    run_exercises()
