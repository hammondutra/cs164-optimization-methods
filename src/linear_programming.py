"""CS164 Session 16: Question 7, industrial LP using Python and CVXPY."""
import numpy as np
import cvxpy as cp

def solve_industrial():
    c = np.array([8., 6.])
    A = np.array([[1., 1.],
                  [5., 10.],
                  [2., 1.]])
    b = np.array([2., 16., 3.])
    x = cp.Variable(2, nonneg=True)
    problem = cp.Problem(cp.Maximize(c @ x), [A @ x <= b])
    problem.solve()
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Unexpected solver status: {problem.status}")
    values = np.asarray(x.value, dtype=float)
    if not np.allclose(values, [1, 1], atol=1e-4):
        raise AssertionError("Unexpected optimal production.")
    if not np.isclose(problem.value, 14., atol=1e-4):
        raise AssertionError("Unexpected optimal profit.")
    if not (np.all(A @ values <= b+1e-5) and np.all(values>=-1e-5)):
        raise AssertionError("Solution is infeasible.")
    return values, problem.value, A @ values, b, problem.status

def main():
    values, optimum, lhs, rhs, status = solve_industrial()
    print("Status:", status)
    print("Optimal production [x,y]:", np.round(values, 6))
    print("Maximum profit:", round(float(optimum), 6))
    print("Ax:", np.round(lhs, 6))
    print("b:", rhs)
    print("Constraints satisfied:", np.all(lhs <= rhs + 1e-5))

if __name__ == "__main__":
    main()
