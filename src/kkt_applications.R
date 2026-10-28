# CS164 Session 15 — Applying & Interpreting the KKT Conditions
# Companion R script for the nonnegative least-squares fitting PCW.
# Run from the repository root: Rscript src/kkt_applications.R
# Base R only; no external packages required.

X <- rbind(
  c(1, 1, 2, 3, 2, 5),
  c(2, 1, 2, 3, 2, 5),
  c(1, 3, 5, 1, 3, 1)
)
y <- c(3, 2, 2, 2, 3, 2)

H <- 2 * X %*% t(X)
b <- as.vector(2 * X %*% y)

# Complementary slackness: for each i, either theta[i] = 0 or mu[i] = 0.
cases <- expand.grid(
  a1 = c(FALSE, TRUE),
  a2 = c(FALSE, TRUE),
  a3 = c(FALSE, TRUE)
)

for (j in seq_len(nrow(cases))) {
  active <- which(unlist(cases[j, ]))
  free <- setdiff(1:3, active)
  label <- if (length(active) == 0) "none" else paste(active, collapse = ",")

  if (length(free) == 0) {
    cat("Active:", label, "- Infeasible (sum cannot equal 1)\n")
    next
  }

  k <- length(free)
  K <- rbind(
    cbind(H[free, free, drop = FALSE], rep(1, k)),
    c(rep(1, k), 0)
  )
  solution <- solve(K, c(b[free], 1))

  theta <- numeric(3)
  theta[free] <- solution[1:k]
  lambda <- solution[k + 1]
  mu <- as.vector(H %*% theta - b + lambda)
  mu[free] <- 0

  error <- sum((as.vector(t(X) %*% theta) - y)^2)
  valid <- all(theta >= -1e-9) && all(mu >= -1e-9)

  cat("\nActive:", label,
      "\ntheta:", round(theta, 6),
      "\nmu:", round(mu, 6),
      "\nlambda:", round(lambda, 6),
      "\nSquared error:", round(error, 6),
      "\nKKT satisfied:", valid, "\n")

  if (valid) {
    theta_opt <- theta
    lambda_opt <- lambda
    mu_opt <- mu
    error_opt <- error
  }
}

cat("\nOptimal mixture:", round(theta_opt, 6), "\n")
cat("Multipliers (lambda, mu):", round(lambda_opt, 6), "/",
    round(mu_opt, 6), "\n")
cat("Minimum squared error:", round(error_opt, 6), "\n")

# Keep the sum-to-one equality, but drop nonnegativity.
K_affine <- rbind(cbind(H, rep(1, 3)), c(rep(1, 3), 0))
theta_affine <- as.vector(solve(K_affine, c(b, 1))[1:3])
error_affine <- sum((as.vector(t(X) %*% theta_affine) - y)^2)
cat("\nEquality-only mixture:", round(theta_affine, 6), "\n")
cat("Equality-only squared error:", round(error_affine, 6), "\n")
