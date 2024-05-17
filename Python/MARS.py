import numpy as np
import matplotlib.pyplot as plt

class MARS:
    def __init__(self, max_terms=10):
        self.max_terms = max_terms
        self.terms = []
        self.directions = []

    def _basis_function(self, x, knot, direction):
        if direction == 'left':
            return np.maximum(0, knot - x)
        else:
            return np.maximum(0, x - knot)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initializing with the intercept term
        self.terms = [(np.ones(n_samples), 0)]  # term and feature index
        self.coefficients = [np.mean(y)]

        residual = y - np.mean(y)  # Initial residuals

        for _ in range(self.max_terms):
            best_score = float('inf')
            best_term = None # basis function's parameters tuple
            best_direction = None
            best_coefficients = None

            # Search for the best basis function
            for feature in range(n_features):
                for sample in range(n_samples):
                    knot = X[sample, feature]
                    for direction in ['left', 'right']:
                        term = self._basis_function(X[:, feature], knot, direction)
                        terms_matrix = np.vstack([term]).T
                        coefficients = np.linalg.lstsq(terms_matrix, residual, rcond=None)[0]
                        score = np.sum((residual - terms_matrix @ coefficients) ** 2)

                        if score < best_score:
                            best_score = score
                            best_term = (feature, knot)
                            best_direction = direction
                            best_coefficients = coefficients

            if best_term is not None:
                self.terms.append(best_term)
                self.directions.append(best_direction)
                self.coefficients.append(best_coefficients)
                residual = residual - self._basis_function(X[:, best_term[0]], best_term[1], best_direction) * best_coefficients
                print(np.sum(residual**2))

    def predict(self, X):
        n_samples = X.shape[0]
        terms = [np.ones(n_samples)]
        for (feature, knot), direction in zip(self.terms[1:], self.directions):
            terms.append(self._basis_function(X[:, feature], knot, direction))
        terms = np.vstack(terms).T
        return terms @ self.coefficients


# Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * np.sin(2 * np.pi * X[:, 0]) + np.random.normal(0, 0.1, size=100)

# Fit the MARS model
model = MARS(max_terms=50)
model.fit(X, y)

# Predict on new data
X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_pred, color='red', label='MARS fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
