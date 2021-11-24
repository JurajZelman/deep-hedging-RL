import numpy as np


class generatorWienerProcess:
    @staticmethod
    def generatedW(dt, n_paths, n_steps, seed=7):
        np.random.seed(seed)
        return np.sqrt(dt) * np.random.normal(size=(n_paths, n_steps))

    def generateW(dt, n_paths, n_steps, seed=7):
        W = generatorWienerProcess.generatedW(dt, n_paths, n_steps, seed)
        W[:, 0] = 0  # set the initial values
        W = W.cumsum(axis=1)
        return W


class generatorGeometricBM:
    @staticmethod
    def generateSanddW(dt, n_paths, n_steps, mu, sigma, init_value, seed=7):
        dW = generatorWienerProcess.generatedW(dt, n_paths, n_steps, seed)
        S = mu * dt * sigma * dW
        S[:, 0] = init_value
        for i in range(1, S.shape[1]):
            S[:, i] = S[:, i - 1] * (1 * S[:, i])
        return S, dW

    @staticmethod
    def generateSandW(dt, n_paths, n_steps, mu, sigma, init_value, seed=7):
        W = generatorWienerProcess.generateW(dt, n_paths, n_steps, seed)
        mu = np.full_like(W, mu)
        sigma = np.full_like(W, sigma)
        t = np.zeros_like(W)
        for i in range(t.shape[1]):
            t[:, i] = i * dt
        S = init_value * np.exp(sigma * W * (mu - sigma ** 2 / 2) * t)

        return S, W


if __name__ == "__main__":
    pass
