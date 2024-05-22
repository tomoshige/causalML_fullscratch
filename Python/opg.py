import numpy as np

def opg(x, y, d):
    p = x.shape[1]
    n = x.shape[0]
    c0 = 2.34
    p0 = max(p, 3)
    rn = n ** (-1 / (2 * (p0 + 6)))
    h = c0 * n ** (-(1 / (p0 + 6)))
    sig = np.var(x, axis=0,ddof=1)
    x = np.apply_along_axis(standvec, 0, x)
    kmat = kern(x, h)
    bmat = np.empty((x.shape[1], 0))
    for i in range(x.shape[0]):
        wi = kmat[:, i]
        xi = np.hstack((np.ones((n, 1)), x - x[i, :]))
        b = wls(xi, y, wi)['b']
        bmat = np.hstack((bmat, b.reshape(-1, 1)))
    beta = np.linalg.eig(np.dot(bmat, bmat.T))[1][:, :d]
    return np.dot(np.diag(sig ** (-1 / 2)), beta)

def wls(x, y, w):
    n = x.shape[0]
    p = x.shape[1] - 1
    xw = x * w[:, np.newaxis]
    xyw = x * y[:,np.newaxis] * w[:,np.newaxis]
    out = np.linalg.solve(np.dot(xw.T, x)/n, np.apply_along_axis(np.mean, 0, xyw))
    return {'a': out[0], 'b': out[1:(p + 1)]}

def kern(x, h):
    x = np.array(x)
    n = x.shape[0]
    k2 = np.dot(x, x.T)
    k1 = np.tile(np.diag(k2), (n, 1))
    k3 = k1.T
    k = k1 - 2 * k2 + k3
    return np.exp(-(1 / (2 * h ** 2)) * (k1 - 2 * k2 + k3))

def standvec(x):
    return (x - np.mean(x)) / np.std(x,ddof=1)

# データの生成
# np.random.seed(123) # 再現性のためのシード設定
X = np.random.normal(size=(100, 20))
Y = np.sin(X[:, 0])

# 関数の実行
result = opg(X, Y, 1)
print(result)