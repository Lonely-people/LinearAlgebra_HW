import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

def gen_data(N = 80, a = 2.0, b = -1.0, lf = -3.0, ri = 3.0, sigma_0 = 0.06, k = 20):
    rng = np.random.default_rng(114514)
    x = rng.uniform(lf, ri, size = N)
    x = np.sort(x)
    # print(x)
    sigma = sigma_0 * (1.0 + k * abs(x))
    eps = rng.normal(loc = 0.0, scale = sigma, size = N)
    y = a * x + b + eps
    return x, y, sigma

def least_squares(x, y):
    A = np.column_stack([x, np.ones(len(x))])
    # print(A)
    sol = np.linalg.solve(A.T @ A, A.T @ y)
    return sol

def weighted_least_squares(x, y, sigma):
    A = np.column_stack([x, np.ones(len(x))])
    inv_v = 1.0 / (sigma ** 2)
    W = np.diag(inv_v)
    # print(W)
    L = A.T @ W @ A
    R = A.T @ W @ y
    sol = np.linalg.solve(L, R)
    return sol

def calc_MSE(x, xx, a, b):
    y = a * x + b
    y_hat = xx[0] * x + xx[1]
    return np.mean((y_hat - y) ** 2)

if __name__ == "__main__":
    a_true = 2.0; b_true = -1.0
    # print(a_true, b_true)
    x, y, sigma = gen_data()
    sol1 = least_squares(x, y)
    sol2 = weighted_least_squares(x, y, sigma)
    # print(sol1)
    # print(sol2)
    print("真实参数 (a, b) = ", a_true, b_true)
    print("普通最小二乘估计 (a_1, b_1) = ", sol1[0], sol1[1])
    print("加权最小二乘估计 (a_2, b_2) = ", sol2[0], sol2[1])

    print("\na 误差: 普通:", abs(sol1[0] - a_true), " 加权:", abs(sol2[0] - a_true))
    print("b 误差: 普通:", abs(sol1[1] - b_true), " 加权:", abs(sol2[1] - b_true))

    print("\n普通 MSE = ", calc_MSE(x, sol1, a_true, b_true))
    print("加权 MSE = ", calc_MSE(x, sol2, a_true, b_true))
    
    # pic1
    plt.figure()
    plt.errorbar(x, y, yerr = sigma, fmt = 'o', markersize = 1, capsize = 1, label = "数据点")
    xx = np.linspace(x.min(), x.max(), 200)
    y_true = a_true * xx + b_true
    y_1 = sol1[0] * xx + sol1[1]
    y_2 = sol2[0] * xx + sol2[1]
    plt.plot(xx, y_true, linewidth = 2, label = "真实直线")
    plt.plot(xx, y_1, linewidth = 2, label = "LS 拟合直线")
    plt.plot(xx, y_2, linewidth = 2, label = "WLS 拟合直线")
    plt.title("普通估计和加权估计的对比")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha = 0.2)
    # plt.show()

    # pic 2
    plt.figure()
    w = 1.0 / (sigma ** 2)
    plt.plot(x, w, marker = 'o', linewidth = 1)
    plt.title("权重曲线")
    plt.xlabel("x")
    plt.ylabel("w")
    plt.grid(True, alpha = 0.2)
    plt.show()
