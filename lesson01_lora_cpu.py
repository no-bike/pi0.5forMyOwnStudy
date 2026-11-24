"""用一个线性层在 CPU 上练习一次 LoRA 更新。运行需要 NumPy。"""

import numpy as np


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 8))       # 4 条样本，每条有 8 个输入特征
    target = rng.normal(size=(4, 6))  # 每条样本有 6 个目标输出
    weight = rng.normal(size=(8, 6))
    base_output = x @ weight

    # 两个小矩阵组成秩最多为 2 的修正量。本例把 LoRA 缩放系数设为 1。
    a = rng.normal(scale=0.1, size=(8, 2))
    b = rng.normal(scale=0.1, size=(2, 6))
    hidden = x @ a
    prediction = base_output + hidden @ b
    error = prediction - target
    loss_before = np.mean(error ** 2)

    # 对均方误差求梯度。先算出两个梯度，再同时更新 a、b。
    grad_output = 2 * error / error.size
    grad_b = hidden.T @ grad_output
    grad_a = x.T @ (grad_output @ b.T)
    learning_rate = 0.01
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b

    prediction = base_output + (x @ a) @ b
    loss_after = np.mean((prediction - target) ** 2)
    print(f"loss before: {loss_before:.6f}")
    print(f"loss after:  {loss_after:.6f}")


if __name__ == "__main__":
    main()
