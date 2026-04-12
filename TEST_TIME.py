import numpy as np
import time
import matplotlib.pyplot as plt


class MMSETest:
    def __init__(self, number_subcarriers, SNR):
        self.number_subcarriers = number_subcarriers
        self.SNR = SNR

    def _mmse(self, y_tensor, H_est):
        number_ofdm_symbols = y_tensor.shape[0]

        x_est = np.zeros((number_ofdm_symbols, self.number_subcarriers, 2), dtype=complex)

        for i in range(number_ofdm_symbols):
            for k in range(self.number_subcarriers):
                Hk = H_est[i, k, :, :]
                yk = y_tensor[i, k, :]

                Hh = Hk.conj().T
                W = np.linalg.inv(Hh @ Hk + (1 / self.SNR) * np.eye(2)) @ Hh

                x_est[i, k, :] = W @ yk

        return x_est


# ================= ПАРАМЕТРЫ =================
subcarriers = 7
SNR = 10

mmse = MMSETest(subcarriers, SNR)

# фиксированная H (2x2)
H_const = np.array([[1+1j, 0.5], [0.3j, 1-0.2j]])

symbols_range = np.arange(10, 1000, 20)
times = []

for Nsym in symbols_range:
    # создаем тензоры
    y_tensor = np.random.randn(Nsym, subcarriers, 2) + 1j * np.random.randn(Nsym, subcarriers, 2)

    # H тензор (копируем одну матрицу)
    H_est = np.tile(H_const, (Nsym, subcarriers, 1, 1))

    start = time.time()
    mmse._mmse(y_tensor, H_est)
    end = time.time()

    times.append(end - start)


# ================= ГРАФИК =================
plt.figure()
plt.plot(symbols_range, times, marker='o')
plt.xlabel("Количество OFDM символов")
plt.ylabel("Время выполнения (сек)")
plt.title("Зависимость времени MMSE от числа OFDM символов")
plt.grid()
plt.show()