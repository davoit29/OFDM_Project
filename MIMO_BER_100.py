import numpy as np
import matplotlib.pyplot as plt
import commpy as cp
import time
from commpy.modulation import QAMModem


class QAM:
    def __init__(self, SNR_db, M, number_ofdm_symbols, number_subcarriers):
        self.number_subcarriers = number_subcarriers
        self.number_ofdm_symbols = number_ofdm_symbols
        self.number_symbols = None

        self.SNR_db = SNR_db
        self.SNR = None

        self.x_qam_first = None
        self.x_qam_second = None
        self.x_qam = None
        self.x_time = None
        self.x_ofdm_tensor = None
        self.x_ofdm_tensor_time = None
        self.qam_matrix_a = None
        self.qam_matrix_b = None

        self.norm = None
        self.ofdm_matrix = None
        self.y = None
        self.bites_number = None
        self.M = M
        self.power_noise = None
        self.power_noise_freq = None

        self.x_bytes_a = None
        self.x_bytes_b = None
        self.x_bytes = None
        self.y_bytes = None
        self.y_zf = None
        self.y_mmse = None
        self.y_freq = None
        self.y_time = None
        self.output_bytes = None

        self.H = None

        self.h_12 = None
        self.h_11 = None
        self.h_21 = None
        self.h_22 = None

        self.H_11 = None
        self.H_12 = None
        self.H_21 = None
        self.H_22 = None

        self.delta_f = 15 * 10 ** 3  # расстояние между поднесущими
        self.bandwidth = 105 * 10 ** 3  # общая полоса
        self.Fs = 1e8  # частота дискретизации
        self.fc = 300 * 10 ** 3

    def time_domain_bandpass(self):
        self.modulating()
        self.ofdm()

        T_symbol = 1 / self.delta_f  # Длительность одного OFDM символа
        dt = 1 / self.Fs  # шаг дискретизации
        t_symbol = np.arange(0, T_symbol, dt)

        # поднесущие вокруг fc
        f_n = np.linspace(self.fc - (self.number_subcarriers // 2) * self.delta_f,
                          self.fc + (self.number_subcarriers // 2) * self.delta_f,
                          self.number_subcarriers)

        s = []
        for k in range(self.number_ofdm_symbols):
            s_symbol = np.zeros(len(t_symbol), dtype=complex)

            for n in range(self.number_subcarriers):
                # Для каждой антенны
                for ant in range(2):
                    s_symbol += self.x_ofdm_tensor_time[k, n, ant] * \
                                np.exp(1j * 2 * np.pi * f_n[n] * t_symbol)

            s.extend(s_symbol.real)  # Берем действительную часть

        s = np.array(s)
        t_total = np.arange(len(s)) * dt

        print(f'Общее время симуляции: {t_total[-1] * 1e6:.2f} мкс')

        plt.figure(figsize=(12, 6))
        plt.plot(t_total * 1e6, s, lw=0.7, color="red")
        plt.title(f'Полосовой OFDM сигнал')
        plt.xlabel("Время, мкс")
        plt.ylabel("Амплитуда")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def impulse_response(self):
        tau_us = np.array([0, 3, 5, 6, 8])  # мкс
        power_dB = np.array([0, -8, -17, -21, -25])

        # Мощность в линейных единицах
        power = 10 ** (power_dB / 10)

        # Амплитуды
        amplitude = np.sqrt(power)

        # Фазы с равномерным распределением
        phase = np.zeros_like(tau_us, dtype=complex)
        N = len(phase)
        for k in range(N):
            phase[k] = 2 * np.pi * np.random.randint(0, N - 1) / N

        # Импульсная характеристика
        h = amplitude * np.exp(1j * phase)
        plt.style.use('seaborn-v0_8-muted')

        # fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        #
        # # Рисуем stem-график с улучшенным форматированием
        # markerline, stemlines, baseline = ax.stem(
        #     tau_us,
        #     amplitude,
        #     linefmt='navy',  # Темно-синий цвет линий
        #     markerfmt='o',  # Круглые маркеры
        #     basefmt='gray'  # Серая базовая линия
        # )
        #
        # # Настройка визуальных эффектов
        # plt.setp(markerline, markersize=8, markerfacecolor='white', markeredgewidth=2)
        # plt.setp(stemlines, linewidth=1.5, alpha=0.7)
        # plt.setp(baseline, linewidth=1, alpha=0.5)
        #
        # # Добавляем сетку (только по горизонтали для акцента на амплитуде)
        # ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        # ax.xaxis.grid(False)
        #
        # # Названия и шрифты
        # ax.set_title('Импульсная характеристика канала', fontsize=18, fontweight='bold', pad=20)
        # ax.set_xlabel('Задержка $\\tau$, мкс', fontsize=14)
        # ax.set_ylabel('Амплитуда $|h|$', fontsize=14)
        #
        # # Убираем лишние границы рамки для чистоты
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        #
        # plt.tight_layout()
        # plt.show()

        return h


    def modulating(self):
        bites_per_antenna = self.number_subcarriers * self.number_ofdm_symbols * int(np.log2(self.M))

        self.x_bytes_a = np.random.randint(0, 2, bites_per_antenna)
        self.x_bytes_b = np.random.randint(0, 2, bites_per_antenna)
        self.x_bytes = np.concatenate((self.x_bytes_a, self.x_bytes_b))

        modem = QAMModem(self.M)

        self.x_qam_first = np.array(modem.modulate(self.x_bytes_a))
        self.x_qam_second = np.array(modem.modulate(self.x_bytes_b))

        self.x_qam = np.concatenate((self.x_qam_first, self.x_qam_second))

        self.qam_matrix_a = np.vstack((self.x_qam_first, self.x_qam_second))
        self.qam_matrix_b = np.column_stack((self.x_qam_first, self.x_qam_second))

        print(f"Число OFDM символов: {self.number_ofdm_symbols}")
        print(f"Число поднесущих: {self.number_subcarriers}")

    def ofdm(self):
        # Создаем тензор (символы, поднесущие, антенны)
        self.x_ofdm_tensor = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 2), dtype=complex)

        # Заполняем данными
        self.x_ofdm_tensor[:, :, 0] = self.x_qam_first.reshape(self.number_ofdm_symbols, self.number_subcarriers)
        self.x_ofdm_tensor[:, :, 1] = self.x_qam_second.reshape(self.number_ofdm_symbols, self.number_subcarriers)

        # IFFT по измерению поднесущих
        self.x_ofdm_tensor_time = np.fft.ifft(self.x_ofdm_tensor, axis=1) * np.sqrt(self.number_subcarriers)

        self.x_time = self.x_ofdm_tensor_time.flatten()

        print(f'Размерность OFDM тензора: {self.x_ofdm_tensor.shape}')

    def power(self):
        power_freq = np.mean(np.abs(self.x_ofdm_tensor.flatten()) ** 2)

        SNR = 10 ** (self.SNR_db / 10)

        self.SNR = SNR
        self.power_noise_freq = power_freq / SNR
        self.power_noise = self.power_noise_freq  # Во временной области та же мощность

        print(f"Мощность сигнала в частотной области: {power_freq}")
        print(f"SNR (линейное): {SNR}")
        print(f"Мощность шума: {self.power_noise}")

    def channel_with_noise(self):

        self.h_11 = self.impulse_response()
        self.h_12 = self.impulse_response()
        self.h_21 = self.impulse_response()
        self.h_22 = self.impulse_response()

        self.h_11 = np.concatenate((self.h_11, np.zeros(self.number_subcarriers - len(self.h_11), dtype=complex)))

        self.h_12 = np.concatenate((self.h_12, np.zeros(self.number_subcarriers - len(self.h_12), dtype=complex)))

        self.h_21 = np.concatenate((self.h_21, np.zeros(self.number_subcarriers - len(self.h_21), dtype=complex)))

        self.h_22 = np.concatenate((self.h_22, np.zeros(self.number_subcarriers - len(self.h_22), dtype=complex)))

        self.H_11 = np.fft.fft(self.h_11)
        self.H_12 = np.fft.fft(self.h_12)
        self.H_21 = np.fft.fft(self.h_21)
        self.H_22 = np.fft.fft(self.h_22)

        self.H = np.zeros((2, 2, self.number_subcarriers), dtype=complex)
        self.H[0, 0, :] = self.H_11
        self.H[0, 1, :] = self.H_12
        self.H[1, 0, :] = self.H_21
        self.H[1, 1, :] = self.H_22

        self.power()

        cp_len = len(self.h_11) - 1

        y_tensor = []

        for i in range(self.number_ofdm_symbols):

            # Берём два передающих вектора одного OFDM символа
            x1 = self.x_ofdm_tensor_time[i, :, 0]
            x2 = self.x_ofdm_tensor_time[i, :, 1]

            # Циклический префикс
            x1_cp = np.concatenate((x1[-cp_len:], x1))
            x2_cp = np.concatenate((x2[-cp_len:], x2))

            res_len = self.number_subcarriers + len(self.h_11) - 1

            y1_conv = np.zeros(res_len, dtype=complex)
            y2_conv = np.zeros(res_len, dtype=complex)

            for k in range(res_len):

                y1_k = 0
                y2_k = 0

                m = 0
                while k >= m and m <= len(self.h_11) - 1:
                    # y1 = h11*x1 + h12*x2
                    # y2 = h21*x1 + h22*x2
                    y1_k += self.h_11[m] * x1_cp[k - m]
                    y1_k += self.h_12[m] * x2_cp[k - m]

                    y2_k += self.h_21[m] * x1_cp[k - m]
                    y2_k += self.h_22[m] * x2_cp[k - m]

                    m += 1

                y1_conv[k] = y1_k
                y2_conv[k] = y2_k

            # Убираем CP
            y1_cut = y1_conv[cp_len:cp_len + self.number_subcarriers]
            y2_cut = y2_conv[cp_len:cp_len + self.number_subcarriers]

            # Шум
            noise1 = (np.random.randn(self.number_subcarriers) +
                      1j * np.random.randn(self.number_subcarriers)) * np.sqrt(self.power_noise / 2)

            noise2 = (np.random.randn(self.number_subcarriers) +
                      1j * np.random.randn(self.number_subcarriers)) * np.sqrt(self.power_noise / 2)

            y1_noisy = y1_cut + noise1
            y2_noisy = y2_cut + noise2

            # FFT
            Y1 = np.fft.fft(y1_noisy) / np.sqrt(self.number_subcarriers)
            Y2 = np.fft.fft(y2_noisy) / np.sqrt(self.number_subcarriers)

            y_symbol = np.column_stack((Y1, Y2))

            y_tensor.append(y_symbol)

        self.y = np.array(y_tensor)  # (Nsym, Nsub, 2)


    def zero_forcing(self):
        x_est = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 2), dtype=complex)

        for i in range(self.number_ofdm_symbols):
            for k in range(self.number_subcarriers):
                Hk = self.H[:, :, k]
                yk = self.y[i, k, :]

                # Решаем систему Hk * x = yk
                try:
                    x_est[i, k, :] = np.linalg.solve(Hk, yk)
                except np.linalg.LinAlgError:
                    # Если матрица вырожденная, используем псевдообратную
                    x_est[i, k, :] = np.linalg.pinv(Hk) @ yk

        self.y_zf = x_est.transpose(2, 0, 1).reshape(-1)

    def mmse(self):
        x_est = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 2), dtype=complex)

        for i in range(self.number_ofdm_symbols):
            for k in range(self.number_subcarriers):
                Hk = self.H[:, :, k]
                yk = self.y[i, k, :]

                # MMSE фильтр
                Hh = Hk.conj().T
                Sn = (1 / self.SNR) * np.eye(2)  # Предполагаем единичную мощность сигнала

                W = np.linalg.inv(Hh @ Hk + Sn) @ Hh
                x_est[i, k, :] = W @ yk

        self.y_mmse = x_est.transpose(2, 0, 1).reshape(-1)

    def decod(self, output):
        modem = QAMModem(self.M)
        output_bytes = modem.demodulate(output, demod_type='hard')
        return np.array(output_bytes)

    def ber(self, dec_bytes, num_err=0, num_bits=0, key='mmse', max_recursion=10):
        """
        Вычисление BER с накоплением до 100 ошибок (рекурсивно)

        Parameters:
        -----------
        dec_bytes : array
            Демодулированные биты для текущей итерации
        num_err : int
            Текущее количество накопленных ошибок
        num_bits : int
            Текущее количество накопленных бит
        key : str
            Тип эквалайзера ('mmse' или 'zf')
        max_recursion : int
            Максимальная глубина рекурсии

        Returns:
        --------
        tuple : (num_err, num_bits) - накопленные ошибки и биты
        """

        print(f"  BER calculation for {key}, current stats: {num_err} errors / {num_bits} bits")

        while num_err <= 100:
            # Сравниваем текущие биты
            N = min(len(self.x_bytes), len(dec_bytes))
            errors = np.count_nonzero(self.x_bytes[:N] != dec_bytes[:N])

            num_err += errors
            num_bits += N

            print(f"    Found {errors} errors in {N} bits, total: {num_err}/{num_bits}")

            if num_err > 100:
                print(f"    Reached 100 errors! Final stats: {num_err}/{num_bits}")
                return num_err, num_bits

            if num_err <= 100:
                print(f"  Need more errors, running new simulation with {key} equalizer...")

                # СОХРАНЯЕМ ТЕКУЩЕЕ СОСТОЯНИЕ
                old_state = {
                    'x_bytes': self.x_bytes.copy() if self.x_bytes is not None else None,
                    'x_bytes_a': self.x_bytes_a.copy() if self.x_bytes_a is not None else None,
                    'x_bytes_b': self.x_bytes_b.copy() if self.x_bytes_b is not None else None,
                    'x_qam_first': self.x_qam_first.copy() if self.x_qam_first is not None else None,
                    'x_qam_second': self.x_qam_second.copy() if self.x_qam_second is not None else None,
                    'x_qam': self.x_qam.copy() if self.x_qam is not None else None,
                    'x_ofdm_tensor': self.x_ofdm_tensor.copy() if self.x_ofdm_tensor is not None else None,
                    'x_ofdm_tensor_time': self.x_ofdm_tensor_time.copy() if self.x_ofdm_tensor_time is not None else None,
                    'y': self.y.copy() if self.y is not None else None,
                    'y_mmse': self.y_mmse.copy() if self.y_mmse is not None else None,
                    'y_zf': self.y_zf.copy() if self.y_zf is not None else None,
                    'H': self.H.copy() if self.H is not None else None
                }

                try:
                    # Генерируем новые случайные биты
                    bites_per_antenna = self.number_subcarriers * self.number_ofdm_symbols * int(np.log2(self.M))
                    self.x_bytes_a = np.random.randint(0, 2, bites_per_antenna)
                    self.x_bytes_b = np.random.randint(0, 2, bites_per_antenna)
                    self.x_bytes = np.concatenate((self.x_bytes_a, self.x_bytes_b))

                    # Модуляция
                    modem = QAMModem(self.M)
                    self.x_qam_first = np.array(modem.modulate(self.x_bytes_a))
                    self.x_qam_second = np.array(modem.modulate(self.x_bytes_b))
                    self.x_qam = np.concatenate((self.x_qam_first, self.x_qam_second))

                    # OFDM
                    self.x_ofdm_tensor = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 2), dtype=complex)
                    self.x_ofdm_tensor[:, :, 0] = self.x_qam_first.reshape(self.number_ofdm_symbols,
                                                                           self.number_subcarriers)
                    self.x_ofdm_tensor[:, :, 1] = self.x_qam_second.reshape(self.number_ofdm_symbols,
                                                                            self.number_subcarriers)
                    self.x_ofdm_tensor_time = np.fft.ifft(self.x_ofdm_tensor, axis=1) * np.sqrt(self.number_subcarriers)

                    # Канал с шумом
                    self.channel_with_noise()

                    # Применяем соответствующий эквалайзер
                    if key == 'mmse':
                        self.mmse()
                        new_dec = self.decod(self.y_mmse)
                    else:  # zf
                        self.zero_forcing()
                        new_dec = self.decod(self.y_zf)

                    # Рекурсивный вызов
                    return self.ber(new_dec, num_err, num_bits, key, max_recursion)

                finally:
                    # ВОССТАНАВЛИВАЕМ СОСТОЯНИЕ
                    for attr, value in old_state.items():
                        if value is not None:
                            setattr(self, attr, value)

        return num_err, num_bits

    def calc_evm_ber_snr_avg(self, snr_range, n_iter):
        ber_mmse_avg = []
        evm_mmse_avg = []
        ber_zf_avg = []
        evm_zf_avg = []

        original_snr = self.SNR_db

        for snr in snr_range:
            ber_mmse_list = []
            evm_mmse_list = []
            ber_zf_list = []
            evm_zf_list = []

            for _ in range(n_iter):
                self.SNR_db = snr

                self.modulating()
                self.ofdm()
                self.channel_with_noise()

                self.mmse()
                self.zero_forcing()

                # Демодуляция
                dec_mmse = self.decod(self.y_mmse)
                dec_zf = self.decod(self.y_zf)

                # BER с накоплением до 100 ошибок для MMSE
                num_err_mmse, num_bits_mmse = self.ber(dec_mmse, num_err=0, num_bits=0, key='mmse', max_recursion=5)
                ber_mmse = num_err_mmse / num_bits_mmse if num_bits_mmse > 0 else 1.0
                ber_mmse_list.append(ber_mmse)

                # BER с накоплением до 100 ошибок для ZF
                num_err_zf, num_bits_zf = self.ber(dec_zf, num_err=0, num_bits=0, key='zf', max_recursion=5)
                ber_zf = num_err_zf / num_bits_zf if num_bits_zf > 0 else 1.0
                ber_zf_list.append(ber_zf)

                # EVM (оставляем как есть)
                signal_power = np.mean(np.abs(self.x_qam) ** 2)
                evm_mmse = np.sqrt(np.mean(np.abs(self.y_mmse - self.x_qam) ** 2) / signal_power)
                evm_zf = np.sqrt(np.mean(np.abs(self.y_zf - self.x_qam) ** 2) / signal_power)

                evm_mmse_list.append(evm_mmse)
                evm_zf_list.append(evm_zf)

            ber_mmse_avg.append(np.mean(ber_mmse_list))
            ber_zf_avg.append(np.mean(ber_zf_list))
            evm_mmse_avg.append(np.mean(evm_mmse_list))
            evm_zf_avg.append(np.mean(evm_zf_list))

            print(f"SNR = {snr} dB: BER_MMSE = {ber_mmse_avg[-1]:.6f}, BER_ZF = {ber_zf_avg[-1]:.6f}")

        self.SNR_db = original_snr

        # Графики строятся так же
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        ber_zf_plot = np.maximum(ber_zf_avg, 1e-6)
        ber_mmse_plot = np.maximum(ber_mmse_avg, 1e-6)

        plt.semilogy(snr_range, ber_zf_plot, 'o-', label="ZF", linewidth=2, markersize=8)
        plt.semilogy(snr_range, ber_mmse_plot, 's-', label="MMSE", linewidth=2, markersize=8)
        plt.xlabel("SNR (dB)", fontsize=12)
        plt.ylabel("BER", fontsize=12)
        plt.title("BER vs SNR", fontsize=14)
        plt.ylim(1e-4, 1)
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend(fontsize=12)

        plt.subplot(1, 2, 2)
        plt.semilogy(snr_range, evm_zf_avg, 'o-', label="ZF", linewidth=2, markersize=8)
        plt.semilogy(snr_range, evm_mmse_avg, 's-', label="MMSE", linewidth=2, markersize=8)
        plt.xlabel("SNR (dB)", fontsize=12)
        plt.ylabel("Нормированная EVM", fontsize=12)
        plt.title("EVM vs SNR", fontsize=14)
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend(fontsize=12)

        plt.tight_layout()
        plt.show()

        return ber_zf_avg, ber_mmse_avg, evm_zf_avg, evm_mmse_avg

    def plot(self):
        self.modulating()
        self.ofdm()
        self.channel_with_noise()
        self.zero_forcing()
        self.mmse()

        decod_yzf = self.decod(self.y_zf)
        decod_y_mmse = self.decod(self.y_mmse)

        ber_yzf = self.ber(decod_yzf)
        ber_ymmse = self.ber(decod_y_mmse)
        print(f"\nBER для ZF: {ber_yzf:.6f}")
        print(f"BER для MMSE: {ber_ymmse:.6f}")

        x_re = self.x_qam.real
        x_im = self.x_qam.imag
        y_re = self.y.flatten().real
        y_im = self.y.flatten().imag
        y_zre = self.y_zf.real
        y_zim = self.y_zf.imag
        y_re_mmse = self.y_mmse.real
        y_im_mmse = self.y_mmse.imag

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0, 0].scatter(x_re, x_im, color='red', s=20, alpha=0.7)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title("Исходный сигнал QAM", fontsize=12)
        axes[0, 0].set_xlabel("In-Phase")
        axes[0, 0].set_ylabel("Quadrature")

        axes[0, 1].scatter(y_re, y_im, s=5, alpha=0.5, label='Принятый')
        axes[0, 1].scatter(x_re, x_im, color='red', s=20, alpha=0.7, label='Исходный')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title(f"Принятый сигнал (SNR = {self.SNR_db} dB)", fontsize=12)
        axes[0, 1].set_xlabel("In-Phase")
        axes[0, 1].set_ylabel("Quadrature")
        axes[0, 1].legend()

        axes[1, 0].scatter(y_zre, y_zim, s=5, alpha=0.5, label='ZF')
        axes[1, 0].scatter(x_re, x_im, color='red', s=20, alpha=0.7, label='Исходный')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title("ZF эквалайзинг", fontsize=12)
        axes[1, 0].set_xlabel("In-Phase")
        axes[1, 0].set_ylabel("Quadrature")
        axes[1, 0].legend()

        axes[1, 1].scatter(y_re_mmse, y_im_mmse, s=5, alpha=0.5, label='MMSE')
        axes[1, 1].scatter(x_re, x_im, color='red', s=20, alpha=0.7, label='Исходный')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title("MMSE эквалайзинг", fontsize=12)
        axes[1, 1].set_xlabel("In-Phase")
        axes[1, 1].set_ylabel("Quadrature")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
        ax2.scatter(y_re_mmse, y_im_mmse, label='MMSE', s=10, alpha=0.6)
        ax2.scatter(y_zre, y_zim, color='red', label='ZF', s=10, alpha=0.6)
        ax2.scatter(x_re, x_im, color='black', s=50, alpha=0.8, marker='x', label='Исходный')
        ax2.set_title('Сравнение MMSE и ZF', fontsize=14)
        ax2.set_xlabel("In-Phase")
        ax2.set_ylabel("Quadrature")
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        plt.tight_layout()
        plt.show()


# Тестирование
if __name__ == "__main__":
    qam_1 = QAM(SNR_db=15, M=16, number_ofdm_symbols=400, number_subcarriers=7)

    # Быстрый тест с графиками
    # qam_1.plot()

    # Тест зависимости BER от SNR
    qam_1.calc_evm_ber_snr_avg(
        snr_range=np.arange(0, 20, 3),
        n_iter=10
    )

    # Демонстрация временного сигнала
    # qam_1.time_domain_bandpass()
