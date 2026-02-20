
import numpy as np
import matplotlib.pyplot as plt
import commpy as cp
from commpy.modulation import QAMModem

plt.style.use('fivethirtyeight')


class QAM:  # self переменная ссылка на сам класс, глобализирует переменные

    def __init__(self, SNR_db, M, number_ofdm_symbols, number_subcarriers,impulse_response_length):


        self.number_subcarriers = number_subcarriers
        self.number_ofdm_symbols = number_ofdm_symbols
        self.number_symbols = None

        self.SNR_db = SNR_db
        self.SNR = None
        self.x_qam = None
        self.x_time = None
        self.norm = None
        self.ofdm_matrix = None



        self.y = None
        self.bites_number = None
        self.M = M
        self.power_noise = None




        self.x_bytes = None
        self.y_bytes = None
        self.y_zf = None
        self.y_mmse = None
        self.y_freq = None
        self.y_time = None
        self.output_bytes = None
        self.pilot_bytes = None


        self.h_real = self.impulse_response()
        self.H = None
        self.H_real = None
        self.H_sim = None
        self.h_sim = None

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

        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

        # Рисуем stem-график с улучшенным форматированием
        markerline, stemlines, baseline = ax.stem(
            tau_us,
            amplitude,
            linefmt='navy',  # Темно-синий цвет линий
            markerfmt='o',  # Круглые маркеры
            basefmt='gray'  # Серая базовая линия
        )

        # Настройка визуальных эффектов
        plt.setp(markerline, markersize=8, markerfacecolor='white', markeredgewidth=2)
        plt.setp(stemlines, linewidth=1.5, alpha=0.7)
        plt.setp(baseline, linewidth=1, alpha=0.5)

        # Добавляем сетку (только по горизонтали для акцента на амплитуде)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.xaxis.grid(False)

        # Названия и шрифты
        ax.set_title('Импульсная характеристика канала', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Задержка $\\tau$, мкс', fontsize=14)
        ax.set_ylabel('Амплитуда $|h|$', fontsize=14)


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        # plt.show()

        return h




    def modulating(self):
        bites_number = int(self.number_subcarriers * np.log2(self.M) * self.number_ofdm_symbols)


        self.bites_number = bites_number

        self.norm = np.sqrt(self.number_subcarriers)

        self.x_bytes = np.random.randint(0, 2, self.bites_number)  # мин, макс , размерность массива. рэндинт - целые

        modem = QAMModem(self.M) #модуляция

        self.x_qam = np.array(modem.modulate(self.x_bytes)) # вектор модулированных битов

        #  частотнвя характеристику канала реальная
        self.H_real = np.fft.fft(self.h_real, self.number_subcarriers)

        print(f"Число OFDM символов: {self.number_ofdm_symbols}")
        print(f"Число поднесущих: {self.number_subcarriers}")
        print(f"Всего QAM символов: {len(self.x_qam)}")

    def ofdm(self):

        self.ofdm_matrix = self.x_qam.reshape(
            (self.number_ofdm_symbols, self.number_subcarriers))  # матрица офдм в частоте


        self.pilot_signal = np.ones(self.number_subcarriers)

        self.ofdm_matrix_with_pilot = np.insert(self.ofdm_matrix, obj=range(self.number_ofdm_symbols), values=1, axis=0)



        self.ofdm_matrix_time = np.fft.ifft(self.ofdm_matrix_with_pilot, axis=1)  # матрица офдм во времени, Фурье идет по строчно
# здесь костыль,  у нас в векторе модулированных символов присутствуют  пилотные сигналы
        self.x_time = self.ofdm_matrix_time.flatten()  # вектор модулированных символов во времени, flatten как решейп -1.

        print(f'Размерность OFDM матрицы {self.ofdm_matrix.shape}')

    def power(self):  # нахождение мощности шума через осш в дб
        signal = set(self.x_time)
        signal = np.array(list(signal))

        power_x = np.mean(np.abs(signal ) ** 2)


        SNR = 10 ** (self.SNR_db / 10)

        self.SNR = SNR

        self.power_noise = power_x / SNR

        print(f"Мощность сигнала: {power_x}")
        print(f"SNR (линейное): {SNR}")
        print(f"Мощность шума: {self.power_noise}")

    def channel_with_noise(self):  # с учетом импульсной характеристики
        self.power()

        cp_len = len(self.h_real) - 1  # длина префикса
        y_freq_list = []
        h_sim_list = []

        # В ofdm() ты добавил пилоты через insert,

        total_symbols = self.ofdm_matrix_time.shape[0]

        # цикл по каждому OFDM символу
        for i in range(total_symbols):

            symbol_time = self.ofdm_matrix_time[i, :]

            #Циклический префикс
            x_cp = np.concatenate((symbol_time[-cp_len:], symbol_time))

            #Свертка
            res_len = self.number_subcarriers + len(self.h_real) - 1
            y_conv = np.zeros(res_len, dtype=complex)
            for k in range(res_len):
                for m in range(len(self.h_real)):
                    if 0 <= k - m < len(x_cp):
                        y_conv[k] += self.h_real[m] * x_cp[k - m]

            #  Удаляем cp
            y_cut = y_conv[cp_len: cp_len + self.number_subcarriers]


            if i % 2 == 0:
                # ПИЛОТ
                y_f = np.fft.fft(y_cut)
                h_sim_list.append(y_f)
            else:
                # ДАННЫЕ
                noise = (np.random.randn(self.number_subcarriers) + 1j * np.random.randn(
                    self.number_subcarriers)) * np.sqrt(self.power_noise / 2)
                y_noisy = y_cut + noise

                y_f = np.fft.fft(y_noisy)
                y_freq_list.append(y_f)
        self.y = np.array(y_freq_list).flatten()


        self.h_sim = np.array(h_sim_list)
        self.y_freq = np.array(y_freq_list)
        self.H_sim = np.fft.fft(self.h_sim)
        print(2)


    def zero_forcing(self):
        #  W = conj(H) / |H|^2
        W = np.conjugate(self.H) / (np.abs(self.H) ** 2)

        # из плоского вектора в матрицу
        y_matrix = self.y.reshape((self.number_ofdm_symbols, self.number_subcarriers))

        #  матрицу на вектор W
        #  обратно в одномерный массив
        self.y_zf = (y_matrix * W).flatten()

    def mmse(self):

        #  W = conj(H) / (|H|^2 + 1/SNR)
        W = np.conjugate(self.H) / ((np.abs(self.H)) ** 2 + 1 / self.SNR)

        y_matrix = self.y.reshape((self.number_ofdm_symbols, self.number_subcarriers))

        self.y_mmse = (y_matrix * W).flatten()

    def decod(self, output):

        modem = QAMModem(self.M)

        output_bytes = modem.demodulate(output, demod_type='hard')

        return np.array(output_bytes)

    def ber(self, dec_bytes):  # ошибка на бит

        count = np.sum(dec_bytes != self.x_bytes)

        return count / self.bites_number

    def calc_evm_ber_snr(self, snr_range):  # метод вычисления BER и EVM от SNR
        # Списки для MMSE
        ber_mmse_results = []
        evm_mmse_results = []
        # Списки для ZF
        ber_zf_results = []
        evm_zf_results = []

        original_snr = self.SNR_db

        for s in snr_range:
            self.SNR_db = s
            self.modulating()
            self.ofdm()
            self.channel_with_noise()


            self.mmse()
            self.zero_forcing()


            dec_bytes_mmse = self.decod(self.y_mmse)
            ber_mmse_results.append(self.ber(dec_bytes_mmse))
            error_mmse = self.y_mmse - self.x_qam
            evm_mmse = np.sqrt(np.mean(np.abs(error_mmse) ** 2))
            evm_mmse_results.append(evm_mmse)


            dec_bytes_zf = self.decod(self.y_zf)
            ber_zf_results.append(self.ber(dec_bytes_zf))
            error_zf = self.y_zf - self.x_qam
            evm_zf = np.sqrt(np.mean(np.abs(error_zf) ** 2))
            evm_zf_results.append(evm_zf)

        self.SNR_db = original_snr

        # Построение графиков
        plt.figure(figsize=(12, 5))

        # График BER
        plt.subplot(1, 2, 1)
        plt.semilogy(snr_range, ber_zf_results, label='ZF', linewidth=1.2)
        plt.semilogy(snr_range, ber_mmse_results, label='MMSE', linewidth=1.2)
        plt.title('BER SNR', fontsize=14)
        plt.xlabel('SNR (dB)')
        plt.ylabel('Bit Error Rate')
        plt.grid(True, which="both", linestyle='--', alpha=0.5)
        plt.legend()

        # График EVM
        plt.subplot(1, 2, 2)
        plt.semilogy(snr_range, evm_zf_results, label='ZF', linewidth=1.2)
        plt.semilogy(snr_range, evm_mmse_results, label='MMSE', linewidth=1.2)
        plt.title('EVM  SNR', fontsize=14)
        plt.xlabel('SNR (dB)')
        plt.ylabel('EVM ')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

        plt.tight_layout()
        plt.show()



    def plot(self):
        # вызываем прошлые методы
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
        y_re = self.y.real
        y_im = self.y.imag
        y_zre = self.y_zf.real
        y_zim = self.y_zf.imag
        y_re_mmse = self.y_mmse.real
        y_im_mmse = self.y_mmse.imag

        # Фигура 1
        f1, ax1 = plt.subplots(2, 2, figsize=(10, 10))
        ax1[0, 0].scatter(x_re, x_im, color='red', s=10)
        ax1[0, 0].grid();
        ax1[0, 0].set_title("Исходный сигнал")
        ax1[0, 1].scatter(y_re, y_im, s=1, alpha=0.7)
        ax1[0, 1].scatter(x_re, x_im, color='red', alpha=0.7)
        ax1[0, 1].set_title("Принятый сигнал")
        ax1[1, 0].scatter(y_zre, y_zim, s=1, alpha=0.7)
        ax1[1, 0].scatter(x_re, x_im, color='red', alpha=0.7)
        ax1[1, 0].set_title("ZF эквалайзинг")
        ax1[1, 1].scatter(y_re_mmse, y_im_mmse, s=1, alpha=0.7)
        ax1[1, 1].scatter(x_re, x_im, color='red', alpha=0.7)
        ax1[1, 1].set_title("MMSE эквалайзинг")

        #
        f2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
        ax2.scatter(y_re_mmse, y_im_mmse, label='MMSE', s=5, alpha=0.7)
        ax2.scatter(y_zre, y_zim, color='red', label='ZF', s=5, alpha=0.7)
        ax2.scatter(x_re, x_im, color='black', s=10, alpha=0.7)
        ax2.set_title('MMSE и ZF');
        ax2.legend(markerscale=10)

        #
        f3, ax3 = plt.subplots(1, 3, figsize=(15, 5))
        ax3[0].scatter(x_re, x_im, color='red', s=100, alpha=0.7)
        ax3[0].set_title("Исходный сигнал QAM")
        ax3[1].scatter(y_re, y_im, color='blue', s=0.5, alpha=0.7)
        ax3[1].scatter(x_re, x_im, color='red', s=10, alpha=0.7)
        ax3[1].set_title(f"Принятый сигнал, SNR = {self.SNR_db} Дб")
        ax3[2].scatter(y_re_mmse, y_im_mmse, label='MMSE', s=0.5, alpha=1)
        ax3[2].scatter(y_zre, y_zim, color='red', label='ZF', s=0.5, alpha=1)
        ax3[2].scatter(x_re, x_im, color='black', s=10, alpha=1)
        ax3[2].set_title('MMSE и ZF ');
        ax3[2].legend(markerscale=5)
        ax3[2].set_xlim(-10, 10)
        ax3[2].set_ylim(-10, 10)
        plt.show()



qam_1 = QAM(SNR_db=15, M=16, number_ofdm_symbols=20, number_subcarriers=7,impulse_response_length=5)
# #
# qam_1.plot()
# qam_1.calc_evm_ber_snr(np.arange(-5,25,1))
qam_1.impulse_response()
qam_1.modulating()
qam_1.ofdm()
qam_1.power()
qam_1.channel_with_noise()