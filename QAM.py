import commpy as cp
import numpy as np
import matplotlib.pyplot as plt
from commpy.modulation import QAMModem


class QAM:  # self переменная ссылка на сам класс, глобализирует переменные


    def __init__(self, number_symbols, H, SNR_db, M):
        self.number_symbols = number_symbols
        self.H = H
        self.SNR_db = SNR_db
        self.SNR = None
        self.x = None
        self.y = None
        self.bites_number = None
        self.M = M
        self.power_noise = None
        self.x_bytes = None
        self.y_bytes = None
        self.y_zf = None
        self.output_bytes = None


    def modulating(self):  # модуляция 0 эти -1 , 1 это 1

        bites_number = int(self.number_symbols * np.log2(self.M))
        self.bites_number = bites_number

        self.x_bytes = np.random.randint(0, 2, self.bites_number)  # мин, макс , размерность массива. рэндинт - целые

        modem = QAMModem(self.M)

        x = modem.modulate(self.x_bytes)

        self.x = np.array(x)


    def power(self):  # нахождение мощности шума через осш в дб

        self.modulating()

        signal = set(self.x)
        signal = np.array(list(signal))

        power_x = np.mean(np.abs(signal) ** 2)
        SNR = 10 ** (self.SNR_db / 10)
        self.SNR = SNR
        self.power_noise = power_x / SNR

        print(f"Мощность сигнала: {power_x}")
        print(f"SNR (линейное): {SNR}")
        print(f"Мощность шума: {self.power_noise}")


    def channel_with_noise(self):  # через плотность моделируем случайный шум, добавляем к H*x

        self.power()

        y = np.zeros(self.number_symbols, dtype=complex)

        for i in range(self.number_symbols):
            noise = (np.random.randn() + 1j * np.random.randn()) * np.sqrt(self.power_noise / 2)
            y[i] = self.H * self.x[i] + noise

        self.y = y


    def zero_forcing(self):

        self.y_zf = ((np.conjugate(self.H) / (np.abs(self.H)) ** 2)) * self.y


    def mmse(self):

        self.y_mmse = self.y * (np.conjugate(self.H) / ((np.abs(self.H)) ** 2 + self.power_noise))


    def decod(self, output):

        output_bytes = np.zeros(self.bites_number)
        modem = QAMModem(self.M)

        output_bytes = modem.demodulate(output, demod_type='hard')

        self.output_bytes = np.array(output_bytes)

        return output_bytes


    def ber(self, dec_bytes):

        count = 0

        for i in range(self.bites_number):
            if dec_bytes[i] != self.x_bytes[i]:
                count += 1

        ber = count / self.bites_number
        return ber


    def ber_snr(self, snr_db_range):  # написал не сам, надо пеерписать по-хорошему

        ber_zf = []
        ber_mmse = []

        for snr_db in snr_db_range:
            qam = QAM(
                number_symbols=self.number_symbols,
                H=self.H,
                SNR_db=snr_db,
                M=self.M
            )

            qam.channel_with_noise()

            qam.zero_forcing()
            dec_zf = qam.decod(qam.y_zf)
            ber_zf.append(qam.ber(dec_zf))

            qam.mmse()
            dec_mmse = qam.decod(qam.y_mmse)
            ber_mmse.append(qam.ber(dec_mmse))

        plt.figure(figsize=(8, 6))
        plt.semilogy(snr_db_range, ber_zf, label='ZF')
        plt.semilogy(snr_db_range, ber_mmse, label='MMSE')
        plt.grid(True, which='both')
        plt.xlabel("SNR, dB")
        plt.ylabel("BER")
        plt.title("BER от SNR (ZF и MMSE)")
        plt.legend()
        plt.show()


    def evm(self, output):

        return np.mean(np.abs(output - self.x) ** 2)


    def evm_snr(self, snr_db_range):

        evm_zf = []
        evm_mmse = []

        for i in snr_db_range:
            qam = QAM(number_symbols=self.number_symbols, H=self.H, SNR_db=i, M=self.M)

            qam.channel_with_noise()

            qam.zero_forcing()
            evm_zf.append(qam.evm(qam.y_zf))

            qam.mmse()
            evm_mmse.append(qam.evm(qam.y_mmse))

        plt.figure(figsize=(8, 6))
        plt.semilogy(snr_db_range, evm_zf, label='ZF')
        plt.semilogy(snr_db_range, evm_mmse, label='MMSE')
        plt.grid(True, which='both')
        plt.xlabel("SNR, dB")
        plt.ylabel("EVM")
        plt.title("EVM (ZF и MMSE)")
        plt.legend()
        plt.show()


    def plot(self):

        self.channel_with_noise()

        self.zero_forcing()
        self.mmse()

        decod_yzf = self.decod(self.y_zf)
        decod_y_mmse = self.decod(self.y_mmse)

        ber_yzf = self.ber(decod_yzf)
        print(ber_yzf)

        ber_ymmse = self.ber(decod_y_mmse)
        print(ber_ymmse)

        x_re = self.x.real
        x_im = self.x.imag

        y_re = self.y.real
        y_im = self.y.imag

        y_zre = self.y_zf.real
        y_zim = self.y_zf.imag

        y_re_mmse = self.y_mmse.real
        y_im_mmse = self.y_mmse.imag

        f1, ax1 = plt.subplots(2, 2, figsize=(10, 10))
        ax1[0, 0].scatter(x_re, x_im, color='red', s=10)
        ax1[0, 0].grid()
        ax1[0, 0].set_xlabel('I')
        ax1[0, 0].set_ylabel('Q')
        ax1[0, 0].set_title("Исходный сигнал")

        ax1[0, 1].scatter(y_re, y_im, s=1)
        ax1[0, 1].scatter(x_re, x_im, color='red')
        ax1[0, 1].set_xlabel('I')
        ax1[0, 1].set_ylabel('Q')
        ax1[0, 1].set_title("Принятый сигнал")
        ax1[0, 1].grid()

        ax1[1, 0].scatter(y_zre, y_zim, s=1)
        ax1[1, 0].scatter(x_re, x_im, color='red')
        ax1[1, 0].set_xlabel('I')
        ax1[1, 0].set_ylabel('Q')
        ax1[1, 0].set_title("ZF эквалайзинг")
        ax1[1, 0].grid()

        ax1[1, 1].scatter(y_re_mmse, y_im_mmse, s=1)
        ax1[1, 1].scatter(x_re, x_im, color='red')
        ax1[1, 1].set_xlabel('I')
        ax1[1, 1].set_ylabel('Q')
        ax1[1, 1].set_title("MMSE эквалайзинг")
        ax1[1, 1].grid()
        f2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
        ax2.scatter(y_re_mmse, y_im_mmse, label='MMSE', s=1)
        ax2.scatter(y_zre, y_zim, color='red', label='ZF', s=1)
        ax2.scatter(x_re, x_im, color='black', s=10)
        ax2.set_xlabel('I')
        ax2.set_ylabel('Q')
        ax2.set_xlabel('I')
        ax2.set_ylabel('Q')
        ax2.set_title('MMSE и ZF')
        ax2.legend()

        f3, ax3 = plt.subplots(1, 3, figsize=(10, 10))

        ax3[0].scatter(x_re, x_im, color='red', s=100)
        ax3[0].grid()
        ax3[0].set_xlabel('I')
        ax3[0].set_ylabel('Q')
        ax3[0].set_title("Исходный сигнал BPSK")

        ax3[1].scatter(y_re, y_im, color='blue', s=1)
        ax3[1].scatter(x_re, x_im, color='red', s=10)
        ax3[1].grid()
        ax3[1].set_xlabel('I')
        ax3[1].set_ylabel('Q')
        ax3[1].set_title(f"Принятый сигнал, H ={self.H}, SNR = {self.SNR_db} Дб")

        ax3[2].scatter(y_re_mmse, y_im_mmse, label='MMSE', s=1)
        ax3[2].scatter(y_zre, y_zim, color='red', label='ZF', s=1)
        ax3[2].scatter(x_re, x_im, color='black', s=10)
        ax3[2].set_xlabel('I')
        ax3[2].set_ylabel('Q')
        ax3[2].set_xlabel('I')
        ax3[2].set_ylabel('Q')
        ax3[2].set_title('MMSE и ZF ')
        ax3[2].legend()

        plt.show()


qam_1 = QAM(number_symbols=10000, H=0.7 + 0.7j, SNR_db=1, M=16)
qam_1.plot()
# qam_1.ber_snr(np.arange(-20, 6, 0.1))
#
#
# qam_1.evm_snr(np.arange(-20, 6, 0.1))
