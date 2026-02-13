import numpy as np
import matplotlib.pyplot as plt
import commpy as cp
from commpy.modulation import QAMModem


plt.style.use('fivethirtyeight') 

class QAM:  # self переменная ссылка на сам класс, глобализирует переменные

    def __init__(self,  SNR_db, M, number_ofdm_symbols, number_subcarriers):
        self.number_subcarriers = number_subcarriers
        self.number_ofdm_symbols = number_ofdm_symbols
        self.number_symbols = None
        self.H = 1
        self.SNR_db = SNR_db
        self.SNR = None
        self.x_qam = None
        self.x_qam_time = None
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
        self.output_bytes = None

    def modulating(self):

        bites_number = int(self.number_subcarriers * np.log2(self.M) * self.number_ofdm_symbols)

        self.bites_number = bites_number
        self.norm = (self.number_ofdm_symbols*self.number_subcarriers*4)

        self.x_bytes = np.random.randint(0, 2, self.bites_number)  # мин, макс , размерность массива. рэндинт - целые

        modem = QAMModem(self.M)

        self.x_qam =np.array( modem.modulate(self.x_bytes))

        self.x_qam = np.array(self.x_qam)

        

        print(f"Число OFDM символов: {self.number_ofdm_symbols}")
        print(f"Число поднесущих: {self.number_subcarriers}")
        print(f"Всего QAM символов: {len(self.x_qam)}")



    def ofdm(self):

        self.ofdm_matrix = self.x_qam.reshape((self.number_ofdm_symbols, self.number_subcarriers))

        self.ofdm_matrix_time = (np.fft.ifft(self.ofdm_matrix, axis=1, norm='ortho'))

        print(f'Размерность OFDM матрицы {self.ofdm_matrix.shape}')





    def power(self):  # нахождение мощности шума через осш в дб



        # signal = set(self.x_qam_time)
        # signal = np.array(list(signal))

        power_x = np.mean(np.abs(self.ofdm_matrix_time) ** 2)
        SNR = 10 ** (self.SNR_db / 10)
        self.SNR = SNR
        self.power_noise = power_x / SNR

        print(f"Мощность сигнала: {power_x}")
        print(f"SNR (линейное): {SNR}")
        print(f"Мощность шума: {self.power_noise}")




    def channel_with_noise(self):  # без импульсной характеристики



        y_time = np.zeros((self.number_ofdm_symbols, self.number_subcarriers), dtype=complex)

        self.power()

        noise = (np.random.randn(self.number_ofdm_symbols, self.number_subcarriers)
                 + 1j * np.random.randn(self.number_ofdm_symbols, self.number_subcarriers)) \
                * np.sqrt(self.power_noise / 2)

        y_time = self.ofdm_matrix_time + noise

        self.y_time = y_time
        self.y_freq = (np.fft.fft(y_time, axis=1,norm='ortho'))
        self.y = self.y_freq.reshape(-1)

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

    def ber(self, dec_bytes):  # ошиька на бит

        count = 0

        for i in range(self.bites_number):
            if dec_bytes[i] != self.x_bytes[i]:
                count += 1

        ber = count / self.bites_number
        return ber

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

        f1, ax1 = plt.subplots(2, 2, figsize=(10, 10))
        ax1[0, 0].scatter(x_re, x_im, color='red', s=10)
        ax1[0, 0].grid()
        ax1[0, 0].set_xlabel('I')
        ax1[0, 0].set_ylabel('Q')
        ax1[0, 0].set_title("Исходный сигнал")

        ax1[0, 1].scatter(y_re, y_im, s=1,alpha=0.7)
        ax1[0, 1].scatter(x_re, x_im, color='red', alpha=0.7)
        ax1[0, 1].set_xlabel('I')
        ax1[0, 1].set_ylabel('Q')
        ax1[0, 1].set_title("Принятый сигнал")
        
        ax1[1, 0].scatter(y_zre, y_zim, s=1,alpha=0.7)
        ax1[1, 0].scatter(x_re, x_im, color='red', alpha=0.7)
        ax1[1, 0].set_xlabel('I')
        ax1[1, 0].set_ylabel('Q')
        ax1[1, 0].set_title("ZF эквалайзинг")
        

        ax1[1, 1].scatter(y_re_mmse, y_im_mmse, s=1,alpha=0.7)
        ax1[1, 1].scatter(x_re, x_im, color='red',alpha=0.7)
        ax1[1, 1].set_xlabel('I')
        ax1[1, 1].set_ylabel('Q')
        ax1[1, 1].set_title("MMSE эквалайзинг")
        


        f2, ax2 = plt.subplots(1, 1, figsize=(10, 10))


        ax2.scatter(y_re_mmse, y_im_mmse, label='MMSE', s=1, alpha=0.7)
        ax2.scatter(y_zre, y_zim, color='red', label='ZF', s=1, alpha=0.7)
        ax2.scatter(x_re, x_im, color='black', s=10, alpha=0.7)
        ax2.set_xlabel('I')
        ax2.set_ylabel('Q')
        ax2.set_xlabel('I')
        ax2.set_ylabel('Q')
        ax2.set_title('MMSE и ZF')
        ax2.legend()

        f3, ax3 = plt.subplots(1, 3, figsize=(10, 10))

        ax3[0].scatter(x_re, x_im, color='red', s=100,alpha=0.7)
        
        ax3[0].set_xlabel('I')
        ax3[0].set_ylabel('Q')
        ax3[0].set_title("Исходный сигнал BPSK")

        ax3[1].scatter(y_re, y_im, color='blue', s=0.5,alpha=0.7)
        ax3[1].scatter(x_re, x_im, color='red', s=10,alpha=0.7)
        
        ax3[1].set_xlabel('I')
        ax3[1].set_ylabel('Q')
        ax3[1].set_title(f"Принятый сигнал, H ={self.H}, SNR = {self.SNR_db} Дб")

        ax3[2].scatter(y_re_mmse, y_im_mmse, label='MMSE', s=0.5,alpha=0.7)
        ax3[2].scatter(y_zre, y_zim, color='red', label='ZF', s=0.5,alpha=0.7)
        ax3[2].scatter(x_re, x_im, color='black', s=10,alpha=0.7)
        ax3[2].set_xlabel('I')
        ax3[2].set_ylabel('Q')
        ax3[2].set_xlabel('I')
        ax3[2].set_ylabel('Q')
        ax3[2].set_title('MMSE и ZF ')
        ax3[2].legend()

        plt.show()


qam_1 = QAM( SNR_db=15, M=16, number_ofdm_symbols=200, number_subcarriers=200)
qam_1.plot()
# qam_1.modulating()
# qam_1.ofdm()
# qam_1.power()
# qam_1.channel_with_noise()
# qam_1.zero_forcing()
# qam_1.mmse()
