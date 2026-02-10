import numpy as np
import commpy as cp
import matplotlib.pyplot as plt


class QAMModulation:
    def __init__(self, band_width, subcarrier_spacing, size_of_modulation, SNR=None):
        self.band_width = band_width
        self.subcarrier_spacing = subcarrier_spacing
        self.M = size_of_modulation
        self.SNR = SNR

        self.subcarriers_total = band_width // subcarrier_spacing
        self.subcarriers_active = int(self.subcarriers_total - 1)
        self.number_of_symbols = int(self.subcarriers_active * np.log2(M) * 200)

        self.input_vector_of_bits = self.create_vector_of_bits()
        if SNR == None:
            self.power_of_noise = np.random.randint(1, 500)*1e-3
        else:
            self.power_of_noise = 10**(-SNR / 10)
        self.h = [0.5, 0.5j, 0.5, -0.5j]

        if len(self.h) != self.subcarriers_total:
            self.h = np.concatenate([self.h, np.zeros(int(self.subcarriers_total - len(self.h)))], axis=0)

        self.H = self.DFT(np.array([self.h]))[0]* np.sqrt(len(self.h))


    def create_vector_of_bits(self):
        return np.random.randint(0, 2, self.number_of_symbols)

    def add_OFDM(self, input_signal_f):
        plt.close()
        #Построение исходного созвездия

        M_10 = list(np.arange(0, self.M))

        M_2 = list(map(lambda x: format(x, f'0{int(np.log2(self.M))}b'), M_10))

        input_vector_of_bits = self.input_vector_of_bits.reshape(-1, int(np.log2(self.M)))

        input_vector_of_bits_to_points = []

        for point in input_vector_of_bits:
            p = ''
            for bit in point:
                p += str(bit)
            input_vector_of_bits_to_points.append(p)

        plt.subplot(221)
        plt.title(f'Сигнал ($X$) (QAM{self.M} модуляция) (количество символов = ${self.number_of_symbols}$)')
        for k in M_2:
            index = input_vector_of_bits_to_points.index(k)
            plt.text(np.real(input_signal_f)[index] + 0.1, np.imag(input_signal_f)[index] + 0.1, k)
            plt.scatter(np.real(input_signal_f)[index], np.imag(input_signal_f)[index], color='red', s=40)
        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)
        plt.xlabel('I', fontsize=16)
        plt.ylabel('Q', fontsize=16)

        # Переход к OFDM (деление на поднесущие частоты)
        input_OFDM_signal_f_active = input_signal_f.reshape(len(input_signal_f) // self.subcarriers_active,
                                                            self.subcarriers_active)

        # Добавление неактивной поднесущей частоты (центральная)
        input_OFDM_signal_f_total = np.zeros((input_OFDM_signal_f_active.shape[0],
                                              input_OFDM_signal_f_active.shape[1] + 1), dtype=complex)

        for m, signal in enumerate(input_OFDM_signal_f_active):
            input_OFDM_signal_f_total[m] = np.concatenate(
                [signal[:self.subcarriers_active // 2], [0 + 0j],
                 signal[self.subcarriers_active // 2:]])

        # Обратное дискретное преобразование Фурье

        input_OFDM_signal_t = self.IDFT(input_OFDM_signal_f_total)

        # Симуляция канала с channel_matrix и AWGN

        output_OFDM_signal_t = self.signal_through_the_channel(input_OFDM_signal_t)

        # Прямое дискретное преобразование Фурье

        output_OFDM_signal_f_total = self.DFT(output_OFDM_signal_t)

        # Извлечение мощности шума из неактивной частоты

        AWGN_power = self.get_power_of_noise(output_OFDM_signal_f_total)

        # Для визуализации без эквалайзера

        output_OFDM_signal_f_active = np.zeros((output_OFDM_signal_f_total.shape[0],
                                                output_OFDM_signal_f_total.shape[-1] - 1), dtype=complex)

        for m, signal in enumerate(output_OFDM_signal_f_total):
            output_OFDM_signal_f_active[m] = np.concatenate([signal[:self.subcarriers_active // 2],
                                                             signal[self.subcarriers_active // 2 + 1:]])

        output_signal_f = output_OFDM_signal_f_active.reshape(1, len(output_OFDM_signal_f_active)
                                                              * self.subcarriers_active)[0]

        plt.subplot(222)
        plt.title(f'Сигнал ($Y$) ($SNR$ $=$ {round(-10 * np.log10(self.power_of_noise), 2)} $dB$)')
        plt.scatter(np.real(output_signal_f), np.imag(output_signal_f), color='black', s=40,
                    label='Сигнал $Y=H \cdot X+N$')
        plt.scatter(np.real(input_signal_f), np.imag(input_signal_f), color='red', s=40,
                    label='Сигнал $X$')
        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)
        plt.xlabel('I', fontsize=16)
        plt.ylabel('Q', fontsize=16)
        plt.legend()

        # Эквализация без смещения

        output_OFDM_signal_f_total_no_unbiasing = self.mmse_equalizer(output_OFDM_signal_f_total, AWGN_power)

        output_OFDM_signal_f_active = np.zeros((output_OFDM_signal_f_total_no_unbiasing.shape[0],
                                                output_OFDM_signal_f_total_no_unbiasing.shape[-1] - 1), dtype=complex)

        for m, signal in enumerate(output_OFDM_signal_f_total_no_unbiasing):
            output_OFDM_signal_f_active[m] = np.concatenate([signal[:self.subcarriers_active // 2],
                                                             signal[self.subcarriers_active // 2 + 1:]])

        output_signal_f_no_unbiasing = output_OFDM_signal_f_active.reshape(1, len(output_OFDM_signal_f_active)
                                                              * self.subcarriers_active)[0]

        # Эквализация со смещением

        output_OFDM_signal_f_total = self.mmse_equalizer(output_OFDM_signal_f_total, AWGN_power, True)

        # Удаление неактивной поднесущей частоты (центральная)

        output_OFDM_signal_f_active = np.zeros((output_OFDM_signal_f_total.shape[0],
                                                output_OFDM_signal_f_total.shape[-1] - 1), dtype=complex)

        for m, signal in enumerate(output_OFDM_signal_f_total):
            output_OFDM_signal_f_active[m] = np.concatenate([signal[:self.subcarriers_active // 2],
                                                             signal[self.subcarriers_active // 2 + 1:]])

        # Переход из OFDM (деление на поднесущие частоты)

        output_signal_f = output_OFDM_signal_f_active.reshape(1, len(output_OFDM_signal_f_active)
                                                              * self.subcarriers_active)[0]

        plt.subplot(223)
        plt.title('Сигнал $\hat{x}$ со и без смещения ($\gamma$)')
        plt.scatter(np.real(output_signal_f_no_unbiasing), np.imag(output_signal_f_no_unbiasing), color='blue', s=40,
                    label='Сигнал $\hat{X} = Y \cdot W_{MMSE}$')
        plt.scatter(np.real(output_signal_f), np.imag(output_signal_f), color='green', s=40,
                    label='Сигнал $\hat{X} = Y \cdot W_{MMSE} \cdot \gamma$')
        plt.scatter(np.real(input_signal_f), np.imag(input_signal_f), color='red', s=40,
                    label='Сигнал $X$')
        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)
        plt.xlabel('I', fontsize=16)
        plt.ylabel('Q', fontsize=16)
        plt.legend()

        epsilon = np.abs(AWGN_power[-1] - self.power_of_noise) / self.power_of_noise

        plt.subplot(224)
        plt.title(f'Определение мощности AWGN ($\epsilon = {round(epsilon*100, 1)}$ $\%$)')
        plt.plot(np.arange(1, len(AWGN_power) + 1), AWGN_power, color='black',
                 label='calculated')
        plt.plot(np.arange(1, len(AWGN_power) + 1), np.ones(len(AWGN_power)) * self.power_of_noise, color='red',
                 label='real')
        plt.xlabel('Отсчёт', fontsize=16)
        plt.ylabel('Мощность', fontsize=16)
        plt.legend()
        plt.grid()

        if self.SNR == -1:
            plt.show()
        elif self.SNR == 3:
            plt.show()
        elif self.SNR == 7:
            plt.show()
        elif self.SNR == 11:
            plt.show()
        elif self.SNR == 15:
            plt.show()

        return output_signal_f, output_signal_f_no_unbiasing

    def modulation_signal(self):
        return cp.QAMModem(self.M).modulate(self.input_vector_of_bits)

    def demodulation_signal(self, output_signal_f):
        return cp.QAMModem(self.M).demodulate(output_signal_f, demod_type='hard')

    @staticmethod
    def IDFT(input_OFDM_signal_f_total):
        N = len(input_OFDM_signal_f_total[0])
        input_OFDM_signal_t = np.zeros(input_OFDM_signal_f_total.shape, dtype=complex)

        for m, signal_f in enumerate(input_OFDM_signal_f_total):
            for k in range(len(input_OFDM_signal_f_total[m])):
                for n in range(len(signal_f)):
                    input_OFDM_signal_t[m][k] += signal_f[n] * np.exp(1j * 2 * np.pi * k * n / N)
                input_OFDM_signal_t[m][k] /= np.sqrt(N)
                if np.abs(np.real(input_OFDM_signal_t[m][k])) < 1e-10:
                    input_OFDM_signal_t[m][k] = 0 + 1j * np.imag(input_OFDM_signal_t[m][k])
                if np.abs(np.imag(input_OFDM_signal_t[m][k])) < 1e-10:
                    input_OFDM_signal_t[m][k] = np.real(input_OFDM_signal_t[m][k]) + 0j

        return input_OFDM_signal_t

    def signal_through_the_channel(self, signal):
        K = len(self.h)
        N = len(signal[0])

        signal_cp = np.concatenate([signal[:, -K + 1:], signal], axis=1)
        new_signal = np.zeros(signal_cp.shape, dtype=complex)

        # Циклическая свёртка h и x

        for s in range(len(signal)):
            for n in range(N):
                for m in range(K):
                    new_signal[s, n] += self.h[m] * signal_cp[s, n - m + K - 1]

        new_signal += ((np.random.randn(*new_signal.shape) + 1j * np.random.randn(*new_signal.shape))
                       * np.sqrt(self.power_of_noise / 2))

        return new_signal[:, :-(K - 1)]

    @staticmethod
    def DFT(output_OFDM_signal_t):
        N = len(output_OFDM_signal_t[0])
        output_OFDM_signal_f_total = np.zeros(output_OFDM_signal_t.shape, dtype=complex)

        for m, signal_t in enumerate(output_OFDM_signal_t):
            for k in range(len(output_OFDM_signal_t[m])):
                for n in range(len(signal_t)):
                    output_OFDM_signal_f_total[m][k] += signal_t[n] * np.exp(
                        -1j * 2 * np.pi * k * n / N)
                output_OFDM_signal_f_total[m][k] /= np.sqrt(N)
                if np.abs(np.real(output_OFDM_signal_f_total[m][k])) < 1e-10:
                    output_OFDM_signal_f_total[m][k] = 0 + 1j * np.imag(output_OFDM_signal_f_total[m][k])
                if np.abs(np.imag(output_OFDM_signal_f_total[m][k])) < 1e-10:
                    output_OFDM_signal_f_total[m][k] = np.real(output_OFDM_signal_f_total[m][k]) + 0j

        return output_OFDM_signal_f_total

    def get_power_of_noise(self, output_OFDM_signal_f_total):
        AWGN_power = np.zeros(output_OFDM_signal_f_total.shape[0])

        s = 1
        sum_power = 0
        for signal in output_OFDM_signal_f_total:
            sum_power += np.abs(signal[int(self.subcarriers_total // 2)]) ** 2
            AWGN_power[s - 1] = sum_power / s
            s += 1

        return AWGN_power

    def mmse_equalizer(self, output_OFDM_signal_f_active, AWGN_power, unbiasing=False):
        output_OFDM_signal_f_active_eq = np.zeros(output_OFDM_signal_f_active.shape, dtype=complex)
        for m in range(len(output_OFDM_signal_f_active)):
            output_OFDM_signal_f_active_eq[m] = (output_OFDM_signal_f_active[m] * np.conj(self.H).T
                                                 / (np.abs(self.H)**2 + AWGN_power[m]))

            if unbiasing:
                alpha = (np.abs(self.H)**2 + AWGN_power[m]) / np.abs(self.H)**2

                output_OFDM_signal_f_active_eq[m] *= alpha

        return output_OFDM_signal_f_active_eq

    def BER_count(self, output_vector_of_bits):
        return np.sum((self.input_vector_of_bits + output_vector_of_bits) % 2) / len(self.input_vector_of_bits)

    @staticmethod
    def EVM_count(input_signal, output_signal):
        return np.mean(np.abs(input_signal - output_signal) ** 2)

    def SISO_with_QAM_simulation(self):
        input_signal_f = self.modulation_signal()

        output_signal_f, output_signal_f_no_unbiasing = self.add_OFDM(input_signal_f)

        output_vector_of_bits = self.demodulation_signal(output_signal_f)
        output_vector_of_bits_no_unbiasing = self.demodulation_signal(output_signal_f_no_unbiasing)

        BER = self.BER_count(output_vector_of_bits)

        EVM = self.EVM_count(input_signal_f, output_signal_f)

        BER_no_unbiasing = self.BER_count(output_vector_of_bits_no_unbiasing)

        EVM_no_unbiasing = self.EVM_count(input_signal_f, output_signal_f_no_unbiasing)

        return (BER, EVM), (BER_no_unbiasing, EVM_no_unbiasing)



bandwidth = 105e3 # 100 kHz
spacing = 15e3 # 15 kHz
subcarriers_total = bandwidth // spacing

symbol_duration = 1 / spacing # ~66.67 μs
cp_duration = 4.69e-6         # 4.69 μs
total_duration = symbol_duration + cp_duration     # ~71.36 μs

sampling_frequency = subcarriers_total / symbol_duration
number_of_CP = cp_duration * sampling_frequency
M = 64

SNR_arr = np.arange(-3, 21)

with_unbiasing = list()
without_unbiasing = list()

for k, SNR in enumerate(SNR_arr):
    example = QAMModulation(band_width=bandwidth,
                            subcarrier_spacing=spacing,
                            size_of_modulation=M,
                            SNR=SNR)

    with_un, without_un = example.SISO_with_QAM_simulation()

    with_unbiasing.append([with_un[0], with_un[1]])
    without_unbiasing.append([without_un[0], without_un[1]])
    plt.close()

plt.subplot(121)
plt.title('BER(SNR)')
plt.semilogy(SNR_arr, np.array(with_unbiasing)[:, 0], color='green', label='With unbiasing')
plt.semilogy(SNR_arr, np.array(without_unbiasing)[:, 0], color='blue', label='Without unbiasing')
plt.legend()
plt.xlabel('SNR, dB', fontsize=16)
plt.ylabel('BER', fontsize=16)

plt.subplot(122)
plt.title('EVM(SNR)')
plt.semilogy(SNR_arr, np.array(with_unbiasing)[:, 1], color='green', label='With unbiasing')
plt.semilogy(SNR_arr, np.array(without_unbiasing)[:, 1], color='blue', label='Without unbiasing')
plt.legend()
plt.xlabel('SNR, dB', fontsize=16)
plt.ylabel('EVM', fontsize=16)

plt.show()