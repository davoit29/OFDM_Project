import numpy as np

import matplotlib.pyplot as plt
import commpy as cp
from commpy.modulation import QAMModem
import commpy.channelcoding.convcode as cc
from concurrent.futures import ProcessPoolExecutor


def _run_sim(snr, M, number_ofdm_symbols, number_subcarriers):
    sim = QAM(SNR_db=snr, M=M, number_ofdm_symbols=number_ofdm_symbols, number_subcarriers=number_subcarriers)
    return sim.ber_until_100(snr)


class QAM:
    def __init__(self, SNR_db, M, number_ofdm_symbols, number_subcarriers):
        self.number_subcarriers = number_subcarriers
        self.number_ofdm_symbols = number_ofdm_symbols
        self.M = M
        self.SNR_db = SNR_db
        self.SNR = None

        # Витерби 1/2
        self.trellis = cc.Trellis(memory=np.array([6]), g_matrix=np.array([[0o133, 0o171]]))

        # = параметры временного моделирования
        self.delta_f = 15 * 10 ** 3
        self.bandwidth = 105 * 10 ** 3
        self.Fs = 1e8
        self.fc = 300 * 10 ** 3

    def impulse_response(self, diagram=False):
        tau_us = np.array([0, 3, 5, 6, 8])  # мкс
        power_dB = np.array([0, -8, -17, -21, -25])
        power = 10 ** (power_dB / 10)
        amplitude = np.sqrt(power)
        phase = np.zeros_like(tau_us, dtype=complex)
        N = len(phase)
        for k in range(N):
            phase[k] = 2 * np.pi * np.random.randint(0, N - 1) / N
        h = amplitude * np.exp(1j * phase)
        if diagram:
            plt.figure()
            plt.title("Импульсная характеристика рассматриваемого канала", fontsize=16)
            plt.stem(tau_us, amplitude, linefmt="black")
            plt.xlabel("Отсчёт, мкс", fontsize=14)
            plt.ylabel("Амплитуда", fontsize=14)
            plt.show()
        return h

    def modulating(self):
        bites_per_antenna = self.number_subcarriers * self.number_ofdm_symbols * int(np.log2(self.M))

        # без кодирования — 4 антенны
        self.x_bytes_a_unc = np.random.randint(0, 2, bites_per_antenna)
        self.x_bytes_b_unc = np.random.randint(0, 2, bites_per_antenna)
        self.x_bytes_c_unc = np.random.randint(0, 2, bites_per_antenna)
        self.x_bytes_d_unc = np.random.randint(0, 2, bites_per_antenna)
        self.x_bytes_unc = np.concatenate((self.x_bytes_a_unc, self.x_bytes_b_unc,
                                           self.x_bytes_c_unc, self.x_bytes_d_unc))

        # с кодированием — 4 антенны
        info_bits_per_antenna = bites_per_antenna // 2
        self.info_bytes_a = np.random.randint(0, 2, info_bits_per_antenna)
        self.info_bytes_b = np.random.randint(0, 2, info_bits_per_antenna)
        self.info_bytes_c = np.random.randint(0, 2, info_bits_per_antenna)
        self.info_bytes_d = np.random.randint(0, 2, info_bits_per_antenna)
        self.info_bytes = np.concatenate((self.info_bytes_a, self.info_bytes_b,
                                          self.info_bytes_c, self.info_bytes_d))

        self.x_bytes_a_cod = cc.conv_encode(self.info_bytes_a, self.trellis, termination='cont')
        self.x_bytes_b_cod = cc.conv_encode(self.info_bytes_b, self.trellis, termination='cont')
        self.x_bytes_c_cod = cc.conv_encode(self.info_bytes_c, self.trellis, termination='cont')
        self.x_bytes_d_cod = cc.conv_encode(self.info_bytes_d, self.trellis, termination='cont')

        # модуляция
        modem = QAMModem(self.M)
        self.x_qam_first_unc = np.array(modem.modulate(self.x_bytes_a_unc))
        self.x_qam_second_unc = np.array(modem.modulate(self.x_bytes_b_unc))
        self.x_qam_third_unc = np.array(modem.modulate(self.x_bytes_c_unc))
        self.x_qam_fourth_unc = np.array(modem.modulate(self.x_bytes_d_unc))
        self.x_qam_unc = np.concatenate((self.x_qam_first_unc, self.x_qam_second_unc,
                                         self.x_qam_third_unc, self.x_qam_fourth_unc))

        self.x_qam_first_cod = np.array(modem.modulate(self.x_bytes_a_cod))
        self.x_qam_second_cod = np.array(modem.modulate(self.x_bytes_b_cod))
        self.x_qam_third_cod = np.array(modem.modulate(self.x_bytes_c_cod))
        self.x_qam_fourth_cod = np.array(modem.modulate(self.x_bytes_d_cod))
        self.x_qam_cod = np.concatenate((self.x_qam_first_cod, self.x_qam_second_cod,
                                         self.x_qam_third_cod, self.x_qam_fourth_cod))

    def ofdm(self):
        # Без кодирования
        self.x_ofdm_tensor_unc = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 4), dtype=complex)
        self.x_ofdm_tensor_unc[:, :, 0] = self.x_qam_first_unc.reshape(self.number_ofdm_symbols,
                                                                       self.number_subcarriers)
        self.x_ofdm_tensor_unc[:, :, 1] = self.x_qam_second_unc.reshape(self.number_ofdm_symbols,
                                                                        self.number_subcarriers)
        self.x_ofdm_tensor_unc[:, :, 2] = self.x_qam_third_unc.reshape(self.number_ofdm_symbols,
                                                                       self.number_subcarriers)
        self.x_ofdm_tensor_unc[:, :, 3] = self.x_qam_fourth_unc.reshape(self.number_ofdm_symbols,
                                                                        self.number_subcarriers)

        # С кодированием
        self.x_ofdm_tensor_cod = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 4), dtype=complex)
        self.x_ofdm_tensor_cod[:, :, 0] = self.x_qam_first_cod.reshape(self.number_ofdm_symbols,
                                                                       self.number_subcarriers)
        self.x_ofdm_tensor_cod[:, :, 1] = self.x_qam_second_cod.reshape(self.number_ofdm_symbols,
                                                                        self.number_subcarriers)
        self.x_ofdm_tensor_cod[:, :, 2] = self.x_qam_third_cod.reshape(self.number_ofdm_symbols,
                                                                       self.number_subcarriers)
        self.x_ofdm_tensor_cod[:, :, 3] = self.x_qam_fourth_cod.reshape(self.number_ofdm_symbols,
                                                                        self.number_subcarriers)

    def power_of_signal(self):
        return 2 / 3 * (self.M - 1)

    def _apply_channel_with_pilots(self, x_ofdm_tensor):
        # 1 дата + 4 пилота на каждые 5 OFDM-символов
        # матрица пилотов 4x4:
        P_mat = np.array([[1, 1, 1, 1],
                          [1, -1, 1, -1],
                          [1, 1, -1, -1],
                          [1, -1, -1, 1]], dtype=complex)
        P_inv = P_mat.T / 4

        res = np.zeros((5 * self.number_ofdm_symbols, self.number_subcarriers, 4), dtype=complex)
        res[0::5, :, :] = x_ofdm_tensor  # каждая пятая строка — инфа
        # остальные 4 строки — пилотные символы по матрице P_mat
        for p in range(4):
            for ant in range(4):
                res[p + 1::5, :, ant] = P_mat[p, ant]

        # теперь между информацией подаётся 4 OFDM-пилотных символа для 4 антенн
        x_time_pilots = np.fft.ifft(res, axis=1) * np.sqrt(self.number_subcarriers)

        def convolution_cp(r, h):
            cp_len = len(h) - 1
            x_cp = np.concatenate((r[-cp_len:], r))
            y_conv = np.convolve(x_cp, h, mode='valid')
            return y_conv

        y_time_pilots = np.zeros_like(x_time_pilots)

        # (канал меняется каждые 5 символов)
        for i in range(self.number_ofdm_symbols):
            # 4x4 матрица импульсных характеристик: h[rx][tx]
            h = [[self.impulse_response() for _ in range(4)] for _ in range(4)]

            for n in range(5):
                idx = 5 * i + n
                tx = [x_time_pilots[idx, :, ant] for ant in range(4)]
                for rx in range(4):
                    # суммируем вклад каждой TX-антенны в rx-ю RX-антенну
                    rx_sig = sum(convolution_cp(tx[tx_ant], h[rx][tx_ant]) for tx_ant in range(4))
                    y_time_pilots[idx, :, rx] = rx_sig

        # шум
        P_signal = self.power_of_signal()
        sigma = np.sqrt(P_signal / (2 * self.SNR))
        noise = np.random.normal(0, sigma, y_time_pilots.shape) + 1j * np.random.normal(0, sigma, y_time_pilots.shape)

        # шум на пилотах = 0
        for idx in range(y_time_pilots.shape[0]):
            if idx % 5 != 0:
                noise[idx, :, :] = 0

        y_time_noisy = y_time_pilots + noise
        Y_OFDM_pilots = np.fft.fft(y_time_noisy, axis=1) / np.sqrt(self.number_subcarriers)

        H_est = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 4, 4), dtype=complex)
        Y_OFDM_data = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 4), dtype=complex)

        
        for i in range(self.number_ofdm_symbols):
            Y_OFDM_data[i] = Y_OFDM_pilots[5 * i]
           
            Y_pilots_mat = np.stack([Y_OFDM_pilots[5 * i + 1 + p] for p in range(4)], axis=0)
            for k in range(self.number_subcarriers):
                Yk = Y_pilots_mat[:, k, :]  # (4 пилота, 4 rx)
                H_est[i, k, :, :] = (P_inv @ Yk).T  # (4 rx, 4 tx)

        return H_est, Y_OFDM_data

    def channel_with_noise(self):
        self.SNR = 10 ** (self.SNR_db / 10)
        self.H_est_unc, self.y_unc = self._apply_channel_with_pilots(self.x_ofdm_tensor_unc)
        self.H_est_cod, self.y_cod = self._apply_channel_with_pilots(self.x_ofdm_tensor_cod)

    def qr_solve(self, A, b):
        # QR-разложение через вращения Гивенса для матрицы 4x4
        R = A.copy().astype(complex)
        rhs = b.copy().astype(complex)
        n = R.shape[0]  # = 4

        # 6 вращений
        for j in range(n):
            for i in range(j + 1, n):
                if abs(R[i, j]) < 1e-15:
                    continue
                r = np.hypot(abs(R[j, j]), abs(R[i, j]))
                if r < 1e-15:
                    continue
                c = R[j, j] / r
                s = R[i, j] / r
                #  вращение Гивенса к строкам j и i матрицы R
                for col in range(j, n):
                    temp = c.conj() * R[j, col] + s.conj() * R[i, col]
                    R[i, col] = -s * R[j, col] + c * R[i, col]
                    R[j, col] = temp
                # то же к правой части
                temp = c.conj() * rhs[j] + s.conj() * rhs[i]
                rhs[i] = -s * rhs[j] + c * rhs[i]
                rhs[j] = temp

        #  подстановка
        x = np.zeros(n, dtype=complex)
        for i in range(n - 1, -1, -1):
            if abs(R[i, i]) < 1e-15:
                x[i] = 0.0
            else:
                x[i] = (rhs[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]
        return x

    def _mmse(self, y_tensor, H_est):
        x_est = np.zeros((y_tensor.shape[0], self.number_subcarriers, 4), dtype=complex)
        for i in range(y_tensor.shape[0]):
            for k in range(self.number_subcarriers):
                Hk = H_est[i, k, :, :]  # 4x4
                yk = y_tensor[i, k, :]  # 4-вектор
                Hh = Hk.conj().T
                A = Hh @ Hk + (1 / self.SNR) * np.eye(4)
                b = Hh @ yk
                x_est[i, k, :] = self.qr_solve(A, b)
        return x_est.transpose(2, 0, 1).reshape(-1)

    def mmse(self):
        self.y_mmse_unc = self._mmse(self.y_unc, self.H_est_unc)
        self.y_mmse_cod = self._mmse(self.y_cod, self.H_est_cod)

    def decode_uncoded(self, output):
        modem = QAMModem(self.M)
        return np.array(modem.demodulate(output, demod_type='hard'))

    def decode_coded(self, output):
        modem = QAMModem(self.M)

        demod_bits = np.array(modem.demodulate(output, demod_type='hard'))
        quarter = len(demod_bits) // 4
        bits_a = demod_bits[0 * quarter:1 * quarter]
        bits_b = demod_bits[1 * quarter:2 * quarter]
        bits_c = demod_bits[2 * quarter:3 * quarter]
        bits_d = demod_bits[3 * quarter:4 * quarter]
        tb_depth = 5 * (self.trellis.total_memory + 1)
        dec_info_a = cc.viterbi_decode(bits_a, self.trellis, tb_depth=tb_depth, decoding_type='hard')
        dec_info_b = cc.viterbi_decode(bits_b, self.trellis, tb_depth=tb_depth, decoding_type='hard')
        dec_info_c = cc.viterbi_decode(bits_c, self.trellis, tb_depth=tb_depth, decoding_type='hard')
        dec_info_d = cc.viterbi_decode(bits_d, self.trellis, tb_depth=tb_depth, decoding_type='hard')
        return np.concatenate((dec_info_a, dec_info_b, dec_info_c, dec_info_d))

    def decode_coded_llr(self, output):
        modem = QAMModem(self.M)
        noise_var = self.power_of_signal() / self.SNR
        llr = modem.demodulate(output, demod_type='soft', noise_var=noise_var)

        llr = np.clip(llr, -20, 20)
        quarter = len(llr) // 4
        llr_a = llr[0 * quarter:1 * quarter]
        llr_b = llr[1 * quarter:2 * quarter]
        llr_c = llr[2 * quarter:3 * quarter]
        llr_d = llr[3 * quarter:4 * quarter]
        tb_depth = 5 * (self.trellis.total_memory + 1)
        dec_a = cc.viterbi_decode(llr_a, self.trellis, tb_depth=tb_depth, decoding_type='unquantized')
        dec_b = cc.viterbi_decode(llr_b, self.trellis, tb_depth=tb_depth, decoding_type='unquantized')
        dec_c = cc.viterbi_decode(llr_c, self.trellis, tb_depth=tb_depth, decoding_type='unquantized')
        dec_d = cc.viterbi_decode(llr_d, self.trellis, tb_depth=tb_depth, decoding_type='unquantized')
        return np.concatenate((dec_a, dec_b, dec_c, dec_d))

    def ber_until_100(self, snr, max_iter=100):
        self.SNR_db = snr
        TARGET_ERRORS = 100
        err = {k: 0 for k in [
            'mmse_unc', 'mmse_cod', 'mmse_llr'
        ]}
        bits = {k: 0 for k in err.keys()}
        evm_mmse_acc = []

        for _ in range(max_iter):
            if all(v >= TARGET_ERRORS for v in err.values()):
                break
            self.modulating()
            self.ofdm()
            self.channel_with_noise()
            self.mmse()

            dec_mmse_unc = self.decode_uncoded(self.y_mmse_unc)
            dec_mmse_cod = self.decode_coded(self.y_mmse_cod)
            dec_mmse_llr = self.decode_coded_llr(self.y_mmse_cod)

            n_unc = len(self.x_bytes_unc)
            n_cod = len(self.info_bytes)

            err['mmse_unc'] += int(np.sum(dec_mmse_unc != self.x_bytes_unc))
            err['mmse_cod'] += int(np.sum(dec_mmse_cod != self.info_bytes))
            err['mmse_llr'] += int(np.sum(dec_mmse_llr != self.info_bytes))

            bits['mmse_unc'] += n_unc
            bits['mmse_cod'] += n_cod
            bits['mmse_llr'] += n_cod

            evm_mmse_acc.append(np.sum(np.abs(self.x_qam_unc - self.y_mmse_unc) ** 2) / len(self.x_qam_unc))

        def safe_ber(e, b):
            return e / b if b > 0 else 0.0

        return (safe_ber(err['mmse_unc'], bits['mmse_unc']),
                safe_ber(err['mmse_cod'], bits['mmse_cod']),
                safe_ber(err['mmse_llr'], bits['mmse_llr']),
                float(np.mean(evm_mmse_acc)) if evm_mmse_acc else 0.0)

    def plot_constellations(self):
        # Отрисовка QAM созвездий 
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.title(f"Modulated signal (QAM{self.M})")
        plt.scatter(np.real(self.x_qam_unc), np.imag(self.x_qam_unc), color="red")
        plt.xlabel("I");
        plt.ylabel("Q")

        plt.subplot(1, 3, 2)
        plt.title(f"Signal after channel with AWGN (SNR={self.SNR_db} dB)")
        y_flat = self.y_unc.flatten()
        plt.scatter(np.real(y_flat), np.imag(y_flat), color="black", s=0.5)
        plt.scatter(np.real(self.x_qam_unc), np.imag(self.x_qam_unc), color="red", s=20)
        plt.xlabel("I");
        plt.ylabel("Q")

        plt.subplot(1, 3, 3)
        plt.title("Equalized signal")
        plt.scatter(np.real(self.y_mmse_unc), np.imag(self.y_mmse_unc), s=0.5, color="green", label="MMSE")
        plt.scatter(np.real(self.x_qam_unc), np.imag(self.x_qam_unc), color="red", s=20)
        plt.xlabel("I");
        plt.ylabel("Q")
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

    def ber_100_avg_plot(self, snr_range, n_iter):
        ber_mmse_unc_avg, ber_mmse_cod_avg, ber_mmse_llr_avg = [], [], []
        evm_mmse_unc_avg = []

        with ProcessPoolExecutor(max_workers=3) as executor:
            for snr in snr_range:
                b_mu, b_mc, b_mllr = [], [], []
                e_mu = []

                futures = [executor.submit(_run_sim, snr, self.M, self.number_ofdm_symbols, self.number_subcarriers) for
                           _ in range(n_iter)]

                for future in futures:
                    res = future.result()
                    b_mu.append(res[0]);
                    b_mc.append(res[1])
                    b_mllr.append(res[2]);
                    e_mu.append(res[3])

                ber_mmse_unc_avg.append(np.mean(b_mu))
                ber_mmse_cod_avg.append(np.mean(b_mc))
                ber_mmse_llr_avg.append(np.mean(b_mllr))
                evm_mmse_unc_avg.append(np.mean(e_mu))
                print(
                    f"SNR {snr:2d} | MMSE Uncoded: {np.mean(b_mu):.3e} | MMSE Coded: {np.mean(b_mc):.3e} | MMSE LLR: {np.mean(b_mllr):.3e}")

        plt.figure(figsize=(12, 5))
        # BER
        plt.subplot(1, 2, 1)
        plt.semilogy(snr_range, ber_mmse_unc_avg, '-', label="MMSE Uncoded", color='orange')
        plt.semilogy(snr_range, ber_mmse_cod_avg, '--', label="MMSE Coded", color='orange')
        plt.semilogy(snr_range, ber_mmse_llr_avg, ':', label="MMSE LLR", color='orange')
        plt.xlabel("SNR (dB)")
        plt.ylabel("BER")
        plt.grid(True, which="both", linestyle='--', alpha=0.6)
        plt.legend(fontsize=8)

        # EVM
        plt.subplot(1, 2, 2)
        plt.semilogy(snr_range, evm_mmse_unc_avg, '-', label="MMSE", color='orange')
        plt.xlabel("SNR (dB)")
        plt.ylabel("EVM")
        plt.grid(True, which="both", linestyle='--', alpha=0.6)
        plt.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig('result_plot.png')
        plt.show()


if __name__ == "__main__":
    qam_1 = QAM(SNR_db=20, M=16, number_ofdm_symbols=20, number_subcarriers=7)
    qam_1.modulating()
    qam_1.ofdm()
    qam_1.channel_with_noise()
    qam_1.mmse()
    qam_1.ber_100_avg_plot(snr_range=np.arange(0, 20, 2), n_iter=5)
