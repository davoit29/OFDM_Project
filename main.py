import numpy as np
import matplotlib.pyplot as plt
import commpy as cp
from commpy.modulation import QAMModem
import commpy.channelcoding.convcode as cc


class QAM:
    def __init__(self, SNR_db, M, number_ofdm_symbols, number_subcarriers):
        self.number_subcarriers = number_subcarriers
        self.number_ofdm_symbols = number_ofdm_symbols
        self.M = M
        self.SNR_db = SNR_db
        self.SNR = None

        #  Витерби  1/2
        self.trellis = cc.Trellis(memory=np.array([6]), g_matrix=np.array([[0o133, 0o171]]))

        # ML
        modem = QAMModem(self.M)
        const = modem.constellation
        self.ml_candidates = np.array([[s1, s2] for s1 in const for s2 in const])

        # = параметры  временного моделирования
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

        # без кодирования
        self.x_bytes_a_unc = np.random.randint(0, 2, bites_per_antenna)
        self.x_bytes_b_unc = np.random.randint(0, 2, bites_per_antenna)
        self.x_bytes_unc = np.concatenate((self.x_bytes_a_unc, self.x_bytes_b_unc))

        # с кодированием
        info_bits_per_antenna = bites_per_antenna // 2
        self.info_bytes_a = np.random.randint(0, 2, info_bits_per_antenna)
        self.info_bytes_b = np.random.randint(0, 2, info_bits_per_antenna)
        self.info_bytes = np.concatenate((self.info_bytes_a, self.info_bytes_b))

        self.x_bytes_a_cod = cc.conv_encode(self.info_bytes_a, self.trellis, termination='cont')
        self.x_bytes_b_cod = cc.conv_encode(self.info_bytes_b, self.trellis, termination='cont')

        # модуляция
        modem = QAMModem(self.M)

        self.x_qam_first_unc = np.array(modem.modulate(self.x_bytes_a_unc))
        self.x_qam_second_unc = np.array(modem.modulate(self.x_bytes_b_unc))
        self.x_qam_unc = np.concatenate((self.x_qam_first_unc, self.x_qam_second_unc))

        self.x_qam_first_cod = np.array(modem.modulate(self.x_bytes_a_cod))
        self.x_qam_second_cod = np.array(modem.modulate(self.x_bytes_b_cod))
        self.x_qam_cod = np.concatenate((self.x_qam_first_cod, self.x_qam_second_cod))

    def ofdm(self):
        # --- Без кодирования ---
        self.x_ofdm_tensor_unc = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 2), dtype=complex)
        self.x_ofdm_tensor_unc[:, :, 0] = self.x_qam_first_unc.reshape(self.number_ofdm_symbols, self.number_subcarriers)
        self.x_ofdm_tensor_unc[:, :, 1] = self.x_qam_second_unc.reshape(self.number_ofdm_symbols, self.number_subcarriers)

        # --- С кодированием ---
        self.x_ofdm_tensor_cod = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 2), dtype=complex)
        self.x_ofdm_tensor_cod[:, :, 0] = self.x_qam_first_cod.reshape(self.number_ofdm_symbols, self.number_subcarriers)
        self.x_ofdm_tensor_cod[:, :, 1] = self.x_qam_second_cod.reshape(self.number_ofdm_symbols, self.number_subcarriers)

    def power_of_signal(self):
        return 2/3 * (self.M - 1)

    def _apply_channel_with_pilots(self, x_ofdm_tensor):

        #  пилотов
        res = np.zeros((3 * self.number_ofdm_symbols, self.number_subcarriers, 2), dtype=complex)
        res[0::3, :, :] = x_ofdm_tensor
        res[1::3, :, :] = 1.0  # Pilot 1
        res[2::3, :, 0] = 1.0  # Pilot 2
        res[2::3, :, 1] = 2.0  # Pilot 2

        # каждая третья срока инфа. теперь между информацией подается один офдм символ из 1 на две антенны, и один из 1 для певрой и 2 для второй


        x_time_pilots = np.fft.ifft(res, axis=1) * np.sqrt(self.number_subcarriers)

        def convolution_cp(r, h):
            cp_len = len(h) - 1
            x_cp = np.concatenate((r[-cp_len:], r))
            y_conv = np.convolve(x_cp, h, mode='valid')
            return y_conv

        y_time_pilots = np.zeros_like(x_time_pilots)

        #  (канал меняется каждые 3 символа)
        for i in range(self.number_ofdm_symbols):
            h11 = self.impulse_response()
            h12 = self.impulse_response()
            h21 = self.impulse_response()
            h22 = self.impulse_response()

            for n in range(3):
                idx = 3 * i + n
                tx1 = x_time_pilots[idx, :, 0]
                tx2 = x_time_pilots[idx, :, 1]

                rx1 = convolution_cp(tx1, h11) + convolution_cp(tx2, h12)
                rx2 = convolution_cp(tx1, h21) + convolution_cp(tx2, h22)

                y_time_pilots[idx, :, 0] = rx1
                y_time_pilots[idx, :, 1] = rx2

        # шум
        P_signal = self.power_of_signal()
        sigma = np.sqrt(P_signal / (2 * self.SNR))
        noise = np.random.normal(0, sigma, y_time_pilots.shape) + 1j * np.random.normal(0, sigma, y_time_pilots.shape)

        #  шум на пилотах = 0
        for idx in range(y_time_pilots.shape[0]):
            if idx % 3 != 0:
                noise[idx, :, :] = 0

        y_time_noisy = y_time_pilots + noise


        Y_OFDM_pilots = np.fft.fft(y_time_noisy, axis=1) / np.sqrt(self.number_subcarriers)


        H_est = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 2, 2), dtype=complex)
        Y_OFDM_data = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 2), dtype=complex)

        # получаем H

        for i in range(self.number_ofdm_symbols):
            Y_OFDM_data[i] = Y_OFDM_pilots[3 * i]
            P1 = Y_OFDM_pilots[3 * i + 1]
            P2 = Y_OFDM_pilots[3 * i + 2]

            H_est[i, :, 0, 0] = 2 * P1[:, 0] - P2[:, 0]
            H_est[i, :, 0, 1] = P2[:, 0] - P1[:, 0]
            H_est[i, :, 1, 0] = 2 * P1[:, 1] - P2[:, 1]
            H_est[i, :, 1, 1] = P2[:, 1] - P1[:, 1]

        return H_est, Y_OFDM_data

    def channel_with_noise(self):
        self.SNR = 10 ** (self.SNR_db / 10)

        self.H_est_unc, self.y_unc = self._apply_channel_with_pilots(self.x_ofdm_tensor_unc)
        self.H_est_cod, self.y_cod = self._apply_channel_with_pilots(self.x_ofdm_tensor_cod)

    def _zf(self, y_tensor, H_est):
        x_est = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 2), dtype=complex)
        for i in range(self.number_ofdm_symbols):
            for k in range(self.number_subcarriers):
                Hk = H_est[i, k, :, :]
                yk = y_tensor[i, k, :]
                x_est[i, k, :] = np.linalg.solve(Hk, yk)
        return x_est.transpose(2, 0, 1).reshape(-1)

    def zero_forcing(self):
        self.y_zf_unc = self._zf(self.y_unc, self.H_est_unc)
        self.y_zf_cod = self._zf(self.y_cod, self.H_est_cod)

    def _mmse(self, y_tensor, H_est):
        x_est = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 2), dtype=complex)
        for i in range(self.number_ofdm_symbols):
            for k in range(self.number_subcarriers):
                Hk = H_est[i, k, :, :]
                yk = y_tensor[i, k, :]
                Hh = Hk.conj().T
                W = np.linalg.inv(Hh @ Hk + (1 / self.SNR) * np.eye(2)) @ Hh
                x_est[i, k, :] = W @ yk
        return x_est.transpose(2, 0, 1).reshape(-1)

    def mmse(self):
        self.y_mmse_unc = self._mmse(self.y_unc, self.H_est_unc)
        self.y_mmse_cod = self._mmse(self.y_cod, self.H_est_cod)

    def _ml(self, y_tensor, H_est):
        x_est = np.zeros((self.number_ofdm_symbols, self.number_subcarriers, 2), dtype=complex)
        for i in range(self.number_ofdm_symbols):
            for k in range(self.number_subcarriers):
                Hk = H_est[i, k, :, :]
                yk = y_tensor[i, k, :]
                metrics = np.linalg.norm(yk - (Hk @ self.ml_candidates.T).T, axis=1) ** 2
                best = np.argmin(metrics)
                x_est[i, k, :] = self.ml_candidates[best]
        return x_est.transpose(2, 0, 1).reshape(-1)

    def ml(self):
        self.y_ml_unc = self._ml(self.y_unc, self.H_est_unc)
        self.y_ml_cod = self._ml(self.y_cod, self.H_est_cod)

    def decode_uncoded(self, output):
        modem = QAMModem(self.M)
        return np.array(modem.demodulate(output, demod_type='hard'))

    def decode_coded(self, output):
        modem = QAMModem(self.M)

        
        demod_bits = np.array(modem.demodulate(output, demod_type='hard'))

        half = len(demod_bits) // 2
        bits_a = demod_bits[:half]
        bits_b = demod_bits[half:]

        tb_depth = 5 * (self.trellis.total_memory + 1)

        dec_info_a = cc.viterbi_decode(bits_a, self.trellis, tb_depth=tb_depth, decoding_type='hard')
        dec_info_b = cc.viterbi_decode(bits_b, self.trellis, tb_depth=tb_depth, decoding_type='hard')

        return np.concatenate((dec_info_a, dec_info_b))

    def ber_until_100(self, snr, max_iter=100):
        self.SNR_db = snr
        TARGET_ERRORS = 100

        err = {k: 0 for k in ['zf_unc', 'mmse_unc', 'ml_unc', 'zf_cod', 'mmse_cod', 'ml_cod']}
        bits = {k: 0 for k in err.keys()}
        evm_zf_acc, evm_mmse_acc = [], []

        for _ in range(max_iter):
            if all(v >= TARGET_ERRORS for v in err.values()):
                break

            self.modulating()
            self.ofdm()
            self.channel_with_noise()
            self.zero_forcing()
            self.mmse()
            self.ml()

            dec_zf_unc = self.decode_uncoded(self.y_zf_unc)
            dec_mmse_unc = self.decode_uncoded(self.y_mmse_unc)
            dec_ml_unc = self.decode_uncoded(self.y_ml_unc)

            dec_zf_cod = self.decode_coded(self.y_zf_cod)
            dec_mmse_cod = self.decode_coded(self.y_mmse_cod)
            dec_ml_cod = self.decode_coded(self.y_ml_cod)

            n_unc = len(self.x_bytes_unc)
            n_cod = len(self.info_bytes)

            err['zf_unc'] += int(np.sum(dec_zf_unc != self.x_bytes_unc))
            err['mmse_unc'] += int(np.sum(dec_mmse_unc != self.x_bytes_unc))
            err['ml_unc'] += int(np.sum(dec_ml_unc != self.x_bytes_unc))
            err['zf_cod'] += int(np.sum(dec_zf_cod != self.info_bytes))
            err['mmse_cod'] += int(np.sum(dec_mmse_cod != self.info_bytes))
            err['ml_cod'] += int(np.sum(dec_ml_cod != self.info_bytes))

            for k in ['zf_unc', 'mmse_unc', 'ml_unc']: bits[k] += n_unc
            for k in ['zf_cod', 'mmse_cod', 'ml_cod']: bits[k] += n_cod


            evm_zf_acc.append(np.sum(np.abs(self.x_qam_unc - self.y_zf_unc) ** 2) / len(self.x_qam_unc))
            evm_mmse_acc.append(np.sum(np.abs(self.x_qam_unc - self.y_mmse_unc) ** 2) / len(self.x_qam_unc))

        def safe_ber(e, b):
            return e / b if b > 0 else 0.0

        return (safe_ber(err['zf_unc'], bits['zf_unc']),
                safe_ber(err['mmse_unc'], bits['mmse_unc']),
                safe_ber(err['zf_cod'], bits['zf_cod']),
                safe_ber(err['mmse_cod'], bits['mmse_cod']),
                safe_ber(err['ml_unc'], bits['ml_unc']),
                safe_ber(err['ml_cod'], bits['ml_cod']),
                float(np.mean(evm_zf_acc)) if evm_zf_acc else 0.0,
                float(np.mean(evm_mmse_acc)) if evm_mmse_acc else 0.0)

    def plot_constellations(self):
        # Отрисовка QAM созвездий (аналог ваших subplot'ов)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.title(f"Modulated signal (QAM{self.M})")
        plt.scatter(np.real(self.x_qam_unc), np.imag(self.x_qam_unc), color="red")
        plt.xlabel("I"); plt.ylabel("Q")

        plt.subplot(1, 3, 2)
        plt.title(f"Signal after channel with AWGN (SNR={self.SNR_db} dB)")
        y_flat = self.y_unc.flatten()
        plt.scatter(np.real(y_flat), np.imag(y_flat), color="black", s=0.5)
        plt.scatter(np.real(self.x_qam_unc), np.imag(self.x_qam_unc), color="red", s=20)
        plt.xlabel("I"); plt.ylabel("Q")

        plt.subplot(1, 3, 3)
        plt.title("Equalized signal")
        plt.scatter(np.real(self.y_zf_unc), np.imag(self.y_zf_unc), s=0.5, color="blue", label="ZF")
        plt.scatter(np.real(self.y_mmse_unc), np.imag(self.y_mmse_unc), s=0.5, color="green", label="MMSE")
        plt.scatter(np.real(self.x_qam_unc), np.imag(self.x_qam_unc), color="red", s=20)
        plt.xlabel("I"); plt.ylabel("Q")
        plt.legend(fontsize=10)

        plt.tight_layout()
        plt.show()

    def ber_100_avg_plot(self, snr_range, n_iter):
        ber_zf_unc_avg, ber_mmse_unc_avg = [], []
        ber_zf_cod_avg, ber_mmse_cod_avg = [], []
        evm_zf_unc_avg, evm_mmse_unc_avg = [], []
        ber_ml_unc_avg, ber_ml_cod_avg = [], []

        for snr in snr_range:
            b_zu, b_mu, b_zc, b_mc, b_mlu, b_mlc = [], [], [], [], [], []
            e_zu, e_mu = [], []

            for _ in range(n_iter):
                res = self.ber_until_100(snr)
                b_zu.append(res[0]); b_mu.append(res[1])
                b_zc.append(res[2]); b_mc.append(res[3])
                b_mlu.append(res[4]); b_mlc.append(res[5])
                e_zu.append(res[6]); e_mu.append(res[7])

            ber_zf_unc_avg.append(np.mean(b_zu)); ber_mmse_unc_avg.append(np.mean(b_mu))
            ber_zf_cod_avg.append(np.mean(b_zc)); ber_mmse_cod_avg.append(np.mean(b_mc))
            ber_ml_unc_avg.append(np.mean(b_mlu)); ber_ml_cod_avg.append(np.mean(b_mlc))
            evm_zf_unc_avg.append(np.mean(e_zu)); evm_mmse_unc_avg.append(np.mean(e_mu))

            print(f"SNR {snr:2d} | ZF Uncoded: {np.mean(b_zu):.3e} | ZF Coded: {np.mean(b_zc):.3e} || MMSE Uncoded: {np.mean(b_mu):.3e} | MMSE Coded: {np.mean(b_mc):.3e}")

        plt.figure(figsize=(12, 5))

        #  BER
        plt.subplot(1, 2, 1)
        plt.semilogy(snr_range, ber_zf_unc_avg, '-', label="ZF Uncoded", color='blue')
        plt.semilogy(snr_range, ber_zf_cod_avg, '--', label="ZF Coded", color='blue')
        plt.semilogy(snr_range, ber_mmse_unc_avg, '-', label="MMSE Uncoded", color='orange')
        plt.semilogy(snr_range, ber_mmse_cod_avg, '--', label="MMSE Coded", color='orange')
        plt.semilogy(snr_range, ber_ml_unc_avg, '-', label="ML Uncoded", color='green')
        plt.semilogy(snr_range, ber_ml_cod_avg, '--', label="ML Coded", color='green')
        plt.xlabel("SNR (dB)")
        plt.ylabel("BER")
        plt.grid(True, which="both", linestyle='--', alpha=0.6)
        plt.legend(fontsize=8)

        #  EVM
        plt.subplot(1, 2, 2)
        plt.semilogy(snr_range, evm_zf_unc_avg, '-', label="ZF", color='blue')
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
    qam_1.zero_forcing()
    qam_1.mmse()



    qam_1.ber_100_avg_plot(snr_range=np.arange(0, 20, 2), n_iter=5)
