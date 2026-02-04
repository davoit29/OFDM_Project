import numpy as np
import matplotlib.pyplot as plt





def BPSK(number_symbols,H,SNR_db):
    x_bytes = np.random.randint(0, 2, number_symbols) #создаем вектор битов
    x = []
    for i in range(number_symbols):# модуляция
        if x_bytes[i]  == 0:
            x.append(1+0j)
        elif x_bytes[i] == 1:
            x.append(-1+0j)
    x = np.array(x)
    x_re = []
    x_im = []
    for i in range(number_symbols):
        x_re.append(x[i].real)
        x_im.append(x[i].imag)




    print(x)

    power_x = np.mean(np.abs(x)**2)
    SNR = 10 ** (SNR_db / 10)
    power_noise = power_x / SNR
    print(power_x)
    print(SNR)
    print(power_noise)


    y = []





    for i in range(number_symbols):
        noise = (np.random.randn(1) + 1j * np.random.randn(1) )* np.sqrt(power_noise / 2)


        y.append(H*x[i]+noise)


    print(y)


    plt.figure()

    plt.scatter(x_re, x_im, s=120, color='red')


    plt.axvline(0, color='k', linewidth=1)
    plt.axhline(0, color='k', linewidth=1)

    plt.grid(True)


    plt.xlabel("Q")
    plt.ylabel("I")
    plt.title("X")

    plt.show()


    plt.figure()

    plt.scatter(np.real(y), np.imag(y), s=10)


    plt.axvline(0, color='k', linewidth=1)
    plt.axhline(0, color='k', linewidth=1)


    plt.scatter([1, -1], [0, 0], color='red', s=120, zorder=5)

    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("Q")
    plt.ylabel("I")
    plt.title("Y")

    plt.show()


print(BPSK(100,1,3))


