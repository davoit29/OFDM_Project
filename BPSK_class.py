import numpy as np
import matplotlib.pyplot as plt


class BPSK: # self переменная ссылка на сам класс, глобализирует переменные 
    def __init__(self, number_symbols=100, H=1, SNR_db=3):
        self.number_symbols = number_symbols
        self.H = H
        self.SNR_db = SNR_db
        self.x=None
        self.y=None
        self.power_noise=None
        
        
    def modulating(self): #  модуляция 0 эти -1 , 1 это 1
        x_bytes = np.random.randint(0, 2, self.number_symbols)
        x = []
        for i in range(self.number_symbols):
            if x_bytes[i] == 0:
                x.append(1+0j)
            elif x_bytes[i] == 1:
                x.append(-1+0j)
        self.x = np.array(x)
        

    def power(self): # нахождение мощности шума через осш в дб
        self.modulating()
        
        power_x = np.mean(np.abs(self.x)**2)
        SNR = 10 ** (self.SNR_db / 10)
        self.power_noise = power_x / SNR
        
        print(f"Мощность сигнала: {power_x}")
        print(f"SNR (линейное): {SNR}")
        print(f"Мощность шума: {self.power_noise}")

        
    def channel_with_noise(self):# через плотность моделируем случайный шум, добавляем к H*x
        self.power()
        y = []
        for i in range(self.number_symbols):
            noise = (np.random.randn(1) + 1j * np.random.randn(1)) * np.sqrt(self.power_noise / 2)
            y.append(self.H * self.x[i] + noise)
        
        self.y = np.array(y)
    def zero_forcing():
        None    


    def mmse():
        None     


    def plot(self):
        self.channel_with_noise()
        
        x_re = self.x.real
        x_im = self.x.imag

        y_re = self.y.real
        y_im = self.y.imag

        
        f, ax = plt.subplots(1,2)
        ax[0].scatter(x_re, x_im,color= 'red')
        ax[0].grid()
        ax[0].set_xlabel('Q')
        ax[0].set_ylabel('I')
        ax[0].set_title("Исходный сигнал")


        ax[1].scatter(y_re, y_im)
        ax[1].scatter(x_re, x_im,color= 'red' )
        ax[1].set_xlabel('Q')
        ax[1].set_ylabel('I')
        ax[1].set_title("Принятый сигнал")
        ax[1].grid()
       




        plt.show()
      


bpsk_1 = BPSK(number_symbols=100, H=1, SNR_db=3)

bpsk_1.plot()
