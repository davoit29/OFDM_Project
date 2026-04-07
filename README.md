# OFDM_Project

2 передающие антенны
2 принимающие
OFDM по поднесущим
QAM модуляция


## Структура OFDM - MIMO
<img width="655" height="299" alt="image" src="https://github.com/user-attachments/assets/bf0e591d-f77e-4e7c-a431-de57747beb1f" />

Для каждой поднесущей и ОФДМ символа
```python
x_ofdm_tensor[i, k] = [x1, x2]
```
## Структура пилотов

<img width="896" height="271" alt="image" src="https://github.com/user-attachments/assets/b8ead995-1e7b-415a-9337-d5661a47ca14" />

Для каждого модуляционного символа своя матрица H 2 на 2
```python
H_est[i, :, 0, 0] = 2 * P1[:, 0] - P2[:, 0]  # h11
H_est[i, :, 0, 1] = P2[:, 0] - P1[:, 0]      # h12

H_est[i, :, 1, 0] = 2 * P1[:, 1] - P2[:, 1]  # h21
H_est[i, :, 1, 1] = P2[:, 1] - P1[:, 1]      # h22
```
## Структура кодирования
## Структура LLR
## ZF
## MMSE
## ML


