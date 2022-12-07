import numpy as np
import pandas as pd

window_size = 3 # 몇 년을 window 사이즈로 잡을 것인지

encoder_x = np.empty((0, 16), float)
decoder_x = np.empty((0, 3), float)
y = np.empty((0, 3), float)

for i in range(1, 399):
    url = "./datasets/players/" + str(i) + ".csv"

    t = pd.read_csv(url).astype(float)
    t = t.fillna(0)
    t_numpy = t.to_numpy()[:, 1:]
    t_numpy = t_numpy.T

    # 만들 수 있는 데이터가 없는 경우
    if(t_numpy.shape[0] < window_size) :
        continue

    for i in range(window_size - 1, t_numpy.shape[0] - 1):
        tmp = t_numpy[i - (window_size - 1) : i + 1]
        tmp = np.delete(tmp, 1, axis = 1)
        encoder_x = np.append(encoder_x, tmp, axis = 0)
        decoder_x = np.append(y, t_numpy[i - (window_size - 1) : i + 1, 1])
        y = np.append(y, t_numpy[i - (window_size - 1) + 1 : i + 2, 1])
        

encoder_x = np.reshape(encoder_x, (-1, 3, 16))
decoder_x = np.reshape(decoder_x, (-1, 3))
y = np.reshape(y, (-1, 3))

print(encoder_x.shape)
print(decoder_x.shape)
print(y.shape)

np.save("./datasets/saved_encoder_x", encoder_x)
np.save("./datasets/saved_decoder_x", decoder_x)
np.save("./datasets/saved_y", y)