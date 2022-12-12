import torch
import numpy as np
np.set_printoptions(precision=6, suppress=True)

# encoder_x = np.load('../datasets/saved_encoder_x.npy', allow_pickle=True)
# decoder_x = np.load('../datasets/saved_decoder_x.npy', allow_pickle=True)

# # saved_x = np.empty(shape=(len(encoder_x), 51), dtype=np.float32)

# encoder_x = np.reshape(encoder_x, (len(encoder_x), 48))

# saved_x = np.concatenate((encoder_x, decoder_x), axis=1)

# np.save('../datasets/saved_x.npy', saved_x)

data = np.load("../datasets/saved_y.npy", allow_pickle=True)

for i in range(len(data)):
    print(data[i])
