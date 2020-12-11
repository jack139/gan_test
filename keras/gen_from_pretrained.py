# coding=utf-8
# 从训练好的权重生成图片

import numpy as np
import gan_qp_l1_l2_ema as qp
from utils import sample

qp.g_train_model.load_weights('checkpoints/g_train_ema_model.weights')
qp.g_model = qp.g_train_model.layers[2]
qp.d_model = qp.g_train_model.layers[3]

n_size = 9

for i in range(5):
    Z = np.random.randn(n_size**2, qp.z_dim)
    sample('samples/gen_%d.png'%i, qp.g_model, qp.img_dim, qp.z_dim, n=n_size, z_samples=Z)
