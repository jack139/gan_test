# coding=utf-8

import numpy as np
from keras import backend as K


# 谱归一化
class SpectralNormalization:
    """层的一个包装，用来加上SN。
    """

    def __init__(self, layer):
        self.layer = layer

    def spectral_norm(self, w, r=5):
        w_shape = K.int_shape(w)
        in_dim = np.prod(w_shape[:-1]).astype(int)
        out_dim = w_shape[-1]
        w = K.reshape(w, (in_dim, out_dim))
        u = K.ones((1, in_dim))
        for i in range(r):
            v = K.l2_normalize(K.dot(u, w))
            u = K.l2_normalize(K.dot(v, K.transpose(w)))
        return K.sum(K.dot(K.dot(u, w), K.transpose(v)))

    def spectral_normalization(self, w):
        return w / self.spectral_norm(w)

    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        if not hasattr(self.layer, 'spectral_normalization'):
            if hasattr(self.layer, 'kernel'):
                self.layer.kernel = self.spectral_normalization(self.layer.kernel)
            if hasattr(self.layer, 'gamma'):
                self.layer.gamma = self.spectral_normalization(self.layer.gamma)
            self.layer.spectral_normalization = True
        return self.layer(inputs)


class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。

    训练：
    EMAer = ExponentialMovingAverage(model) # 在模型compile之后执行
    EMAer.inject() # 在模型compile之后执行
    model.fit(x_train, y_train) # 训练模型

    预测：
    EMAer.apply_ema_weights() # 将EMA的权重应用到模型中
    model.predict(x_test) # 进行预测、验证、保存等操作

    继续训练：
    EMAer.reset_old_weights() # 继续训练之前，要恢复模型旧权重。还是那句话，EMA不影响模型的优化轨迹。
    model.fit(x_train, y_train) # 继续训练
    """
    def __init__(self, model, momentum=0.999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]
    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            #self.model.metrics_updates.append(op)
            self.model.add_metric(op, 'ema_metrics') # 自定义的 metrics
    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))
    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))
    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))