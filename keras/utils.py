# coding=utf-8

import numpy as np
from keras import backend as K
import imageio


# 采样函数
def sample(path, g_model, img_dim, z_dim, n=9, z_samples=None):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    if z_samples is None:
        z_samples = np.random.randn(n**2, z_dim)
    for i in range(n):
        for j in range(n):
            z_sample = z_samples[[i * n + j]]
            x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype(np.uint8)
    imageio.imwrite(path, figure)


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
    EMAer.initialize() # 在模型compile之后执行
    EMAer.ema_on_batch() # 每个batch完成后执行

    预测：
    EMAer.apply_ema_weights() # 将EMA的权重应用到模型中
    model.predict(x_test) # 进行预测、验证、保存等操作

    继续训练：
    EMAer.reset_old_weights() # 继续训练之前，要恢复模型旧权重。还是那句话，EMA不影响模型的优化轨迹。
    """
    def __init__(self, model, momentum=0.999):
        self.momentum = momentum
        self.model = model
    def inject(self):
        """与旧代码兼容
        """
        self.initialize()
    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        self.mv_trainable_weights_vals = {x.name: K.get_value(x) for x in
                                          self.model.trainable_weights}
    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        for weight in self.model.trainable_weights:
             K.set_value(weight, self.mv_trainable_weights_vals[weight.name])
    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))
    def ema_on_batch(self):
        for weight in self.model.trainable_weights:
            old_val = self.mv_trainable_weights_vals[weight.name]
            self.mv_trainable_weights_vals[weight.name] -= \
                (1.0 - self.momentum) * (old_val - K.get_value(weight))


class ExponentialMovingAverage_NEED_Keras_patch:
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
    '''
    权重滑动平均 EMA （需要给Keras 2.3.1打个补丁，才能生效） 
    diff --git a/keras/engine/training.py b/keras/engine/training.py
    index 0a556f21..1a9a374e 100644
    --- a/keras/engine/training.py
    +++ b/keras/engine/training.py
    @@ -328,7 +328,7 @@ class Model(Network):
                     self.train_function = K.function(
                         inputs,
                         [self.total_loss] + metrics_tensors,
    -                    updates=updates + metrics_updates,
    +                    updates=updates + metrics_updates + (self._other_metrics if hasattr(self, '_other_metrics') else []),
                         name='train_function',
                         **self._function_kwargs)
    '''
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
            #self.model.metrics_updates.append(op) # 在 keras 2.2.4 有效
            if not hasattr(self.model, '_other_metrics'):
                self.model._other_metrics = []
            self.model._other_metrics.append(op)
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
