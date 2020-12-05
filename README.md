## 收集的一些GAN的例子

[1]: https://github.com/bojone/gan
[2]: https://github.com/eriklindernoren/Keras-GAN


### 在keras 2.3.1使用权重滑动平均（EMA）的patch

```
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
```

EMA的实现见```keras/utils.py```中```ExponentialMovingAverage```。EMA的使用见```keras/dcgan_sn_ema.py```。
