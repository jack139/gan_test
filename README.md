## 收集的一些GAN网络

### 部分测试结果对比

> 说明：
>
> 1. 生成器基本结构都是卷积网络
> 2. 两张图，左为循环3000次，右为循环5000次
> 3. 缩写含义：SN - 谱归一化（Spectral Normalization）；EMA - 权重滑动平均（Exponential Moving Average）
> 4. 训练数据集使用CASIA-maxpy-clean，去除单色图片

#### DCGAN + SN

<center class="half">
<img src="keras/results/dcgan_sn_3000.png" alt="dcgan_sn_3000" width="350" /><img src="keras/results/dcgan_sn_5000.png" alt="dcgan_sn_5000" width="350" />
</center>

#### DCGAN + SN + EMA

<center class="half">
<img src="keras/results/dcgan_sn_ema_3000.png" alt="dcgan_sn_ema_3000" width="350" /> <img src="keras/results/dcgan_sn_ema_5000.png" alt="dcgan_sn_ema_5000" width="350" />
</center>

#### RSGAN + SN

<center class="half">
<img src="keras/results/rsgan_sn_3000.png" alt="rsgan_sn_3000" width="350" /> <img src="keras/results/rsgan_sn_5000.png" alt="rsgan_sn_5000" width="350" />
</center>

#### WGAN + SN

<center class="half">
<img src="keras/results/wgan_sn_3000.png" alt="wgan_sn_3000" width="350" /> <img src="keras/results/wgan_sn_5000.png" alt="wgan_sn_5000" width="350" />
</center>


#### WGAN-GP

<center class="half">
<img src="keras/results/wgan_gp_3000.png" alt="wgan_gp_3000" width="350" /> <img src="keras/results/wgan_gp_5000.png" alt="wgan_gp_5000" width="350" />
</center>

#### WGAN-DIV

<center class="half">
<img src="keras/results/wgan_div_3000.png" alt="wgan_div_3000" width="350" /> <img src="keras/results/wgan_div_5000.png" alt="wgan_div_5000" width="350" />
</center>

#### GAN-QP + L1

<center class="half">
<img src="keras/results/gan_qp_l1_3000.png" alt="gan_qp_l1_3000" width="350" /> <img src="keras/results/gan_qp_l1_5000.png" alt="gan_qp_l1_5000" width="350" />
</center>

#### GAN-QP + L1 + EMA

<center class="half">
<img src="keras/results/gan_qp_l1_ema_3000.png" alt="gan_qp_l1_ema_3000" width="350" /> <img src="keras/results/gan_qp_l1_ema_5000.png" alt="gan_qp_l1_ema_5000" width="350" />
</center>

#### GAN-QP + L2

<center class="half">
<img src="keras/results/gan_qp_l2_3000.png" alt="gan_qp_l2_3000" width="350" /> <img src="keras/results/gan_qp_l2_5000.png" alt="gan_qp_l2_5000" width="350" />
</center>


### 在Keras 2.3.1使用权重滑动平均（EMA）的patch

```diff
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

EMA的实现见```keras/utils.py```中```ExponentialMovingAverage```。使用EMA的例子见```keras/dcgan_sn_ema.py```。



### 原始代码来源

[1]: https://github.com/bojone/gan	"https://github.com/bojone/gan"
[2]: https://github.com/eriklindernoren/Keras-GAN	"https://github.com/eriklindernoren/Keras-GAN"
