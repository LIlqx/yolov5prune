
剪枝算法依据：

Learning Efficient Convolutional Networks Through Network Slimming（https://arxiv.org/abs/1708.06519）

该方法基于BN层系数gamma剪枝。

在一个卷积-BN-激活模块中，BN层可以实现通道的缩放。如下：

<p align="center">
<img src="img/Screenshot from 2021-05-25 00-26-23.png">
</p>

BN层的具体操作有两部分：

<p align="center">
<img src="img/Screenshot from 2021-05-25 00-28-15.png">
</p>

在归一化后会进行线性变换，那么当系数gamma很小时候，对应的激活（Zout）会相应很小。这些响应很小的输出可以裁剪掉，这样就实现了bn层的通道剪枝。

通过在loss函数中添加gamma的L1正则约束，可以实现gamma的稀疏化。

<p align="center">
<img src="img/Screenshot from 2021-05-25 00-28-52.png">
</p>



上面损失函数L右边第一项是原始的损失函数，第二项是约束，其中g(s) = |s|，λ是正则系数，根据数据集调整

实际训练的时候，就是在优化L最小，依据梯度下降算法：

​														𝐿′=∑𝑙′+𝜆∑𝑔′(𝛾)=∑𝑙′+𝜆∑|𝛾|′=∑𝑙′+𝜆∑𝛾∗𝑠𝑖𝑔𝑛(𝛾)

所以只需要在BP传播时候，在BN层权重乘以权重的符号函数输出和系数即可，对应添加如下代码:

```python
            # Backward
            loss.backward()
            # scaler.scale(loss).backward()
            # # ============================= sparsity training ========================== #
            srtmp = opt.sr*(1 - 0.9*epoch/epochs)
            if opt.st:
                ignore_bn_list = []
                for k, m in model.named_modules():
                    if isinstance(m, Bottleneck):
                        if m.add:
                            ignore_bn_list.append(k.rsplit(".", 2)[0] + ".cv1.bn")
                            ignore_bn_list.append(k + '.cv1.bn')
                            ignore_bn_list.append(k + '.cv2.bn')
                    if isinstance(m, nn.BatchNorm2d) and (k not in ignore_bn_list):
                        m.weight.grad.data.add_(srtmp * torch.sign(m.weight.data))  # L1
                        m.bias.grad.data.add_(opt.sr*10 * torch.sign(m.bias.data))  # L1
            # # ============================= sparsity training ========================== #

            optimizer.step()
                # scaler.step(optimizer)  # optimizer.step
                # scaler.update()
            optimizer.zero_grad()
```

这里并未对所有BN层gamma进行约束，详情见yolov5s每个模块 https://blog.csdn.net/IEEE_FELLOW/article/details/117536808
分析，这里对C3结构中的Bottleneck结构中有shortcut的层不进行剪枝，主要是为了保持tensor维度可以加：

<p align="center">
<img src="img/Screenshot from 2021-05-27 22-20-33.png">
</p>

实际上，在yolov5中，只有backbone中的Bottleneck是有shortcut的，Head中全部没有shortcut.

如果不加L1正则约束，训练结束后的BN层gamma分布近似正太分布：

<p align="center">
<img src="img/Screenshot from 2021-05-23 20-19-08.png">
</p>

是无法进行剪枝的。

稀疏训练后的分布：

<p align="center">
<img src="img/Screenshot from 2021-05-23 20-19-30.png">
</p>

可以看到，随着训练epoch进行，越来越多的gamma逼近0.

训练完成后可以进行剪枝，一个基本的原则是阈值不能大于任何通道bn的最大gamma。然后根据设定的裁剪比例剪枝。

剪掉一个BN层，需要将对应上一层的卷积核裁剪掉，同时将下一层卷积核对应的通道减掉。

这里在某个数据集上实验。

首先使用train.py进行正常训练：

```
python train.py --weights yolov5s.pt --adam --epochs 100
```

然后稀疏训练：

```
python train_sparsity.py --st --sr 0.0001 --weights yolov5s.pt --adam --epochs 100
```

sr的选择需要根据数据集调整，可以通过观察tensorboard的map，gamma变化直方图等选择。
在run/train/exp*/目录下:
```
tensorboard --logdir .
```
然后点击出现的链接观察训练中的各项指标.

训练完成后进行剪枝：

```
python prune.py --weights runs/train/exp1/weights/last.pt --percent 0.5 --cfg models/yolov5s.yaml
```

裁剪比例percent根据效果调整，可以从小到大试。注意cfg的模型文件需要和weights对应上,否则会出现[运行prune 过程中出现键值不对应的问题](https://github.com/midasklr/yolov5prune/issues/65),裁剪完成会保存对应的模型pruned_model.pt。

微调：

```
python finetune_pruned.py --weights pruned_model.pt --adam --epochs 100
```

在VOC2007数据集上实验,训练集为VOC07 trainval, 测试集为VOC07 test.作为对比,这里列举了faster rcnn和SSD512在相同数据集上的实验结果, yolov5输入大小为512.为了节省时间,这里使用AdamW训练100 epoch.

| model             | optim&epoch | sparity | mAP@.5      | mode size | forward time |
| ----------------- | ----------- | ------- | ----------- | --------- | ------------ |
| faster rcnn       |             | -       | 69.9(paper) |           |              |
| SSD512            |             | -       | 71.6(paper) |           |              |
| yolov5s           | sgd 300     | 0       | 67.4        |           |              |
| yolov5s           | adamw 100   | 0       | 66.3        |           |              |
| yolov5s           | adamw 100   | 0.0001  | 69.2        |           |              |
| yolov5s           | sgd 300     | 0.001   | Inf. error  |           |              |
| yolov5s           | adamw 100   | 0.001   | 65.7        | 28.7      | 7.32 ms      |
| 55% prune yolov5s |             |         | 64.1        | 8.6       | 7.30 ms      |
| fine-tune above   |             |         | 67.3        |           | 7.21 ms      |
| yolov5l           | adamw 100   | 0       | 70.1        |           |              |
| yolov5l           | adamw 100   | 0.001   | 0.659       |           | 12.95 ms     |



在自己数据集上的实验结果:

| model                 | sparity | map   | mode size |
| --------------------- | ------- | ----- | --------- |
| yolov5s               | 0       | 0.322 | 28.7 M    |
| sparity train yolov5s | 0.001   | 0.325 | 28.7 M    |
| 65% pruned yolov5s    | 0.001   | 0.318 | 6.8 M     |
| fine-tune             | 0       | 0.325 | 6.8 M     |


