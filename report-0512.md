## Overview

Code Base 中提供了两种 Generator 网络架构，我们下面分别用 `carla2cs` 和 `carla2cs_ie2` 来称呼。

```yaml
# carla2cs

generator:
  type: hr
  config:
    encoder_type: ENCODER
    stem_norm: group
    num_stages: 4
    other_norm: group
    gbuffer_norm: RAD
    gbuffer_encoder_norm: residual
    num_gbuffer_layers: 1
  optimizer:
    type: adam
    learning_rate: 0.0001
    adam_beta: 0.9
    adam_beta2: 0.999
    clip_gradient_norm: 1000
  scheduler:
    type: 'step'
    step: 100000
    gamma: 0.5
```

```yaml
# carla2cs_ie2

generator:
  type: hr_new
  config:
    encoder_type: ENCODER
    stem_norm: group
    num_stages: 6
    other_norm: group
    gbuffer_norm: RAC
    gbuffer_encoder_norm: residual2
    num_gbuffer_layers: 1
  optimizer:
    type: adam
    learning_rate: 0.0001
    adam_beta: 0.9
    adam_beta2: 0.999
    clip_gradient_norm: 1000
  scheduler:
    type: 'step'
    step: 100000
    gamma: 0.5
```

两者使用完全相同的 Discriminator 和除了权重系数以外完全相同的 loss。

### `carla2cs`

现在 `carla2cs` 任务经过 5 次实验，试着切换了随机数发生的种子，甚至是重新 sample 生成了 crops 数据集，每次都会在 ~5000 次 Step 时 loss 准时飞走。在后端 `epe2` session 中我 track 到了 `loss=NaN` 第一次时的情况，但是我不知道怎么进一步检视问题。

我尝试读取 step=5000 次时的 checkpoint，然后用其来 infer 图片，得出结果如下：

![image-20220512004137121](https://s2.loli.net/2022/05/12/oZfxKliLs83QP4V.png)

<center>Carla 仿真生成数据</center>

![image-20220512004010172](https://s2.loli.net/2022/05/12/aU1VGowthgLZRqp.png)

<center> carla2cs ~5000 steps 生成的图像 </center>



从结果来看它确实是在 train 了，但是在 train 的过程中由于某些原因导致 loss 飞升。

### `carla2cs_ie2`

这个模型 loss 不会飞，到晚上大概已经 train 了 40000 多个 iteration。但是这东西似乎啥都学不会：

![image-20220512004723782](https://s2.loli.net/2022/05/12/4iuEOZTD6HrGwfQ.png)

<center>Carla 仿真图</center>

![image-20220512004558236](https://s2.loli.net/2022/05/12/D6ELQAnceR7yigm.png)

<center>carla2cs_ie2 ~40000 steps</center>





这两个模型喂进去的是同样的数据，但是 train 的效果完全不一样。下一步我该怎么检视这个 bug？

+ IP: 10.0.0.14
+ Account: gaoha
+ Password: `,.,.,.,.`
+ `tmux attach -t epe2`
+ CodeBase @ `/home/gaoha/epe/code`
+ readme-c7w.md 和 `Carla/gen.sh` 是我做的所有操作记录

