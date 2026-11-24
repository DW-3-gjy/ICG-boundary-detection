# 2026-10-22 跨患者静态分割实验

本目录归档今天实际完成的8组、每组3个种子共24次训练，以及10月19日真实历史模型的追加对比。供代码审查、复算和算法交接使用。主结果采用三次训练均值与样本标准差，不以较高的单次分数替代总体结果。

这是一项两患者回顾性探索：患者一开发，患者二测试。仅有1位测试患者，没有真实连续视频的独立时序标注，不能推断临床有效性或荧光消退后的动态保持能力。

## 先看这些文件

|问题|对应文件|
|---|---|
|今天究竟运行了哪个训练文件？|`source_snapshot/run_study.py`，这是实际执行文件的逐字节归档；它调用同目录冻结的`model/src/icg_pipeline/training`。并非直接用默认`train.py`得出本表。|
|如何准备相同划分、启动训练和复算？|仓库`model/scripts/cross_patient_study.py`的`prepare-replay`、`run-replay`、`summarize`，见下方命令。|
|用了哪些样本、如何划分？|`split_manifest.csv`与`DATASET.md`。只有匿名ID、阶段、患者代号、来源组和文件哈希。|
|训练参数、损失和随机种子？|`protocol.public.json`内24条完整配置；各次`runs/<run_id>/config.json`相互核对。|
|每轮训练/验证、入选轮次？|`runs/<run_id>/checkpoints/history.json`、`run_summary.json`、`checkpoint_lock.json`。|
|每张图的原始评估结果？|`runs/<run_id>/test/per_image.csv`与`evaluation.json`；标注与掩膜不入库。|
|8组均值与标准差？|`RESULTS.md`、`primary_metrics.csv`、`summary.json`。|
|前几天那一版的对比？|`historical_comparison/`，真正的10月19日检查点，不是本轮UD套旧代码。|
|结果是否可追溯？|`manifest_sha256.json`、`test_prediction_hashes.json`、`verification/`；可执行`verify-public`。|

`protocol.public.json`是去除本机绝对路径后的公开衍生件，**不是原始冻结协议文件**。原始协议SHA256为`0c00e48cfd63f5365be099b8dc27fd61d288f9841c431ffbeea1789123e3371c`，公开件及重现件各有自己的哈希，不能混用。原数据、权重、协议、日志均保留在本地且未改写。校验清单覆盖本目录的数值、代码和说明文件，清单自身不参与自哈希。

## PR76和今天训练的关系

[PR76](https://github.com/DW-3-fjm/ICG-boundary-detection/pull/76)已合并至`develop`，合并提交`8ca707ebdfb368980c7b2bdd7de71bd93588b23c`。其修复为：丢点时禁止错误闭合轮廓、统一序列HD95字段和无定义输出、视频导出不再修改跟踪状态。单帧分割网络及训练损失并未由PR76改变。针对修复点的12项合成软件回归检查为修复前1项通过、修复后12项通过，不能当作患者动态实验。

本轮实际训练基于`c4e22869d06e23e5aead82435cef86950b578967`，即PR76之后的Windows中文路径推理修复。本PR一并提交该修复及其测试。算法对比用的是此前已有的普通U-Net与ResNet18-U-Net、损失组合、ImageNet初始化和BN策略，结果不能全部归因于PR76。

## 划分与选模

- 患者一Z25：训练20、验证5。训练和验证来源文件夹分开，但同一患者内仍存在相关性，缺少时间戳，不能保证真实时间间隔。
- 患者二：69张有标注帧全部不参与梯度训练或检查点选择。**主测试25张=原Z12+确认的Z且L13**；Q17、H10、阶段未确认L17共44张仅为原研究预设辅助分析，绝不计入本表的主Dice。
- Q/Z/H为显像前/中/后；L表示器械遮挡时的目标区域，不是器械本身。目标始终为既有多边形内部，不按荧光颜色反转。
- 患者一在10月19日的5张内部测试图，本轮转入训练，`original_experiment_split`明确记录；它们不能继续作为本轮独立测试。
- 每次从零开始优化器；普通U-Net和RS随机初始化，其余ResNet配置加载官方ImageNet编码器，不加载历史ICG权重。24次均训练120轮，每次480个优化器更新，未按测试结果早停。
- 按患者一512×288验证前景Dice选检查点，相同分数保留较早轮。原4K验证指标用于描述，与选模时的分辨率不同。
- 训练完成后锁定检查点和数据哈希，排他创建测试开始标记，再执行最终测试。模型训练阶段的测试目录内容访问由审计钩子阻止。

## 在没有患者数据的电脑复核数值

从仓库根目录运行，标准库Python即可，不需要Torch、GPU、患者图片或权重：

```powershell
python model/scripts/cross_patient_study.py verify-public
```

它会核对公开文件哈希、94条划分、24个检查点的记录身份、120轮历史、验证选模规则，以及所有预设子组的逐次均值、三次均值与样本SD。它不声称在缺少权重和原始掩膜时重新验证模型推理；掩膜复算证据单独保留在`verification/`。

## 获授权数据后的新目录重现

公开仓库不提供患者数据下载。临床数据负责人需通过获批准渠道另行提供原私有配对目录及匿名映射。不要从GitHub获取或向GitHub上传患者图像、视频、LabelMe坐标、掩膜、病例对比图或权重。

准备Python 3.12及与`environment.json`相容的PyTorch/Torchvision环境，安装`model/requirements-resnet.txt`。本轮实录为Torch 2.7.1+cu128、Torchvision 0.22.1+cu128、CUDA12.8、RTX5070Ti Laptop GPU。运行重现前，在Torch缓存中准备协议记录的官方ResNet18 ImageNet权重，CLI会核对其完整SHA256。未承诺跨驱动/GPU逐比特相同：TF32关闭、确定性为warn-only，原运行记录过CUDA算子无确定性实现的警告。

以下路径仅为占位示例，换为获授权的本地目录：

```powershell
# 1. 只准备新协议和冻结代码；检查开发集哈希，不打开测试图像内容。
python model/scripts/cross_patient_study.py prepare-replay --data-root "D:/approved-icg/cross_patient_20261022_01" --output-root "model/logs/cross_patient_replay_NEW"

# 2. 按归档顺序完成24次新训练，然后锁定权重并测试一次。不要对今天的原研究目录运行。
python model/scripts/cross_patient_study.py run-replay --study-root "model/logs/cross_patient_replay_NEW"

# 3. 测试完整结束后，复算纯数值，输出到一个尚不存在的新文件。
python model/scripts/cross_patient_study.py summarize --study-root "model/logs/cross_patient_replay_NEW" --output "model/logs/cross_patient_replay_NEW/numeric_summary.json"
```

目录重现不覆盖原始研究；协议会记录原私有协议、公开衍生件和新协议的哈希。重复运行已开始最终测试的目录会拒绝，不能删除锁后反复测试。重现仍使用已经查看过的患者二，不能算新增独立验证。

`source_snapshot/analyze_study.py`是原分析文件的逐字节归档，包含原本机图表入口，供追溯而非推荐直接运行。跨路径的数值复算请用上述`summarize`命令，避免误用其归档默认目录。

## 结果解释必须保留的限制

三种子SD是训练随机性，既不是25张图的标准差，也不是患者置信区间。历史模型仅一次训练，不伪造历史误差线或三次配对；逐图比较固定种子42，该次RB为三次中的最高值。

历史与本轮的训练样本量、归一化、损失和轮次不同，只能描述整体流程差异。患者二曾在接收时查看及确认阶段，之后又用于多方案比较；历史对比属于事后追加分析，不是新盲测。后续根据结果改算法，应另收新患者确认。

工具遮挡分层同时受到视角、亮度、形变等影响，原Z也未被证明完全无器械，不能据此声称已消除器械干扰。评估边界为掩膜外轮廓，包括器械接触和画面裁切闭合段，不等于单独的解剖肺段界面；距离单位为像素，没有毫米标定。
