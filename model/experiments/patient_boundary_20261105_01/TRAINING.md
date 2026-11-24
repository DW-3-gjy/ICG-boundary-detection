# 实际训练流程与复核入口

本次训练于2026-11-05启动、2026-11-07完成。以下记录已经执行的流程；取消开放折线的部分约束、难负样本和高分辨率局部细化只是后续建议，未用于这次结果。

## 运行了哪个文件

实际训练从私有研究目录的`code/scripts/patient_boundary_study.py`运行，其逐字节公开副本为[source_snapshot/scripts/patient_boundary_study.py](source_snapshot/scripts/patient_boundary_study.py)。配套14份冻结Python源码及哈希一并归档，主训练命令为：

```powershell
python model/logs/patient_boundary_20261105_01/code/scripts/patient_boundary_study.py run --study model/logs/patient_boundary_20261105_01 --microbatch 4
```

原执行链只在上述命令成功后运行同一冻结脚本的`test`命令。**原研究已经完成，不应再次执行这些历史命令、删除测试开始标记或重写缓存。**

通用`model/scripts/train.py`没有产生本研究主表。`analyze_patient_boundary_study.py`用于完成后的数值复算和固定种子病例重放，不负责训练选模。新边界头通过研究runner/分析脚本使用，尚未接入默认`predict.py`视频接口。

基准代码提交为`9cb6f4e7873a2d502b8f9801b4190b9b96f52a08`，新边界代码当时是其后的工作区改动；实际运行身份以冻结源码哈希确定，不能用该基准提交冒充完整新算法版本。本次归档PR基于develop独立提交，源代码快照保留当时真实运行内容。

## 数据冻结和开发

129帧先按患者划为开发106帧/9人、测试23帧/3人，再生成缓存并冻结协议、清单与代码。开发过程的访问审计拒绝打开`sealed_test`缓存。详细纳入规则、折内样本量见[DATASET.md](DATASET.md)。

五组A-E分别研究边界分支、荧光可见L样本与旧训练样本的贡献；每组3折、种子42/43/44，共45次开发训练。验证集合对所有组相同，不能把A/B解释成无器械场景，或把E解释成整个流程未使用旧信息。

## 单次训练

|项目|实际设置|
|---|---|
|模型|ResNet18-U-Net，B/D/E额外启用显式边界分支|
|初始化|官方ImageNet编码器；新解码器/头/优化器；不加载历史ICG检查点|
|编码器|卷积继续学习，BN运行统计冻结|
|图像|RGB768×432、ImageNet归一化|
|抽样|先等概率选训练患者，再等概率选其合格帧；有放回|
|增强|每张图独立水平翻转50%、旋转±10度、亮度/对比度扰动|
|有效batch与microbatch|4与4|
|优化|AdamW，编码器3e-5、其他3e-4，weight decay1e-5；余弦最低1e-6，梯度裁剪5|
|开发预算|每次2400次优化器更新，每40次验证，共60验证点|
|数值|AMP，TF32关闭，确定性warn-only；不保证跨GPU/驱动逐比特一致|

一次update是有效4张图反向传播后的一次参数更新。一次开发训练有9600次帧抽样，不是9600张独立样本，也不是2400个epoch。同一种子各组共享参数初始化匹配；边界头单独初始化，避免它改变其他层的初始参数。

完整区域使用0.5 Tversky＋0.4 CE＋0.1 BoundaryCE；所有组有0.1公共软区域轮廓监督。开放折线跳过完整区域损失，但当前仍有软区域轮廓监督。B/D/E额外加入0.1直接边界损失和0.05边界头/区域轮廓一致性项。因此，当前开放折线训练不能描述成只有独立边界头监督。

## 选择更新数和阈值

每个开发运行在固定更新点及阈值0.2/0.3/0.4/0.5/0.6/0.7/0.8中，按验证患者等权Boundary F1@20px选择最佳组合。同分保留较早更新；阈值同分优先靠近0.5，再选较低值。验证图不反向传播，但参与选择，因此不能作为最终独立测试。

每组9次开发入选的更新数、阈值分别取中位数，再从ImageNet重新训练3个最终模型。并非继续使用交叉验证最佳权重，也不在最终重训中再次选检查点。

|组|最终梯度训练样本|更新数|边界提取阈值|
|---|---|---:|---:|
|A|9位开发患者，81张Z|1880|0.2|
|B|9位开发患者，81张Z|1920|0.8|
|C|9位开发患者，106张Z＋荧光L|1880|0.2|
|D|9位开发患者，106张Z＋荧光L|2160|0.8|
|E|7位新开发患者，56张Z＋荧光L|2040|0.8|

共45次开发＋15次最终重训，总参数更新137640次。边界阈值0.2/0.8不能当作区域Dice的阈值；区域Dice始终按区域概率0.5计算。

## 测试与统计

15个模型于2026-11-07 13:09:21锁定，13:09:24开始统一测试。逐图指标均归档，空预测F1计零，距离未定义保留，不删失败图提高均值。固定种子42的病例展示与115次可视化重放不属于新选模。

主指标为20原图像素容差Boundary F1，10/40像素为预设敏感性分析。网络网格一个像素对应原图5像素，骨架点中心映射回4K后匹配原图参考，不是原生4K预测。先逐图、再患者内平均、再患者等权，最后对三种子计算均值与样本SD；SD不是患者置信区间。区域Dice仅对2位患者11张完整区域参考计算。

## 没有患者数据时复核

从仓库根目录运行，Python标准库即可：

```powershell
python model/scripts/patient_boundary_archive.py verify-public
```

这检查公开文件哈希、匿名划分、开发历史与选模、最终训练预算/阈值和逐图数值聚合。没有患者图像或权重时，它不能重算网络预测、确认实际张量内容或独立裁定临床标注；相关私有重放与哈希核验作为原研究审计记录保存。

## 获授权数据后的新目录重现

原`prepare`依赖本机接收记录。便携入口是`model/scripts/replay_patient_boundary_study.py`，输入公开归档及由负责人另行提供的**私有映射CSV**，字段为：

```text
sample_id,image_path,annotation_or_mask_path,annotation_format
```

`annotation_format`为`labelme_json`或`binary_mask`。样本ID须与公开清单一致，源图和标注/掩膜哈希必须匹配；路径只写在私有映射中，不提交到GitHub。保留完整区域与开放折线语义，不能把开放折线转换成填充区域。

安装`model/requirements-resnet.txt`及与归档`environment.json`相容的PyTorch/Torchvision。原运行使用Python3.12、Torch2.7.1+cu128和RTX5070Ti Laptop；官方ImageNet编码器权重也需与协议哈希匹配。

```powershell
python model/scripts/replay_patient_boundary_study.py prepare-replay --archive model/experiments/patient_boundary_20261105_01 --source-map LOCAL_PRIVATE_MAP.csv --output model/logs/patient_boundary_replay_NEW
python model/scripts/replay_patient_boundary_study.py run-replay --study model/logs/patient_boundary_replay_NEW
```

第二条命令会实际训练，只有准备通过后才应执行；本次公开归档整理没有运行它。重现必须使用新目录，不覆盖原研究。相同既有患者的重现不是新增独立验证，后续算法改进需要新患者确认。

## 与10月22日研究的区别

上轮为1位开发患者20训/5验证、另1位测试患者25帧；120个epoch、batch5，合计480次参数更新，每组3种子共24次训练。选模依据低分辨率验证前景Dice，主要参考为区域。

本轮将开发阶段也按患者三折隔离，新增开放折线适配及直接边界头；输入从512×288变为768×432，主选模指标改为患者宏平均边界F1。ImageNet预训练不是首次加入。两轮测试患者、参考形式及Dice分母不同，不能用旧Dice0.771到新0.841直接宣称算法提高7个百分点。PR76本身是跟踪/评估修复，不是本轮边界分支的来源。
