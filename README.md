# ICG 荧光胸腔镜边界识别

吲哚菁绿(Indocyanine Green, ICG)荧光胸腔镜图像的自动边界识别 / 分割,辅助肺段间平面界定。

## 项目目标

- **初期目标(11 月)**:完成图像分割 baseline 模型 + 项目申报书
- **技术路线**:胸腔镜 ICG 图像 → 边界标注 → 训练分割模型 → 输出边界线 → 临床评估

## 成员与分工

| 成员 | 角色 | 分工 |
|---|---|---|
| DW-3-fjm | 主负责人 | 统筹、导师对接、整合申报书 |
| JaimeFine | 算法 | 模型选型、训练、调参、评估 |
| cT1ngS0w | 工程 | UI、接口、申报书计算机部分 |
| zhouyaowei0909 | 医学主负责人 | 文献、公开数据、申报书医学部分 |
| Lizimeng-007 | 临床 | 标注质控、临床评估、数据脱敏 |

> 成员信息已填写完毕。

## 目录结构

```
ICG-boundary-detection/
├── README.md           # 本文件:项目说明 + 分工 + 规则
├── .gitignore          # 排除数据、模型权重、环境文件
├── model/              # CS-A:训练 / 评估脚本
├── api/                # CS-B:模型推理接口
├── ui/                 # CS-B:演示界面
├── tools/              # 标注工具配置、数据预处理脚本
└── docs/
    ├── proposal/       # 申报书
    └── literature/     # 文献笔记
```

## 当前算法基线

`model/` 包含普通 U-Net / ResNet18-U-Net 前景分割与 Farneback 光流研究原型，使用双类别 logits 的 softmax/argmax，并在视频中区分分割支持、光流传播与跟踪失败。安装、推理、评估与交接事项见 [`model/README.md`](model/README.md) 和 [`model/HANDOFF.md`](model/HANDOFF.md)。

2026-10-22 已完成两患者回顾性静态分割探索：患者一训练20张、验证5张；患者二主测试为 Z12张 + Z且L13张。8组各3个种子共24次训练的匿名划分、完整配置、每轮历史、逐图指标与复算入口见[本轮实验档案](model/experiments/cross_patient_20261022_01/README.md)及[主结果](model/experiments/cross_patient_20261022_01/RESULTS.md)。基础UD测试Dice为 **0.690 ± 0.015**，完整RB为 **0.771 ± 0.029**；SD是训练种子间标准差，独立测试患者只有1位，不代表患者置信区间。PR76是跟踪和评估修复，不能把这些训练差异全部归因于PR76。静态结果尚不能证明荧光消退后的动态跟踪或临床有效性。

## 分支规则

- `main`:**稳定交付分支**,只有全部通过测试的代码才合并进来
- `develop`:**日常集成分支**,所有开发先合到这里
- 流程:`feature/xxx` → PR 到 `develop` → 全部通过后 `develop` → `main`
- 医学生在网页右上角分支下拉框选 `develop` 编辑文档即可

> 详细提交规则(提交到哪里、如何提交、AI 率控制)见 [提交规则.md](提交规则.md)。

## 数据红线(必读)

**患者胸腔镜图像 / 视频、标注图,严禁上传 GitHub(无论公开还是私有)。**

数据只放本地或网盘,位置在此说明(只写路径,不传数据):

- 本轮私有配对目录:`model/data/cross_patient_20261022_01/`（本地，已被Git忽略）
- 匿名登记与本地接收格式:[数据集说明](model/experiments/cross_patient_20261022_01/DATASET.md)、[划分清单](model/experiments/cross_patient_20261022_01/split_manifest.csv)
- 原始图像、LabelMe坐标及身份映射由数据负责人通过获批准渠道另行提供；仓库不包含患者数据下载或权重。

## 协作约定

- **CS 同学**:用 git / 命令行,走分支 + PR
- **医学生**:用 GitHub 网页版(编辑文档、回复 issue、上传文件),不学 git
- **任务跟踪**:用 Issues,对照分工表拆成一条条 issue 派给对应人
