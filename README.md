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

`model/` 包含 U-Net / ResNet18-U-Net 区域分割、可见边界实验和 Farneback 光流研究原型。安装、通用推理、评估与交接事项见 [`model/README.md`](model/README.md) 和 [`model/HANDOFF.md`](model/HANDOFF.md)。荧光区域与所需显示界面的关系由标注定义，当前结果不构成临床有效性验证。

2026-11-07 已完成新旧数据联合静态边界实验：12位患者129帧，9位开发患者106帧，另外3位患者23帧独立测试。五组方案完成45次患者级交叉验证训练及15次最终重训。相同Z及荧光可见L数据下，边界模块使20像素容差的患者宏平均F1从 **0.354 ± 0.022** 升至 **0.392 ± 0.026**；10像素指标下降，完整区域Dice基本持平。SD仅反映三个训练种子的波动。

运行文件、匿名数据划分、逐次配置/历史、15份逐图测试表和标准库数值复核入口见[本轮实验档案](model/experiments/patient_boundary_20261105_01/README.md)、[训练流程](model/experiments/patient_boundary_20261105_01/TRAINING.md)及[完整数值结果](model/experiments/patient_boundary_20261105_01/RESULTS.md)。患者像素、坐标、mask和权重仅本地保存；本轮未验证连续视频中荧光消退后的动态保持。PR76是跟踪/评估修复，并未引入本轮的边界分支。

## 分支规则

- `main`:**稳定交付分支**,只有全部通过测试的代码才合并进来
- `develop`:**日常集成分支**,所有开发先合到这里
- 流程:`feature/xxx` → PR 到 `develop` → 全部通过后 `develop` → `main`
- 医学生在网页右上角分支下拉框选 `develop` 编辑文档即可

> 详细提交规则(提交到哪里、如何提交、AI 率控制)见 [提交规则.md](提交规则.md)。

## 数据红线(必读)

**患者胸腔镜图像 / 视频、标注图,严禁上传 GitHub(无论公开还是私有)。**

数据只放本地或网盘,位置在此说明(只写路径,不传数据):

- 本轮私有研究目录:`model/logs/patient_boundary_20261105_01/`（本地，Git忽略）
- 公开匿名登记:[数据说明](model/experiments/patient_boundary_20261105_01/DATASET.md)及[划分清单](model/experiments/patient_boundary_20261105_01/split_manifest.csv)
- 原图、原始标注和匿名映射由数据负责人另行提供；仓库不包含患者数据下载链接或模型权重。

## 协作约定

- **CS 同学**:用 git / 命令行,走分支 + PR
- **医学生**:用 GitHub 网页版(编辑文档、回复 issue、上传文件),不学 git
- **任务跟踪**:用 Issues,对照分工表拆成一条条 issue 派给对应人
