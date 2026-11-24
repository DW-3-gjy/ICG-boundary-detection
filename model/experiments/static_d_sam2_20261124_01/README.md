# 静态 ICG 区域分割实验 · 2026-11-24

本轮从 11 月 5 日冻结的 D 组三个权重开始。原图、双尺度及水平翻转的前景概率融合后，用 Lab 修正生成框、点和掩膜提示。SAM 2.1 Large 的官方解码器与微调解码器等权融合，两路都使用原图和水平翻转预测。微调只更新掩膜解码器，开发折和最终重训各 180 步。

## 同图对照

| 方法 | 旧测试集患者等权 Dice |
| --- | ---: |
| 原冻结旧 D，三个种子的分数均值 | 84.10% |
| 旧 D 多尺度集成 | 87.19% |
| 集成及 Lab 修正 | 88.00% |
| 锁定的 Large 解码器融合 | **90.12%** |

历史回放使用两位患者的 7 张和 4 张完整掩膜。预测映射回原图后逐图计算区域 Dice，患者内取均值再患者等权。该结果属于已经查看过的旧测试集。新患者独立 Dice 尚未取得。

锁定配置在三折开发的 66 张完整掩膜上为 89.47%，在其中 41 张 Z 期图上为 91.10%。开发患者分组沿用冻结 D，只有 5 位患者有可用于区域 Dice 的完整掩膜。公开记录保留逐图分数、匿名分组、原始收据和源文件字节哈希。授权的患者文件与权重由数据负责人另行提供。

## 公开核验

```sh
python model/experiments/static_d_sam2_20261124_01/verify_public.py
```

只需 Python 标准库。核验重算公开的患者等权、分层和历史逐图数据，并检查原执行源码与训练收据的哈希。真实推理需要下述授权资产。

## 授权资产与运行入口

- 冻结基线来自 11 月 5 日联合研究（#78）。使用 `model/logs/patient_boundary_20261105_01/` 中原私有冻结协议、代码、图像与标注来源映射、开发缓存、折内 D 权重和三个最终 D 权重。`python model/scripts/static_d_baseline.py prepare` 核验后建立开发副本。
- 原运行环境为 Python 3.12.14、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、OpenCV 5.0.0、NumPy 2.5.4。SAM 2 源码固定在 `2b90b9f5ceec907a1c18123530e92e794ad901a4`。用 `--dependencies` 指向隔离的运行依赖目录。依赖版本和元数据 SHA-256 保存在候选收据的 `runtime_metadata_sha256`。本轮关闭 SAM 可选 CUDA 扩展后处理。
- 官方 [SAM 2.1 Large 权重](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) SHA-256 为 `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`。授权的重训解码器放在 `model/logs/static_d_sam2_large_refit_20261124_01/decoder.pt`，其 SHA-256 和三份旧 D 权重身份见 `records/static_d_sam2_large_refit_20261124_01/candidate.json`。

从仓库根目录运行。以下变量分别指向 SAM 源码、隔离依赖、官方 Tiny 和官方 Large 权重。已有冻结运行目录时，沿用收据并只运行所需的回放命令。

```sh
python model/scripts/train_static_d_sam2_decoder.py --source "$SAM_SOURCE" --dependencies "$SAM_DEPS" --weights "$SAM_TINY"
python model/scripts/evaluate_static_d_sam2_large.py --source "$SAM_SOURCE" --dependencies "$SAM_DEPS" --weights "$SAM_TINY" --large-weights "$SAM_LARGE"
python model/scripts/train_static_d_sam2_large_precision.py --source "$SAM_SOURCE" --dependencies "$SAM_DEPS" --weights "$SAM_LARGE"
python model/scripts/refit_static_d_sam2_large_decoder.py --source "$SAM_SOURCE" --dependencies "$SAM_DEPS" --weights "$SAM_LARGE"
python model/scripts/evaluate_static_d_sam2_historical.py --source "$SAM_SOURCE" --dependencies "$SAM_DEPS" --weights "$SAM_LARGE"
```

历史回放入口复测旧 D 与锁定版。精确复测使用运行产生的冻结 `candidate.json` 与 `decoder.pt`，开发对照按原研究目录名就位。`records/` 中的三份运行收据按原字节保存，供核对授权运行目录。GPU 重训的参数更新存在非确定性，实测报告绑定当次锁定权重。

独立患者评分使用 `refit_static_d_sam2_large_decoder.py` 的 `--manifest`、`--evaluation`、`--identity-confirmed`。先加 `--freeze-only` 冻结清单，再执行评分。清单列为 `sample_id,patient_id,image,reference,reference_kind`，支持同尺寸二值区域掩膜或 LabelMe 闭合多边形。`audit_independent_z_references.py` 可检查人工标注交接目录。病例图、标注、二值掩膜、模型权重、带图 PDF 和本机路径保留在授权的私有研究资料中。
