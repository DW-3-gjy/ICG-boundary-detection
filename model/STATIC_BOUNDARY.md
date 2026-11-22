# 静态 ICG 边界精修训练与对照

`scripts/static_boundary_compare.py` 从同一组静态图像、粗模型和患者划分生成四组结果。

| 对照 | 最终边界 | 区域掩膜 |
| --- | --- | --- |
| `coarse_d` | D 组边界头骨架 | D 组原始掩膜 |
| `local` | D 组加原图条带 CNN | D 组原始掩膜 |
| `bilateral` | 双侧组织对照加定位可靠性判断 | D 组原始掩膜 |
| `bilateral_region` | 与 `bilateral` 相同 | 对可靠、单一的原有区域过渡作局部位移 |

`bilateral` 通过每个法向候选位置两侧留出间隔的特征池计算偏移。两侧共用局部亮度坐标，保留相对差异。训练时翻转法向和切向并同步翻转定位标签。推理同时判断干扰概率和预测位置附近的概率质量。参考线不清楚时可断开片段。`bilateral_region` 只对粗掩膜中有明确前景与背景过渡的位置作小范围修正。开放线不会被填成区域，且缺少可信过渡时沿用原掩膜。

相关方法依据包括 Singaravelu 等的跨界采样、Dalli 等的归一化比较原则、Mao 等的组织支持和可见性思路，以及专利的粗分割与局部边界重定位。此实现以单帧双侧特征和位置可靠性估计为核心。Mao 等的点追踪与各文献的时间曲线属于视频流程。

- [Singaravelu 2024，International Journal of Surgery](https://doi.org/10.1097/JS9.0000000000002136)
- [Mao 2026，International Journal of Computer Assisted Radiology and Surgery](https://link.springer.com/article/10.1007/s11548-026-03733-w)
- [Dalli 2024，Surgical Endoscopy](https://pmc.ncbi.nlm.nih.gov/articles/PMC11133155/)
- [CN120107301A，《图像处理方法及装置》](https://patents.google.com/patent/CN120107301A/zh)

## 准备数据

使用 Python 3.10 或以上和与设备匹配的 PyTorch、Torchvision。从仓库根目录安装其余依赖。

```powershell
python -m pip install -r model/requirements-resnet.txt
```

在本地建立 `model/data/static_manifest.csv`。相对路径以 CSV 所在目录为基准。建议使用匿名患者 ID。同一患者的全部图片放在同一分组。

```csv
sample_id,patient,split,image,annotation,kind,complete_boundary,valid_mask
s01,p01,coarse,images/s01.png,labels/s01.json,labelme,true,
s02,p02,train,images/s02.png,labels/s02.json,labelme,true,
s03,p03,val,images/s03.png,labels/s03.png,region_mask,true,
s04,p04,test,images/s04.png,labels/s04.png,boundary_mask,false,review/s04.png
```

- `labelme` 支持同一图中的多边形或开放折线。`region_mask` 为完整前景掩膜，`boundary_mask` 为非零像素标注的开放线。`tristate` 用 255 表示目标边界，0 表示已复核负例，128 表示未知。
- `complete_boundary=true` 表示项目要识别的可见边界已标全。否则在 `valid_mask` 标记可评价的范围，或使用三态标签。只有已复核负例才用于训练拒绝机制。数据类型和标注规则在对照组间相同。
- `coarse` 拟合粗 D 模型，`train` 拟合精修器，`val` 选择精修检查点，`test` 做最终对照。比较流程在锁定精修检查点前不推理测试图片。

## 训练和比较

**已有合适的 D 权重时**，可以复用带 `model` 和 `threshold` 字段的 D 组 `best.pt`。在本地建立 `model/data/coarse_plan.json`。下例权重的训练、选模、阈值拟合患者为 `p01`，用它向另外三位患者生成预测。

```json
[
  {
    "checkpoint": "checkpoints/D_best.pt",
    "fitted_patients": ["p01"],
    "predict_patients": ["p02", "p03", "p04"]
  }
]
```

如要复用折外训练权重，可添加多条路由。`train` 可按患者走各自真实折外检查点，`val` 和 `test` 必须使用同一个粗检查点。`fitted_patients` 包括曾用来选检查点或阈值的患者。已经用全体开发数据拟合过的历史 D 权重，不能直接用于向同一批患者生成精修训练输入。已有旧测试患者可用于历史回归观察，需要用新增独立患者确认新收益。

```powershell
python model/scripts/static_boundary_compare.py compare --manifest model/data/static_manifest.csv --coarse-plan model/data/coarse_plan.json --device cuda --seed 42 --output model/logs/static_s42
```

**没有合适的 D 权重时**，保留 CSV 的独立 `coarse` 患者，用现有 D 组主干和损失训练固定预算的粗模型。该入口使用官方 ImageNet ResNet18 权重。权重可由 Torchvision 下载或提前缓存。粗模型训练使用单图片批量，产生这一实验自己的同期基线。已有合适权重时可以省掉该阶段。

```powershell
python model/scripts/static_boundary_compare.py compare --manifest model/data/static_manifest.csv --device cuda --seed 42 --output model/logs/static_s42
```

模型和日志写入新目录，单次对照使用同一个种子。建议正式对照重复 42、43、44，输出目录分别取名。显存不足时先下调条带阶段的 `--batch-size`。本轮主要看 `f1_10`、`f1_20` 和 `recall_20`，完整区域标注的帧另看 `region_dice`。`region_changed_frames` 显示局部区域修正实际改变了多少帧。边界按原图坐标用相同弧长间距采样，分别逐图、患者内再患者等权汇总。空预测保留在 F1 统计中。对照结果见 `results.json` 和四份 `*_per_image.csv`。

以下两组比较分别回答精修和区域的问题。

- `local` 对 `bilateral` 比较双侧组织和位置判断的整体增益。可新建另一个训练目录传 `--position-mass 0`，观察关闭位置判断后的表现。需要扩展粗候选时，再在开发数据比较 `--candidate-threshold` 的影响。
- `coarse_d` 对 `bilateral_region` 比较区域 Dice。其他两组继承粗掩膜。F1 和 Dice 口径及分母单独报告。现有约 84% 的历史 Dice 只作为背景，训练时用同一批图上的 `coarse_d` 作为实际对照。

## 单张图片查看结果

```powershell
python model/scripts/static_boundary_compare.py predict --image model/data/images/s04.png --coarse model/logs/static_s42/coarse.pt --refiner model/logs/static_s42/bilateral.pt --correct-region --device cuda --output model/predictions/static_s04
```

采用复用权重时将 `--coarse` 指向对应 D 权重。输入大小、采样间距和可选候选阈值与对照命令一致。输出包含 `overlay.png`、`region_mask.png` 和原图坐标的 `boundary.json`。对照输出目录和患者图片、标注、掩膜、权重保持本地。
