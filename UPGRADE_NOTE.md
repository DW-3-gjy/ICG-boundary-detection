# Native boundary refiner / 原图边界细化升级记录

2026-11-12. Base: `ab576212bc9cbcb8ad51fb0aca295535192a039d`; branch `feature/native-boundary-refiner`. Repository: https://github.com/DW-3-gjy/ICG-boundary-detection.git. Changes are local, not pushed.

## What was implemented / 已实现

An actual trainable four-channel RGB/coarse-probability strip CNN predicts bounded normal displacement or nuisance rejection. Unknown labels are ignored; rejection splits curves into open fragments without bridging gaps. It has patient-balanced training, patient/provenance manifest checks, validation checkpoint selection, load/predict CLI, optional baseline predictor integration, JSON/overlay export and geometry/gradient tests. Native geometry is original-image xy pixel centers. Defaults: normal radius32, tangent radius16, 24 hidden channels. Class indices 0..64 represent offsets -32..32; class65 is rejection. This is not calibrated visibility estimation.

实现可训练的局部条带 CNN：输入原图 RGB 和粗概率，输出法线方向有限位移或干扰拒绝。未知标注忽略，拒绝点将曲线断开，不跨遮挡连接。包含患者均衡训练、患者／粗模型训练来源检查、验证集选权重、训练与推理命令、可选基线集成、JSON 和叠加图输出。默认法线半径32、切线半径16、隐藏通道24；0..64类别对应 -32..32 像素，65为拒绝。这不是经过校准的可见性估计。

Upstream main is an older plain U-Net, NOT the November ResNet18 boundary experiment. Its imported data package was missing; restored binary-region data/preprocessing utilities so baseline tests run. Only paired flips are implemented; other configured augmentations are not inferred. Region masks/closed-contour tracking remain separate from refined open boundaries. The updated refiner can consume exported probabilities/curves from the November model without architecture assumptions.

上游 main 是旧版普通 U-Net，并非11月 ResNet18 实验，且缺失被导入的 data 包。本次补齐基础区域数据／预处理以运行原测试；仅实现成对翻转，不隐式实现其他增强。区域 mask／闭合轮廓跟踪与开放细化边界分开。11月模型可导出概率与曲线后直接接入细化器。

## Run / 运行

From `model/`, using Python with requirements installed:

```powershell
python -m pytest tests -q
python scripts/refine_boundary.py smoke --output logs/refiner_smoke
python scripts/refine_boundary.py train --manifest LOCAL/manifest.json --output checkpoints/refiner --device cuda --epochs 20
python scripts/refine_boundary.py predict --checkpoint checkpoints/refiner/best_refiner.pt --image LOCAL/frame.png --probability LOCAL/probability.npy --curves LOCAL/curves.json --output predictions/boundary.json --overlay predictions/boundary.png
```

The smoke command generates explicitly synthetic data, performs two training epochs, reloads the checkpoint and predicts. It is an execution check, NOT a performance benchmark. 本命令生成明确标注的合成数据，训练两轮、重载并推理，仅验证执行链路。

Manifest: JSON list; file paths relative to manifest. One row:

```json
{"patient_id":"anonymous_train_01","split":"train","image":"frame.png","probability":"coarse.npy","curves":"curves.json","labels":"boundary_labels.png","coarse_train_patients":["anonymous_other_01"]}
```

Require patient-disjoint `train` and `val`; no `test`. `coarse_train_patients` lists the patients actually used to train that row's coarse model; the row's patient and all outer-validation patients must be absent. Use inner patient-excluded coarse predictions for refiner training. Checks validate the declared manifest, not identity truth. Metadata without truthful provenance cannot establish independence.

必须提供患者不重叠的 train／val，不允许 test。每行 coarse_train_patients 真实列出生成该预测的粗模型见过的患者，不能含该行患者或任何外层验证患者。细化器训练使用内层按患者排除的粗预测；代码只能核对声明，不能证明身份记录真实。

`probability.npy`: finite foreground probabilities [0,1], HxW; resized bilinearly to native image if necessary. `curves.json`: list of ordered curves, each [[x,y],...], native coordinates, >=2 points. Caller controls sampling density; default maximum adjacent refined-point link is12 original pixels. Dense curves recommended. No automatic closure. Native `labels.png`: 255 reviewed visible interface, 0 reviewed nuisance/background, 128 unknown. Unmarked partial references must remain128. Multiple separated positive crossings along a normal are ignored as ambiguous. Inspect correspondence targets before patient training.

概率为有限 H×W、[0,1]；曲线为原图坐标的有序点列表，不自动闭合。推荐密集采样，默认修正后相邻点超过12原图像素即断开。原图标签：255经复核目标，0经复核负例，128未知；部分参考未标注处保留128。法线遇到多个分离正例交叉时忽略。患者训练前必须检查对应关系。

To enable ordinary predictor integration, set `refinement.enabled: true` and `refinement.checkpoint` in config. It proposes dense region contours and adds `refined_boundary_fragments` plus yellow open-curve overlay/JSON; original region mask and tracker are retained. This proposal adapter may miss interfaces absent from region contours; for November boundary-head proposals use the independent CLI/API. Video refined curves are framewise; no claim of temporal persistence.

配置打开 refinement.enabled 并指定 checkpoint 后，普通预测器用区域轮廓提供候选，增加开放细化片段、黄色叠加和 JSON；原区域／跟踪接口保留。区域候选本来没有的界面无法恢复，11月边界头候选请走独立 CLI/API。视频逐帧细化，不主张动态保持。

## Limits / 限制

No private patient data or trained clinical checkpoint supplied; no patient accuracy gain or novelty established. Training selection uses patient-macro validation CE, not boundary F1; thresholds need development calibration. The local refinement cannot recover entirely absent proposals. Compare rejection-only, no-refiner and positional controls before attributing a gain. Do not select on the already exposed test patients. Real-data training/evaluation is the teammate's next executable task, not a request to brainstorm another method.

未提供患者数据或临床权重，尚无患者精度提升或创新性结论。按患者宏平均验证 CE 选权重，阈值需开发集校准。完全漏检候选不可恢复。需对比仅拒绝、无细化及定位对照，不能用已公开测试患者选方法。队友下一步是运行现有升级并评价，不是再讨论选方法。
