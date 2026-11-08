# tools/

CS-B 负责:标注工具配置、数据预处理脚本。

- 标注工具:LabelStudio / Labelme(部署后给医学生网页使用)
- 数据预处理:图像格式转换、裁剪、增强

## 本地训练进度窗口

使用已经安装LabelMe的Python环境运行，无需额外安装依赖：

```powershell
.venv-labelme/Scripts/pythonw.exe tools/training_progress_window.py
```

窗口每2秒读取研究日志，展示总进度、当前更新、五组状态和粗略剩余时间；整体百分比按训练运行次数计算，不代表耗时占比。只读取JSON日志，不读取模型、患者图像或测试缓存。关闭窗口不会停止训练。项目工作目录另有“查看训练进度.lnk”快捷方式。
