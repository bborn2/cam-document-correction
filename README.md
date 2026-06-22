# A4 纸实时检测与高亮

用 Windows 摄像头实时拍摄桌面，检测画面中的 **A4 纸**，并以**旋转框（OBB 风格）高亮**显示。

## 检测器（五选一）

| 后端 | 说明 | 依赖 |
|---|---|---|
| `pose` | **YOLOv8-Pose 关键点模型**，直接预测 A4 四个角点，并在新窗口显示**透视矫正**后的展平文档 | onnxruntime |
| `seg` | **YOLOv8-Seg 分割模型**，从掩码轮廓算出 A4 四个顶点并画出（红点+编号） | onnxruntime |
| `obb` | **YOLOv8-OBB 模型**，旋转框输出 | onnxruntime |
| `cv` | 传统 CV 轮廓检测，免训练、即开即用 | opencv |
| `yolo` | ultralytics `.pt` 直接推理（需 torch） | ultralytics |

> 注：A4 纸没有现成深度学习预训练模型（YOLOv8-OBB 官方是 DOTA 航拍类、Seg 是 COCO 无"纸"类）。
> 本项目的模型是在自标注 A4 数据集上训练的单类模型（类别 `document`，显示为 `A4`）。

## 安装

```bash
uv python pin 3.12   # 已固定；torch/ultralytics 暂不支持 3.14
uv sync              # 核心依赖：numpy + opencv + onnxruntime（无 torch）
```

## 运行

```bash
uv run a4detect --detector seg --weights best.onnx --imgsz 640 --warp --max-res --flip180

# 用 Pose 关键点模型，检测四角 + 透视矫正展平（新窗口）
uv run a4detect --detector pose --weights best-pose.onnx

# 用分割模型，画出 A4 四个顶点
uv run a4detect --detector seg --weights best.onnx

# 用 OBB 模型（旋转框）
uv run a4detect --detector obb --weights best-30.onnx

# 不用模型，传统 CV
uv run a4detect

# 指定摄像头 / 调阈值
uv run a4detect --detector pose --weights best-pose.onnx --camera 1 --conf 0.3
```

窗口中把 A4 纸放入视野即可看到绿色旋转高亮框。按 `q` 或 `ESC` 退出。

## 从 .pt 导出 ONNX（如需重新导出）

```bash
uv sync --extra yolo     # 装 ultralytics + torch（仅导出/训练时需要）
uv run yolo export model=best.pt format=onnx imgsz=640
```

## 项目结构

```
src/a4detect/
├── main.py              # 入口与实时主循环（--detector cv|obb|seg|pose|yolo）
├── camera.py            # 摄像头封装（CAP_DSHOW，Windows 友好）
├── detector.py          # Detector 抽象接口 + Detection 数据结构
├── cv_detector.py       # 传统 CV 检测 A4
├── onnx_detector.py     # ONNX Runtime YOLOv8-OBB 推理（旋转框）
├── onnx_seg_detector.py # ONNX Runtime YOLOv8-Seg 推理（掩码→顶点）
├── onnx_pose_detector.py# ONNX Runtime YOLOv8-Pose 推理（角点+透视矫正）
├── smoothing.py         # 跨帧时间平滑（防抖）
├── yolo_detector.py     # ultralytics .pt 推理（可选）
└── draw.py              # 高亮 + 顶点绘制
```

## 调参

- 模型：`--conf`（置信度阈值，默认 0.25）、`--iou`（NMS，默认 0.45）。
- 防抖：`--alpha`（越小越稳，默认 0.5）、`--no-smooth`（关闭平滑）。
- CV：调 `cv_detector.py` 顶部常量 `CANNY_LOW/HIGH`、`MIN_AREA_RATIO`、`RATIO_TOLERANCE`。
