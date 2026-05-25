from ultralytics import YOLO
import cv2
from typing import List, Tuple, Optional
import numpy as np

class YOLODetector:
    """
    YOLO 目标检测封装类，支持单张图像推理、标注图像生成、检测结果提取，
    以及判断图像中是否包含“人”。
    """

    def __init__(self, model_path: str = "yolo26x.pt"):
        """
        初始化检测器。

        Args:
            model_path: 训练好的 YOLO 权重文件路径（.pt）。
        """
        self.model = YOLO(model_path)
        self.names = self.model.names  # 类别名称字典

    def predict(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        save_annotated: bool = False,
        output_path: Optional[str] = None,
        show: bool = False
    ) -> Tuple[List[dict], np.ndarray]:
        """
        对单张图像进行目标检测。

        Args:
            image_path: 输入图像路径（支持本地文件、URL）。
            conf: 置信度阈值。
            iou: NMS 的 IoU 阈值。
            save_annotated: 是否保存标注后的图像。
            output_path: 标注图像的保存路径（若为 None，自动在原文件名后加 "_annotated"）。
            show: 是否弹窗显示标注图像（会阻塞程序，按任意键关闭）。

        Returns:
            detections: 检测结果列表，每个元素为字典，包含：
                - 'bbox': (x1, y1, x2, y2) 整数坐标
                - 'confidence': 浮点数置信度
                - 'class_id': 整数类别 ID
                - 'class_name': 类别名称字符串
            annotated_image: 标注后的图像（BGR 格式的 numpy 数组）。
        """
        # 推理
        results = self.model(image_path, conf=conf, iou=iou)
        result = results[0]  # 单张图像取第一个结果

        # 提取检测信息
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()      # (N,4) 绝对坐标
            confs = result.boxes.conf.cpu().numpy()      # (N,)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

            for box, conf_val, cls_id in zip(boxes, confs, class_ids):
                x1, y1, x2, y2 = map(int, box)  # 转为整数
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(conf_val),
                    'class_id': cls_id,
                    'class_name': self.names[cls_id]
                })

        # 生成标注图像
        annotated_img = result.plot()  # 返回 BGR numpy 数组，已绘制框和标签

        # 保存
        if save_annotated:
            if output_path is None:
                # 自动在源文件名后添加 "_annotated"
                if '.' in image_path:
                    base, ext = image_path.rsplit('.', 1)
                    output_path = f"{base}_annotated.{ext}"
                else:
                    output_path = image_path + "_annotated.png"
            cv2.imwrite(output_path, annotated_img)
            print(f"标注图像已保存至: {output_path}")

        # 显示
        if show:
            cv2.imshow("YOLO Detection", annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detections, annotated_img

    def predict_and_print(self, image_path: str, **kwargs):
        """
        便捷方法：执行检测并打印结果信息。
        """
        detections, _ = self.predict(image_path, **kwargs)
        print(f"检测到 {len(detections)} 个目标：")
        for i, det in enumerate(detections, 1):
            print(f"{i}. {det['class_name']} (置信度: {det['confidence']:.2f}) - 位置: {det['bbox']}")
        return detections

    def has_person(self, image_path: str, conf: float = 0.25, iou: float = 0.45) -> bool:
        """
        判断图像中是否有人。

        Args:
            image_path: 输入图像路径。
            conf: 置信度阈值。
            iou: NMS 的 IoU 阈值。

        Returns:
            True 表示图像中至少检测到一个人，否则 False。
        """
        detections, _ = self.predict(
            image_path,
            conf=conf,
            iou=iou,
            save_annotated=False,
            show=False
        )
        # 只判断类别名称为 'person' 的目标（COCO 数据集标准名称）
        for det in detections:
            if det['class_name'] == 'person':
                return True
        return False


# ================= 使用示例 =================
if __name__ == "__main__":
    # 初始化检测器（请确保 yolo26x.pt 文件存在）
    detector = YOLODetector("yolo26x.pt")

    # 示例图像路径（请替换为您自己的图像）
    test_image = "image.png"

    # 1. 完整检测：获取所有目标，保存标注图，并显示
    detections, annotated = detector.predict(test_image, save_annotated=False, show=True)

    # 2. 仅判断是否有人
    if detector.has_person(test_image):
        print("图像中检测到了人！")
    else:
        print("图像中未检测到人。")