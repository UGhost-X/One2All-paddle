import sys
import os
import platform

# 跨平台：Windows需要添加OpenCV DLL路径
if platform.system() == "Windows":
    # 可以通过环境变量指定OpenCV路径，或使用默认路径
    opencv_bin = os.environ.get("OPENCV_BIN", r"D:\software\opencv\build\x64\vc16\bin")
    if os.path.exists(opencv_bin):
        os.add_dll_directory(opencv_bin)

import numpy as np
import cv2

# 跨平台：动态确定模块路径
# 优先级：1.环境变量 2.相对路径 3.默认构建路径
module_paths = [
    os.environ.get("SHAPE_MATCHING_PY_PATH"),  # 环境变量指定
    os.path.join(os.path.dirname(__file__), "build", "Release"),  # Windows构建目录
    os.path.join(os.path.dirname(__file__), "build"),  # Linux构建目录
]

for path in module_paths:
    if path and path not in sys.path and os.path.exists(path):
        sys.path.insert(0, path)
        break

import shape_based_matching_py as sbm

# 跨平台路径处理
prefix = os.path.join(os.path.dirname(__file__), "test")


def train():
    img_path = os.path.join(prefix, "case4", "train-1.png")
    img = cv2.imread(img_path)
    assert img is not None, f"check your img path: {img_path}"

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = np.ones(img.shape, dtype=np.uint8) * 255

    shapes = sbm.ShapeInfoProducer(img, mask)
    shapes.setAngleRange(0, 360)
    shapes.setAngleStep(1)
    shapes.produceInfos()

    detector = sbm.Detector(128, [4, 8])

    infos_have_templ = []
    class_id = "test"

    infos = shapes.getInfos()
    print(f"Total infos to process: {len(infos)}")

    for i, info in enumerate(infos):
        if i % 30 == 0:
            print(f"Processing {i}/{len(infos)}, angle: {info['angle']}")

        src = shapes.srcOf(info)
        mask_transformed = shapes.maskOf(info)

        templ_id = detector.addTemplate(src, class_id, mask_transformed)

        if templ_id != -1:
            infos_have_templ.append(info)

    # 跨平台路径
    templ_path = os.path.join(prefix, "case4", "%s_templ.yaml")
    detector.writeClasses(templ_path)

    info_path = os.path.join(prefix, "case4", "test_info.yaml")
    shapes.saveInfos(infos_have_templ, info_path)

    print(f"\nTrain end. Total templates: {len(infos_have_templ)}")


if __name__ == "__main__":
    train()
