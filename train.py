import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('/home/cbr-detr/ultralytics/cfg/models/rt-detr/cbr-detr.yaml')
    model.load('/home/cbr-detr/weights/rtdetr-weights-new/weights/rtdetr-r18.pt') # loading pretrain weights
    model.train(data='/home/cbr-detr/dataset/data.yaml',
                cache=True,
                imgsz=640,
                epochs=150,#300,
                batch=4, #4 
                workers=4, #4 
                device='0', 
                # resume='', # last.pt path
                # amp=True,            # 开启混合精度
                project='runs/train',
                name='exp',
                )