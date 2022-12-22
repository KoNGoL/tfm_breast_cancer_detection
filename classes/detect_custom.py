import argparse
import time
import sys
sys.path.insert(1, '/home/fundamentia/python/tfm_breast_cancer_detection/')
from pathlib import Path

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils_yolo.datasets import  LoadImages
from utils_yolo.general import check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh, set_logging, increment_path
from utils_yolo.plots import plot_one_box
from utils_yolo.torch_utils import select_device, time_synchronized, TracedModel
from classes.img_preprocess_utils import cut_boxes, save_box
from classes.bounding_box import BoundingBox

def detect(opt,save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16


    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        boxes = []
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # i = 1
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=None, line_thickness=1)
                    
                    # # recortamos las imagenes
                    # coord = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    # save_box("/home/fundamentia/python/corpus/transformadas_640/pruebas_yolo/recortadas/{}.png".format(i),
                    #                             cv2.imread(path), coord)
                    # i += 1
                    boxes.append(BoundingBox(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
                cv2.imwrite("/home/fundamentia/tmp/img.png", im0)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')


            # Save results (image with detections)
            # if save_img:
            #     cv2.imwrite(save_path, im0)
            #     print(f" The image with the result is saved in: {save_path}")
                

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    return boxes


def main(weights='/home/fundamentia/Descargas/best (2).pt', source='/home/fundamentia/python/corpus/transformadas_640/pruebas_yolo/', save_files = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights')
    parser.add_argument('--source')
    parser.add_argument('--img-size')
    parser.add_argument('--conf-thres')
    parser.add_argument('--iou-thres')
    parser.add_argument('--device')
    parser.add_argument('--view-img')
    parser.add_argument('--save-txt')
    parser.add_argument('--save-conf')
    parser.add_argument('--nosave')
    parser.add_argument('--classes')
    parser.add_argument('--agnostic-nms')
    parser.add_argument('--augment')
    parser.add_argument('--update')
    parser.add_argument('--project')
    parser.add_argument('--name')
    parser.add_argument('--exist-ok')
    parser.add_argument('--no-trace')
    args = parser.parse_args(['--weights', weights, '--source', source, '--device', '', '--project', 'runs/detect', '--name', 'exp'])
    # args = parser.parse_args(['--weights', weights])
    args.view_img = save_files
    args.save_txt = save_files
    args.nosave = not save_files
    args.agnostic_nms = False
    args.augment = False
    args.exist_ok = False
    args.img_size = 640
    args.conf_thres = 0.01
    args.iou_thres = 0.45
    args.save_conf = False
    args.update = False

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default=weights, help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default=source, help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', default = save_files, help='display results')
    # parser.add_argument('--save-txt', action='store_true', default = save_files, help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--nosave', action='store_true', default = not save_files, help='do not save images/videos')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    # opt = parser.parse_args()
    # print(opt)
    print(args)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        boxes = detect(args)
        print(boxes)
        return boxes

if __name__ == '__main__':
    main()
