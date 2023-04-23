import argparse
import os
import time

import cv2
import torch

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]

#These are the labels used in training ['pedestrian', 'vehicle', 'scooter', 'bicycle']
#These labels are used for submission:  1:vehicle 2:pedestran 3:scooter 4:bicycle
label_map = {0:2, 1:1, 2:3, 3:4 }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="model config file path")
    parser.add_argument("--model", help="model file path")
    parser.add_argument("--path", default="./demo", help="path to images")
    args = parser.parse_args()
    return args


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names


def main():
    args = parse_args()
    local_rank = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device="cuda:0")
    current_time = time.localtime()
    
    save_folder = os.path.join(
        cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    mkdir(local_rank, save_folder)
    with open(save_folder + '/submission.csv', 'w') as writer:
        writer.write('image_filename,label_id,x,y,w,h,confidence\n')

    if os.path.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    confidence_th = 0.35
    for image_name in files:
        _, res = predictor.inference(image_name)
        image = os.path.basename(image_name)
        for label_id in range(4):
            dets = res[0][label_id]
            dets = [x for x in dets if x[-1]>confidence_th]
            if len(dets) > 0:
                for det in dets:
                    x1, y1, x2, y2, conf = tuple(det)
                    x = str(round(x1))
                    y = str(round(y1))
                    w = str(round(x2-x1))
                    h =  str(round(y2-y1))
                    conf = str(conf)
                    with open(save_folder + '/submission.csv', 'a') as appd:
                        result = ','.join([image,str(label_map[label_id]), x,y,w,h,conf])
                        appd.write(result+'\n')

    


if __name__ == "__main__":
    main()
