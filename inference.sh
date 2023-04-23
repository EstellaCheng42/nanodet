python inference.py --demo image --config config/shufflev2_vis_train.yml  --model workspace/shufflev2_vis_train/model_best/nanodet_model_best.pth  --path D:/data/AIdea2023/Public_Private_Testing_Dataset 
    

python inf_track.py --config config/ck2.yml --model workspace/ck2/model_best/nanodet_model_best.pth --path 20230107_095033_20D6.mkv 

python predict_comp.py --config config/shufflev2_vis_train.yml --model workspace/shufflev2_vis_train/model_best/nanodet_model_best.pth --path D:\\data\\AIdea2023\\Public_Private_Testing_Dataset_Only_for_detection\\JPEGImages\\All --save_result

python export_onnx.py --cfg_path config/shufflev2_vis_train.yml --model_path workspace/shufflev2_vis_train/model_best/nanodet_model_best.pth --out_path workspace/shufflev2_vis_train/nanodet_model_best.onnx

python export_onnx.py --cfg_path config/fbnetv3c_416.yml --model_path workspace/fbnetc_416/model_best/nanodet_model_best.pth --out_path workspace/fbnetc_416/nanodet_model_best.onnx


python inference.py --demo image --config config/shufflev2_vis_train.yml  --model workspace/shufflev2_vis_train/model_best/nanodet_model_best.pth  --path sample_images 
