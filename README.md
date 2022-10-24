# SW-AI-Korean-OCR

## Model 
https://github.com/Belval/TextRecognitionDataGenerator  
https://github.com/clovaai/deep-text-recognition-benchmark

## 데이터 출처
[aihub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=105)  


## Create Text Image
```
python3 trdg/run.py -c 10000 -l ko -f 128 -k 15 -rk -bl 2 -rbl -b 3 --output_dir input/trdg -w 1 --font_dir trdg/fonts/ko
```

## Create LMDB file
```
python3 deep-text-recognition-benchmark/create_lmdb_dataset.py --inputPath ./ --gtFile gt_external.txt --outputPath lmdb/aihub
```


## Data Infomation
### Directory Tree
```
├─lmdb
│  ├─train
│  │  ├─aihub
│  │  └─trdg
│  └─val
│     ├─aihub
│     └─trdg
├─test
│ ├─test_00001.png
│ ├─test_00002.png
│ ├─test_00003.png
│ ├─test_00004.png

```
### Train image
dataset_root:    lmdb/train	 dataset: aihub  
sub-directory:	/aihub	 num samples: 600,322  
sub-directory:	/trdg	 num samples: 324,000  
**Tootal num samples: 924,322**

### Validation Image
dataset_root:    lmdb/val	 dataset: /  
sub-directory:	/aihub	 num samples: 12,114  
sub-directory:	/trdg	 num samples: 36,000  
**Tootal num samples: 48,114**

## Model Training
```
python3 deep-text-recognition-benchmark/train.py \
  --train_data lmdb/train \
  --valid_data lmdb/val \
  --num_iter 50000 \
  --select_data aihub-trdg \
  --batch_ratio 1-1 \
  --Transformation TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction Attn \
  --language ko-sh \
  --workers 0 \
  --valInterval 500 \
  --manualSeed 4444 \
  --batch_size 128
```

## Model Test
```
python3 deep-text-recognition-benchmark/demo.py \
  --Transformation TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction Attn \
  --image_folder test \
  --saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Seed4444/best_accuracy.pth \
  --language ko-sh \
  --output_name log_demo_pred.csv
```
