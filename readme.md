# Preliminary Preparation

1. Input images are put into `dataset\haze`, reference images are put into `dataset\clear`, UIEB test set images are put into `dataset\test\UIEB`, NYU-v2 test set images are put into `dataset\test\UWCNN`
2. The input image and the test image are named uniformly as `dataset name_image parent number_sub number_sub number`, for example `UIEB_123_1_1`, and the reference image is named uniformly as `dataset name_image number`, for example `UIEB_123`, and the `dataset name_image parent number` of the input image and the reference image must correspond, and the same for the test image.
3. Put the model `(*.pth)` into the `snapshots` folder
4. Modify the path information in the `tran.py`
   1. UIEB results folder: `config.gen_UIEB_dir`
   2. NUYv2 results folder: `config.gen_UWCNN_dir`
   3. Input image map: `config.hazy_images_path`
   4. Reference image set: `config.orig_images_path`
   5. UIEB test set: `config.test_UIEB_dir`
   6. NUYv2 test set: `config.test_UWCNN_dir`
   7. Model storage folder: `config.snapshots_folder`
5. Modify the path information in the `test.py` file
   1. Reference image set: `real_file_path`

   2. UIEB test set: `UIEB_test_path`
   3. NUYv2 test set: `UWCNN_test_path` 
   4. UIEB results folder: `UIEB_result_path`
   5. NUYv2 results folder: `UWCNN_result_path`
   6. 
      Model path: `modle_path

# testing

```
python test.py
```

# training

```
python train.py 
```

