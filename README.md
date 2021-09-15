# Low-Latency Auditory Spatial Attention Detection Based on Spectro-Spatial Features from EEG

This repository contains the python scripts developed as a part of the work presented in the paper "Low-Latency Auditory Spatial Attention Detection Based on Spectro-Spatial Features from EEG"

## Getting Started

### Dataset

The public [KUL dataset](https://zenodo.org/record/3997352#.YUGaZdP7R6q) is used in the paper. The dataset itself comes with matlab processing program, please adjust it according to your own needs.

### Prerequisites

- python 3.7.9
- tensorflow 2.2.0
- keras 2.4.3

### Run the Code

1. Download the preprocessed data from [here](https://mailscuteducn-my.sharepoint.com/:f:/g/personal/202021058399_mail_scut_edu_cn/Evu3JoynOJxJlYtpKft2UfIBcZuNbkSrbymvDHLNdpiK9w?e=gWx9J0).

2. Modify the `args.data_document_path` variable in model.py to point to the downloaded data folder

3. Run the model:

   ```powershell
   python model.py
   ```

4. If you want to run multiple subjects in parallel, you can modify the variable `path` in multi_processing.py and run:

   ```powershell
   python multi_processing.py
   ```

## Paper

![model](./pic/model.png)

Paper Link: [**Low-Latency Auditory Spatial Attention Detection Based on Spectro-Spatial Features from EEG**](https://arxiv.org/abs/2103.03621)

The proposed convolutional neural network (CNN) with spectro-spatial feature (SSF) for auditory spatial attention detection, that is referred to as SSF-CNN model. The SSF-CNN network is trained to output two values, i.e., 0 and 1, to indicate the spatial location of the attended speaker.

Please cite our paper if you find our work useful for your research:

```tex
@article{cai2021low,
  title={Low-latency auditory spatial attention detection based on spectro-spatial features from EEG},
  author={Cai, Siqi and Sun, Pengcheng and Schultz, Tanja and Li, Haizhou},
  journal={arXiv preprint arXiv:2103.03621},
  year={2021}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Contact

Siqi Cai, Pengcheng Sun, Tanja Schultz , and Haizhou Li

Siqi Cai, Pengcheng Sun and Haizhou Li are with the Department of Electrical and Computer Engineering, National University of Singapore, Singapore.

Tanja Schultz is with Cognitive Systems Lab, University of Bremen, Germany.

