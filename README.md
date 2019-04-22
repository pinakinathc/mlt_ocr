# A Multi-Lingual OCR using CTC
-----------------------------------
This repository contains only the OCR unit that is used in [this](https://github.com/MichalBusta/E2E-MLT) repository.

Please use Anaconda or miniconda for installation.

To run this model, you would need the following steps:
* `conda env create -f environment.yml`
* `wget http://ptak.felk.cvut.cz/public_datasets/SyntText/e2e-mlt.h5`
* `conda activate ocr`

OR

simply run:
* `bash setup.sh`
* `conda activate ocr`

```make sure that you have a GPU```

now you have 2 choices.

Run the OCR on images present in `input_data` and save the output in `output_data`:

`bash start.sh`

**Please Note**: the format in which the recognition result is saved is:

`<image_name>_<recogintion_result>.png`

Example:

if your image name is: `img_1.jpg`, **and** your **recognition result is:** *hello_world*.

The output image name would be: *img_1_hello_world.png*.

## how to run images present in some random <input_image_path> and store output in some random <output_image_path>?
to do this **instead of** `bash start.sh` run: `python eval.py -input-path=your_random_image_path -output_path=your_random_image_path`

## Reference:
```
@article{buvsta2018e2e,
  title={E2E-MLT-an unconstrained end-to-end method for multi-language scene text},
  author={Bu{\v{s}}ta, Michal and Patel, Yash and Matas, Jiri},
  journal={arXiv preprint arXiv:1801.09919},
  year={2018}
}
```

## About me: Go to http:www.pinakinathc.me