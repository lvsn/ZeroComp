<h1 align="center"> ZeroComp: Zero-shot Object Compositing from Image Intrinsics via Diffusion </h1>

<p align="center"><a href="https://zzt76.github.io/" target="_blank">Zitian Zhang</a>, <a href="https://ca.linkedin.com/in/lefreud/en" target="_blank">Frédéric Fortier-Chouinard</a>, <a href="https://mathieugaron.ca/" target="_blank">Mathieu Garon</a>, <a href="https://anandbhattad.github.io/" target="_blank">Anand Bhattad</a>, <a href="https://vision.gel.ulaval.ca/~jflalonde/" target="_blank">Jean-François Lalonde</a>

<p align="center">WACV 2025 (Oral)</p>

<p>               
 <center>
    <span style="font-size:24px"><a href='https://lvsn.github.io/ZeroComp/'>[Website]</a></span>
    <span style="font-size:24px"><a href='https://arxiv.org/abs/2410.08168'>[Paper]</a></span>
    <span style="font-size:24px"><a href='https://lvsn.github.io/ZeroComp/supp/index.html'>[Supplementary]</a></span><br>
</center>
</p>

## Environments

1. Clone the repo and submodules
```bash
git clone --recursive https://github.com/zzt76/zerocomp.git
```
Make sure you also clone the submodule ```predictors```.

2. Install required wheels, note that different intrinsic predictors may require different libraries.
```bash
pip install -r requirements.txt
```

3. Download [Stable Diffussion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) from huggingface, and modify the ```pretrained_model_name_or_path``` variable in ```configs/sg_labo.yaml```.


## Pretrained weights

Pretrained weights are only available for non-commercial use under CC BY-NC-SA 4.0<img
style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"
src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img
style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"
src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img
style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"
src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img
style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"
src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a> license.

```Depth, normals, albedo```
Openrooms 7days: [link](https://hdrdb-public.s3.valeria.science/zerocomp/openrooms_7days.zip)

```Normals, albedo```
Openrooms 2days: [link](https://hdrdb-public.s3.valeria.science/zerocomp/openrooms_2days_wo_depth.zip)

```Depth, normals, albedo, roughness, metallic```
Interior Verse 2days: [link](https://hdrdb-public.s3.valeria.science/zerocomp/interior_verse_2days.zip)
Interior Verse 7days: [link](https://hdrdb-public.s3.valeria.science/zerocomp/interior_verse_7days.zip)

## Tips for the user cases when the footprint depth of the object is not available
In our paper, the footprint depth of the object is needed to align the object depth with the background depth (or the other way around if the depth is relative disparity). However, we notice that the footprint depth is not always available. So here we provide two different solution:
1. In the newest version, when the footprint depth is not available, we use the smallest bg depth value inside the object mask as the minimum object depth. Please refer to Line 274-292.
2. Another solution is you can use the pretrained model without the depth channel. You can download this model with the link provided above, and change the following arguments in the config file (as in configs/sg_labo_wo_depth.yaml):
```yaml
conditioning_maps: [normal, diffuse, shading, mask]

eval:
    controlnet_model_name_or_path: checkpoints/openrooms_2days_wo_depth
    shading_maskout_mode: BBox
    shading_maskout_bbox_dilation: 50 # This is a hyperparameter deciding how large we should mask around the object
``` 


## Predictors
You can get the intrinsic predictor weights from the original repos, the links are provided in the following. After downloading, move them to ```.cache/checkpoints``` folder.

#### Depth
ZoeDepth: [ZoeD_M12_NK.pt](https://github.com/isl-org/ZoeDepth)
DepthAnything: [depth_anything_metric_depth_indoor.pt](https://github.com/LiheYoung/Depth-Anything)
DepthAnythingV2: [depth_anything_v2_vitl.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)

#### Normals
OmniDataV2: [omnidata_dpt_normal_v2.ckpt](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch)
StableNormal: [stable-normal-v0-1](https://github.com/Stable-X/StableNormal)

#### Materials
For diffuse only, you can use our own not that good model [dfnet](https://hdrdb-public.s3.valeria.science/zerocomp/dfnet_w_hypersim2.safetensors).
For diffuse, roughness and metallic, you can precompute these maps by [IntrinsicImageDiffusion](https://github.com/Peter-Kocsis/IntrinsicImageDiffusion), [RGB<->X](https://github.com/zheng95z/rgbx) or other predictors. Name them as in the provided test dataset and load them by changing ```predictor_names``` to ```precompute```.

#### Custom predictors
You can use other predictors you prefer by modifying ```controlnet_input_handle.py```. Implement ```handle_***``` functions and modify ```ToPredictors``` class.

All predictors are subject to their own licenses. Please check the relating conditions carefully.

## ZeroComp Test dataset
You can download the ZeroComp test dataset [here](https://hdrdb-public.s3.valeria.science/zerocomp/labo.zip).

## Inference
To run the evaluations as in the paper:
```bash
python eval_controlnet_composite.py --config-name sg_labo
```

## Live demo
ZeroComp trained on Openrooms:
```bash
python gradio_composite.py
```

ZeroComp trained on InteriorVerse, with roughness and metallic:
```bash
python gradio_composite_w_rm.py
```

## Training
### Openrooms dataset
The Openrooms dataset should be structured as follows:
```
openrooms_mainxml1/
├── Geometry/
│   └── main_xml1/
│       └── scene0001_00/
│           ├── imdepth_1.dat
│           ├── imnormal_1.png
├── Image/
│   └── main_xml1/
│       └── scene0001_00/
│           ├── im_1.hdr
│           ├── im_1.png
├── Mask/
│   └── main_xml1/
│       └── scene0001_00/
│           ├── immask_1.png
├── Material/
│   └── main_xml1/
│       └── scene0001_00/
│           ├── imbaseColor_1.png
│           ├── imroughness_1.png
│           ├── immetallic_1.png
└── Shading/
    └── main_xml1/
        └── scene0001_00/
            ├── imshading_1.png
            ├── imshadow_1.png
```
### Training command
On a single GPU, you can run:
```bash
python train_controlnet.py --config-name train_openrooms
```
For multi-GPU training, you can use:
```bash
accelerate launch train_controlnet.py --config-name train_openrooms
```

## Acknowledgements
This research was supported by NSERC grants RGPIN 2020-04799 and ALLRP 586543-23, Mitacs and Depix. Computing resources were provided by the Digital Research Alliance of Canada. We also thank Louis-Étienne Messier and Justine Giroux for their help as well as all members of the lab for discussions and proofreading help.

This implementation builds upon Hugging Face’s [Diffusers](https://github.com/huggingface/diffusers) library. We also acknowledge [Gradio](https://www.gradio.app/) for providing a developer-friendly tool to create the interative demos for our models.

## BibTex
If you find it useful, please consider citing ZeroComp:
```
@InProceedings{zhang2025zerocomp,
    author    = {Zhang, Zitian and Fortier-Chouinard, Fr\'ed\'eric and Garon, Mathieu and Bhattad, Anand and Lalonde, Jean-Fran\c{c}ois},
    title     = {ZeroComp: Zero-Shot Object Compositing from Image Intrinsics via Diffusion},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {483-494}
}
```
Our follow-up work [SpotLight](https://lvsn.github.io/spotlight/) focuses on training-free local relighting, please feel free to check it out if you're interested :)
```
@misc{fortierchouinard2025spotlightshadowguidedobjectrelighting,
      title={SpotLight: Shadow-Guided Object Relighting via Diffusion}, 
      author={Frédéric Fortier-Chouinard and Zitian Zhang and Louis-Etienne Messier and Mathieu Garon and Anand Bhattad and Jean-François Lalonde},
      year={2025},
      eprint={2411.18665},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.18665}, 
}
```

## License
The codes, pretrained weights and test dataset are all for non-commercial use only.

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title"
        rel="cc:attributionURL" href="https://lvsn.github.io/ZeroComp/">ZeroComp: Zero-shot Object Compositing from
        Image Intrinsics via Diffusion</a> by
    <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://zzt76.github.io/">Zitian
        Zhang</a>, Frédéric Fortier-Chouinard, Mathieu Garon, Anand Bhattad,
    Jean-François Lalonde is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1"
        target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img
            style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"
            src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img
            style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"
            src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img
            style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"
            src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img
            style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"
            src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a>
</p>
