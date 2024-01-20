<p align="center">
  <img src="https://photo-maker.github.io/assets/logo.png" height=100>

</p>

<!-- ## <div align="center"><b>PhotoMaker</b></div> -->
<div align="center">
  
## PhotoMaker: 通过堆叠 ID Embedding 来生成逼真的人像照片  [![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-md-dark.svg)](https://huggingface.co/papers/2312.04461)
[[论文](https://huggingface.co/papers/2312.04461)] &emsp; [[项目地址](https://photo-maker.github.io)] &emsp; [[模型信息](https://huggingface.co/TencentARC/PhotoMaker)] <br>

[[🤗 Demo (Realistic[逼真])](https://huggingface.co/spaces/TencentARC/PhotoMaker)] &emsp; [[🤗 Demo (Stylization[风格化])](https://huggingface.co/spaces/TencentARC/PhotoMaker-Style)] <be>

如果 ID 保真度还不够，请试试我们的 [风格化应用](https://huggingface.co/spaces/TencentARC/PhotoMaker-Style), 你可能会有惊喜地发现。
</div>


---

官网实现 **[PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding](https://huggingface.co/papers/2312.04461)**.

### 🌠  **主要特性：**

1. 在**几秒**内快速定制，无需额外的 LoRA 训练。
2. 确保令人印象深刻的 ID 保真度，提供多样性，承诺文本可控性，并具有高质量生成。
3. 可以用作 **Adapter**，与社区中的其他基础模型和 LoRA 模块协同工作。

---

❗❗ 注意：如果有任何基于 PhotoMaker 的资源和应用，请在讨论中留言，我们会将它们列入 README 文件中的相关资源部分。目前我们已知实现了 **Replicate**、**Windows**、**ComfyUI** 和 **WebUI**。感谢大家!

<div align="center">

![photomaker_demo_fast](https://github.com/TencentARC/PhotoMaker/assets/21050959/e72cbf4d-938f-417d-b308-55e76a4bc5c8)
</div>


## 🚩 **新特性/更新**
- ✅ 2024 年 1 月 15 日。我们发布了 PhotoMaker。

---

## 🔥 **示例**


### Realistic generation 逼真生成

- [![Huggingface PhotoMaker](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/TencentARC/PhotoMaker)
- [**PhotoMaker notebook demo**](photomaker_demo.ipynb)

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6285a9133ab6642179158944/BYBZNyfmN4jBKBxxt4uxz.jpeg" height=450>
</p>

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6285a9133ab6642179158944/9KYqoDxfbNVLzVKZzSzwo.jpeg" height=450>
</p>

### Stylization generation 风格化生成

注意:仅更改基础模型并添加 LoRA 模块以获得更好的风格化效果

- [![Huggingface PhotoMaker-Style](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/TencentARC/PhotoMaker-Style)
- [**PhotoMaker-Style notebook demo**](photomaker_style_demo.ipynb) 

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6285a9133ab6642179158944/du884lcjpqqjnJIxpATM2.jpeg" height=450>
</p>
  
<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6285a9133ab6642179158944/-AC7Hr5YL4yW1zXGe_Izl.jpeg" height=450>
</p>

# 🔧 依赖和安装

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.0](https://pytorch.org/)
```bash
conda create --name photomaker python=3.10
conda activate photomaker
pip install -U pip

# Install requirements
pip install -r requirements.txt

# Install photomaker
pip install git+https://github.com/TencentARC/PhotoMaker.git
```

Then you can run the following command to use it
```python
from photomaker import PhotoMakerStableDiffusionXLPipeline
```

# ⏬ 下载模型 
The model will be automatically downloaded through the following two lines:

```python
from huggingface_hub import hf_hub_download
photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
```

You can also choose to download manually from this [url](https://huggingface.co/TencentARC/PhotoMaker).

# 💻 如何测试

## 参考 [diffusers](https://github.com/huggingface/diffusers)

- Dependency
```py
import torch
import os
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from photomaker import PhotoMakerStableDiffusionXLPipeline

### Load base model
pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model_path,  # can change to any base model based on SDXL
    torch_dtype=torch.bfloat16, 
    use_safetensors=True, 
    variant="fp16"
).to(device)

### Load PhotoMaker checkpoint
pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_path),
    subfolder="",
    weight_name=os.path.basename(photomaker_path),
    trigger_word="img"  # define the trigger word
)     

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

### Also can cooperate with other LoRA modules
# pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=lora_model_name, adapter_name="xl_more_art-full")
# pipe.set_adapters(["photomaker", "xl_more_art-full"], adapter_weights=[1.0, 0.5])

pipe.fuse_lora()
```

- Input ID Images
```py
### define the input ID images
input_folder_name = './examples/newton_man'
image_basename_list = os.listdir(input_folder_name)
image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])

input_id_images = []
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))
```

<div align="center">

<a href="https://github.com/TencentARC/PhotoMaker/assets/21050959/01d53dfa-7528-4f09-a1a5-96b349ae7800" align="center"><img style="margin:0;padding:0;" src="https://github.com/TencentARC/PhotoMaker/assets/21050959/01d53dfa-7528-4f09-a1a5-96b349ae7800"/></a>
</div>

- Generation
```py
# Note that the trigger word `img` must follow the class word for personalization
prompt = "a half-body portrait of a man img wearing the sunglasses in Iron man suit, best quality"
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth, grayscale"
generator = torch.Generator(device=device).manual_seed(42)
images = pipe(
    prompt=prompt,
    input_id_images=input_id_images,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=num_steps,
    start_merge_step=10,
    generator=generator,
).images[0]
gen_images.save('out_photomaker.png')
```

<div align="center">

<a href="https://github.com/TencentARC/PhotoMaker/assets/21050959/703c00e1-5e50-4c19-899e-25ee682d2c06" align="center"><img width=400 style="margin:0;padding:0;" src="https://github.com/TencentARC/PhotoMaker/assets/21050959/703c00e1-5e50-4c19-899e-25ee682d2c06"/></a>

</div>

## 启动本地 gradio demo
Run the following command:

```python
python gradio_demo/app.py
```

您可以在这个 [文件](gradio_demo/app.py) 中自定义该脚本。

如果您想在 MAC 上运行它，应按照此[说明](MacGPUEnv.md) 进行操作，然后运行 app.py。

## 使用提示：
- 上传更多要定制的人物的照片以提高 ID 保真度。如果输入是亚洲人脸，建议在类词之前添加“亚洲”，例如 `asian woman img`
- 进行风格化时，生成的人脸看起来太逼真了吗？将风格强度调整为 30-50，数字越大，ID 保真度越低，但风格化能力会更好。您还可以尝试具有良好风格化效果的其他基础模型或 LoRA。
- 为了更快的速度，请减少生成的图像数量和采样步数。但是，请注意，减少采样步数可能会损害 ID 保真度。

# 相关资源
### PhotoMaker 的 Replicate 演示: 
[Demo link](https://replicate.com/jd7h/photomaker) by [@yorickvP](https://github.com/yorickvP), transfer PhotoMaker to replicate.
### PhotoMaker 的 Windows版本:
1. [bmaltais/PhotoMaker](https://github.com/bmaltais/PhotoMaker/tree/v1.0.1) by [@bmaltais](https://github.com/bmaltais), easy to deploy PhotoMaker on Windows. The description can be found in [this link](https://github.com/TencentARC/PhotoMaker/discussions/36#discussioncomment-8156199).
2. [sdbds/PhotoMaker-for-windows](https://github.com/sdbds/PhotoMaker-for-windows/tree/windows) by [@sdbds](https://github.com/bmaltais).
### ComfyUI:
1. https://github.com/ZHO-ZHO-ZHO/ComfyUI-PhotoMaker
2. https://github.com/StartHua/Comfyui-Mine-PhotoMaker
3. https://github.com/shiimizu/ComfyUI-PhotoMaker

### Graido demo in 45 lines
Provided by [@Gradio](https://twitter.com/Gradio/status/1747683500495691942)

# 🤗 致谢
- PhotoMaker 由腾讯 ARC 实验室和南开大学 [MCG-NKU](https://mmcheng.net/cmm/) 共同主办。
- 受到许多优秀演示和仓库的启发，包括 [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [multimodalart/Ip-Adapter-FaceID](https://huggingface.co/spaces/multimodalart/Ip-Adapter-FaceID), [FastComposer](https://github.com/mit-han-lab/fastcomposer), 和 [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter). 感谢他们的伟大作品!
- 感谢腾讯 PCG 的 Venus 团队的反馈和建议。
- 感谢 HuggingFace 团队的慷慨支持! 

# 免责声明
本项目旨在对AI驱动的图像生成领域产生积极影响。用户有自由使用此工具创建图像的权利,但应遵守当地法律并以负责任的方式使用它。开发人员不对用户的潜在滥用承担任何责任。

# BibTeX
如果您发现PhotoMaker对研究和应用有用，请使用以下 BibTeX 引用：

```bibtex
@article{li2023photomaker,
  title={PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding},
  author={Li, Zhen and Cao, Mingdeng and Wang, Xintao and Qi, Zhongang and Cheng, Ming-Ming and Shan, Ying},
  booktitle={arXiv preprint arxiv:2312.04461},
  year={2023}
}
