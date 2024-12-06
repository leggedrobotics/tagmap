# Tag Map: A Text-Based Map for Spatial Reasoning and Navigation with Large Language Models

[Mike Zhang](https://mikez.xyz), [Kaixian Qu](https://www.linkedin.com/in/kaixian-qu-66a86215a), [Vaishakh Patil](https://www.linkedin.com/in/vaishakhpatil), [Cesar Cadena](https://n.ethz.ch/~cesarc), [Marco Hutter](https://rsl.ethz.ch/the-lab/people/person-detail.MTIxOTEx.TGlzdC8yNDQxLC0xNDI1MTk1NzM1.html)


[[Project Page](https://tag-mapping.github.io/)] [[Paper](https://arxiv.org/abs/2409.15451)]


![overview](https://tag-mapping.github.io/media/images/method_overview.svg)


### Abstract
Large Language Models (LLM) have emerged as a tool for robots to generate task plans using common sense reasoning. For the LLM to generate actionable plans, scene context must be provided, often through a map. Recent works have shifted from explicit maps with fixed semantic classes to implicit open vocabulary maps based on queryable embeddings capable of representing any semantic class. However, embeddings cannot directly report the scene context as they are implicit, requiring further processing for LLM integration. To address this, we propose an explicit text-based map that can represent thousands of semantic classes while easily integrating with LLMs due to their text-based nature by building upon large-scale image recognition models. We study how entities in our map can be localized and show through evaluations that our text-based map localizations perform comparably to those from open vocabulary maps while using two to four orders of magnitude less memory. Real-robot experiments demonstrate the grounding of an LLM with the text-based map to solve user tasks.


---
## Installation

Create a virtual environment.
```
virtualenv -p python3.8 <env name>
source <env name>/bin/activate
pip install --upgrade pip
```

Install torch
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

Install the image tagging model. Currently, this repo only supports the [Recognized Anything](https://github.com/xinyu1205/recognize-anything) set of image tagging models.
```
pip install -r thirdparty/recognize-anything/requirements.txt
pip install -e thirdparty/recognize-anything/.
```

Download image tagging model checkpoints
```
# Recognize Anything Model (RAM)
wget -P <path_to_checkpoint> https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth

# Recognize Anything Plus Model (RAM++)
wget -P <path_to_checkpoint> https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth
```


Install the `tag_mapping` package
```
pip install -r tag_mapping/requirements.txt
pip install -e tag_mapping/.
```


---
## Demos

Notebooks demonstrating the Tag Map construction and localization pipelines can be found in the `demos` folder.

---
## Evaluation
The `evaluation` folder contains instructions and scripts for evaluating the Tag Map localizations. 


---
## Citation
If you found our paper or code useful, please cite:
```
@inproceedings{zhang2024tagmap,
  author  = {Zhang, Mike and Qu, Kaixian and Patil, Vaishakh and Cadena, Cesar and Hutter, Marco},
  title   = {Tag Map: A Text-Based Map for Spatial Reasoning and Navigation with Large Language Models},
  journal = {Conference on Robot Learning (CoRL)},
  year    = {2024},
}
```