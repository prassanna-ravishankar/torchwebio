[![Coveralls](https://img.shields.io/coveralls/github/prassanna-ravishankar/torchwebio/main.svg)](https://coveralls.io/r/prassanna-ravishankar/torchwebio)
![Tests](https://github.com/prassanna-ravishankar/torchwebio/actions/workflows/ci.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/torchwebio/badge/?version=latest)](https://torchwebio.readthedocs.io/en/latest/?badge=latest)


# torchwebio

> Yet another model to web app generator

torchwebio is a simple package that turns your pytorch model into a web application.

## Why?

Researchers often train models or fine tune models using pytorch and use dashboards like tensorboard to track the progress of the training routine.

These dashboards often give you metrics like loss and accuracies and visualizing some sample images. But they don't help you visualize the actual real-world performance of your model in-training.

Imagine you want to test your model against some other sample images that you didn't plan before starting the training routine. Imagine you want to share your "latest-greatest-and-bestest" model with your manager. Imagine you wanted to continuously train your model and always share the latest-and-greatest with the world.

> That's where this package comes in.

Read the relevant blog post [here](#)

Find the documentation [here](https://torchwebio.readthedocs.io/en/latest/)

## How?

The code that powers this package is actually **very, very** simple. It is built upon the shoulders of giants: pywebio for app generation, and timm for Image classification models.

I am to be very opinionated, and support interfaces set out by the popular neural network libraries. This allows the `webalyzer` interface to be extremely simple, without needing to support all the "chaos" that exists out there.

| Problem type                            | Library      | Status      |
|-----------------------------------------|--------------|-------------|
| Image Classification                    | [timm](https://github.com/rwightman/pytorch-image-models)         | Implemented |
| Object detection, Instance segmentation | [detectron2](https://github.com/facebookresearch/detectron2)   | Coming Soon |
| NLP models                              | [transformers](https://huggingface.co/docs/transformers/index) | Coming soon |

Credits to [pywebio](https://www.pyweb.io/) for a super simple framework for python web UI generation.

## Install

`pip install torchwebio`


## Usage

### Simple visualization

```python
import timm
from torchwebio import webalyzer

# Load a TIMM-like model or a regular pytorch model
model = timm.create_model('tf_efficientnet_b0_apss', pretrained=True)

# ....
# Fine tune the model
# ....


# Launch the web UI
webalyzer(model)

```

### Auto-updating application (coming soon!)
```python
import timm
from torchwebio import webalyzer, updater

# Load a TIMM-like model or a regular pytorch model
model = timm.create_model('tf_efficientnet_b0_apss', pretrained=True)

webalyzer(model)

for idx, (data, labels) in enumerate(dataloader):
     # do some finetuning
     outputs = model(data)
     loss = criterion(outputs, labels)
     loss.backward()
     # ....

     # update the app every 1000 iterations
     if idx % 1000 == 0:
          updater(model)
```



## Contribute
Check out [CONTRIBUTING.md](./CONTRIBUTING.md). More to come!

## Help out

I don't do this a day job, but if you do want to help out, buy me a coffee. I promise to donate 100% of the proceeds on the account :)

More than the money, this give me a data point on people "actually" interested in this project

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/prass)
