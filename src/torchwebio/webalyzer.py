import functools
import io
from enum import IntEnum

from PIL import Image
from pywebio import start_server
from pywebio.input import file_upload, input_group
from pywebio.output import put_image, put_markdown, put_table
from torch import nn

from torchwebio.exceptions import ComingSoonException
from torchwebio.models.image.imageclassificationmodel import ImageClassificationModel


class Model_Types(IntEnum):
    IMAGE_MODEL = 1
    NLP_MODEL = 2


Unsupported = [Model_Types.NLP_MODEL]


def image_class_webio(model: nn.Module, title, subtitle, category_list=None):
    """
    Implementation of a PyWebIO application for image classification models.

    Accepts an image and outputs a table of classification scores.

    Parameters
    ----------
    model : nn.Module
        Pytorch model that needs to do the prediction
    title : _type_
        Title for the application
    subtitle : _type_
        Subtitle for the application
    category_list : _type_, optional
        URL to the category list to map categories
        indices to labels, by default None
    """
    img_model = ImageClassificationModel(model=model, category_list=category_list)
    put_markdown(f"# {title}")
    put_markdown(f"{subtitle}")

    uploaded = input_group(
        "Upload Image",
        [
            file_upload(
                # label="Image",
                accept="image/*",
                placeholder="Choose file",
                name="img",
            ),
        ],
    )

    if uploaded:
        image_data = uploaded["img"]["content"]
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        results = img_model.process_img(image)
        put_markdown("## Results")
        put_markdown("### Uploaded Image")
        put_image(image)
        put_markdown("### Labels")
        put_markdown(f"You uploaded an image of {results[0][0]}")
        put_table(results, header=["Label", "Score"])
    pass


def webalyzer(
    model: nn.Module,
    title: str = "Image Classification",
    subtitle: str = "Calculates classsification labels on ImageNet",
    model_type: Model_Types = Model_Types.IMAGE_MODEL,
    class_category_list: str = None,
):
    """
    Function to generate an application based on the model.
    Currently supports timm-like Image classification models
    Coming soon: NLP and more Computer vision model suppport

    Parameters
    ----------
    model : nn.Module
        Pytorch model to be turned into an application
    title : str, optional
        Title for the application, by default "Image Classification"
    subtitle : str, optional
        Subtitle or body of the application, by default
         "Calculates classsification labels on ImageNet"
    model_type : Model_Types, optional
        Type of the model, by default Model_Types.IMAGE_MODEL
    class_category_list : str, optional
        Mapping output vectors to class categories., by default None

    Raises
    ------
    ComingSoonException
        If a non ImageModel Model_Types is passed, an exception is raised
    """
    if model_type in Unsupported:
        raise ComingSoonException
    partial_webalyzer = functools.partial(
        image_class_webio, model, title, subtitle, class_category_list
    )
    start_server(partial_webalyzer, debug=True, port=8080)
