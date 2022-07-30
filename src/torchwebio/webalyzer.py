import functools
import io

from PIL import Image
from pywebio import start_server
from pywebio.input import file_upload, input_group
from pywebio.output import put_image, put_markdown, put_table
from torch import nn

from torchwebio.models.image.imageclassificationmodel import ImageClassificationModel


def image_class_webio(model: nn.Module, title, subtitle):
    img_model = ImageClassificationModel(model=model)
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
    title="Image Classification",
    subtitle="Calculates classsification labels on ImageNet",
):
    partial_webalyzer = functools.partial(image_class_webio, model, title, subtitle)
    start_server(partial_webalyzer, debug=True, port=8080)


if __name__ == "__main__":
    model = ImageClassificationModel(model_name="adv_inception_v3")._model
    webalyzer(model)
