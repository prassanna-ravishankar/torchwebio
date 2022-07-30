import tempfile
import urllib

from PIL import Image

from torchwebio.models.image.imageclassificationmodel import (
    MODEL_NAMES,
    ImageClassificationModel,
)


def get_sample_img(filename):
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        filename,
    )
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert("RGB")
    return img


def infer_on_one_model(model_name="adv_inception_v3"):
    with tempfile.NamedTemporaryFile() as named_f:
        my_model = ImageClassificationModel(model_name)
        results = my_model.process_img(get_sample_img(named_f.name))
    assert len(results) == my_model._number_of_results

    assert type(results) is list
    for res in results:
        assert type(res) is tuple
        assert type(res[0]) is str
        assert type(res[1]) is float


def test_all_models(subtests):
    for model_info in MODEL_NAMES:
        model_name = model_info["model_name"]
        with subtests.test(model_name=model_name):
            infer_on_one_model(model_name)
