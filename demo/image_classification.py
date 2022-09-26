from torchwebio.models.image.imageclassificationmodel import ImageClassificationModel
from torchwebio.webalyzer import webalyzer

if __name__ == "__main__":
    model = ImageClassificationModel(model_name="tf_efficientnet_b0_ap")._model
    webalyzer(model)
