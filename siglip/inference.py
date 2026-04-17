import argparse
import joblib
import torch

from classifier_utils import MODEL_CHOICES, get_model_path
from config import CLASS_NAMES, MODEL_NAME, DEVICE
from feature_utils import get_normalized_image_features
from hf_utils import load_siglip_processor_and_model
from utils.image_loading import load_image_rgb


@torch.no_grad()
def embed_image(image_path, processor, model):
    image = load_image_rgb(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    feat = get_normalized_image_features(model, inputs)
    return feat.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="logreg",
        help="Classifier checkpoint to use on top of SigLIP embeddings.",
    )
    args = parser.parse_args()

    processor, model = load_siglip_processor_and_model(MODEL_NAME, DEVICE)

    clf = joblib.load(get_model_path(args.model))

    emb = embed_image(args.image, processor, model)
    pred = clf.predict(emb)[0]
    prob = clf.predict_proba(emb)[0, 1]

    label = CLASS_NAMES[pred]
    positive_class = CLASS_NAMES[1]
    print(f"Classifier: {args.model}")
    print(f"Prediction: {label}")
    print(f"{positive_class.capitalize()} probability: {prob:.4f}")


if __name__ == "__main__":
    main()
