import torch


def get_normalized_image_features(model, inputs):
    image_features = model.get_image_features(**inputs)

    if not torch.is_tensor(image_features):
        if hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
            image_features = image_features.pooler_output
        elif hasattr(image_features, "image_embeds") and image_features.image_embeds is not None:
            image_features = image_features.image_embeds
        elif isinstance(image_features, (tuple, list)) and image_features:
            image_features = image_features[0]
        else:
            raise TypeError(
                f"Unsupported image feature output type: {type(image_features)}"
            )

    return torch.nn.functional.normalize(image_features, dim=-1)
