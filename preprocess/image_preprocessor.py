from PIL import Image

class ImagePreprocessor:
    def load_image(self, path: str):
        return Image.open(path).convert("RGB")