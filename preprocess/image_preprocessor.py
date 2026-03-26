from PIL import Image, ImageOps


class ImagePreprocessor:
    def load_image(self, path: str):
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
