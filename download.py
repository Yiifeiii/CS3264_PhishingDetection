from huggingface_hub import hf_hub_download
import shutil

# Download detector
detector_path = hf_hub_download(
    repo_id="huzaifanasirrr/faceforge-detector",
    filename="detector_best.pth"
)


shutil.copy(detector_path, "models/detector_best.pth")