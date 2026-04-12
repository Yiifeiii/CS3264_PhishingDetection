def _load_component_local_first(component_cls, model_name, component_label):
    try:
        return component_cls.from_pretrained(model_name, local_files_only=True)
    except Exception:
        try:
            return component_cls.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load the SigLIP {component_label} for {model_name}. "
                "The files are not fully cached locally, and downloading from "
                "Hugging Face failed. Connect to the internet once and rerun."
            ) from exc


def load_siglip_processor_and_model(model_name, device):
    from transformers import AutoModel, AutoProcessor

    processor = _load_component_local_first(AutoProcessor, model_name, "processor")
    model = _load_component_local_first(AutoModel, model_name, "model").to(device)
    model.eval()
    return processor, model
