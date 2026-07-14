from transformers import AutoModelForSequenceClassification, AutoConfig

def get_model(model_name_or_path: str, num_labels: int = 9, local_files_only: bool = False):
    """Loads a pre-trained AutoModelForSequenceClassification with the target number of labels."""
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        local_files_only=local_files_only
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        local_files_only=local_files_only
    )
    
    return model
