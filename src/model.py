from transformers import AutoModelForSequenceClassification, AutoConfig

def get_model(model_name_or_path: str, num_labels: int = 9):
    """Loads a pre-trained AutoModelForSequenceClassification with the target number of labels."""
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config
    )
    
    return model
