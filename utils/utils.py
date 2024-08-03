import re

def get_vocab_size(corpus):
    """
    Get the vocabulary size of the given corpus.

    Parameters:
    corpus (str): The text corpus to analyze.

    Returns:
    int: The size of the vocabulary (number of unique words and punctuation).
    """
    words = preprocess_text(corpus)
    unique_words = set(words)
    return len(unique_words)

def preprocess_text(text):
    """
    Preprocess the text by converting to lowercase and splitting into words and punctuation.

    Parameters:
    text (str): The text to preprocess.

    Returns:
    list: A list of words and punctuation.
    """
    text = text.lower()
    words = re.findall(r'\b\w+\b|[^\w\s]', text)
    return words




def model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    formatted_total_params = "{:,}".format(total_params)
    print(f"Total parameters: {formatted_total_params}")


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    formatted_trainable_params = "{:,}".format(trainable_params)
    print(f"Trainable parameters: {formatted_trainable_params}")


    non_trainable_params = total_params - trainable_params
    formatted_non_trainable_params = "{:,}".format(non_trainable_params)
    print(f"Non-trainable parameters: {formatted_non_trainable_params}")