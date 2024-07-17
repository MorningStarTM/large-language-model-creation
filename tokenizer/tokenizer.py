



class LetterLevelTokenizer:
    def __init__(self):
        """Initialize the tokenizer with dictionaries for encoding and decoding."""
        self.char_to_index = {}
        self.index_to_char = {}
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary of unique characters and their corresponding indices."""
        # Define the characters you want to include in your tokenizer
        characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
        
        # Create a mapping from character to index and index to character
        self.char_to_index = {char: idx for idx, char in enumerate(characters)}
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}

    def tokenize(self, text):
        """
        Tokenize the given text at the letter level.

        Parameters:
        text (str): The text to tokenize.

        Returns:
        list: A list of integer indices representing each character.
        """
        return [self.char_to_index[char] for char in text if char in self.char_to_index]

    def detokenize(self, tokens):
        """
        Convert a list of tokens back into a string.

        Parameters:
        tokens (list): The list of tokens to detokenize.

        Returns:
        str: The combined string.
        """
        return ''.join(self.index_to_char[token] for token in tokens)