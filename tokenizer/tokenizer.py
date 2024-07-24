from tqdm import tqdm
import re

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
    


class WordLevelTokenizer:
    def __init__(self):
        """Initialize the tokenizer with dictionaries for encoding and decoding."""
        self.word_to_index = {}
        self.index_to_word = {}
        self.next_index = 0

    def fit(self, corpus):
        """
        Build vocabulary from the given text corpus.

        Parameters:
        corpus (str): The entire text corpus to build the vocabulary.
        """
        words = self._preprocess_text(corpus)
        unique_words = set(words)
        for word in tqdm(unique_words, desc="Building Vocabulary"):
            if word not in self.word_to_index:
                self.word_to_index[word] = self.next_index
                self.index_to_word[self.next_index] = word
                self.next_index += 1

    def _preprocess_text(self, text):
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

    def tokenize(self, text):
        """
        Tokenize the given text at the word and punctuation level.

        Parameters:
        text (str): The text to tokenize.

        Returns:
        list: A list of integer indices representing each word and punctuation.
        """
        words = self._preprocess_text(text)
        return [self.word_to_index[word] for word in words if word in self.word_to_index]

    def detokenize(self, tokens):
        """
        Convert a list of tokens back into a string.

        Parameters:
        tokens (list): The list of tokens to detokenize.

        Returns:
        str: The combined string.
        """
        return ' '.join(self.index_to_word[token] for token in tokens)
