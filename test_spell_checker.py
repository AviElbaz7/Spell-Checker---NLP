import re
from spelling_confusion_matrices import error_tables as confusion_matrices
from ex1 import Spell_Checker


def normalize_text(text):
    """
    Normalize the text: lowercase, remove punctuation, and keep only alphabetic characters.

    Args:
        text (str): The raw input text.
    Returns:
        str: Normalized text.
    """
    return re.sub(r'[^a-zA-Z\s]', '', text).lower()


def load_corpus():
    """
    Load and normalize the big.txt corpus.

    Returns:
        str: The normalized text from the corpus.
    """
    with open("big.txt", "r", encoding="utf-8") as f:
        raw = f.read()
    return normalize_text(raw)


def main():
    # Load and prepare language model
    corpus = load_corpus()
    lm = Spell_Checker.Language_Model(n=3)
    lm.build_model(corpus)

    # Initialize spell checker
    sc = Spell_Checker()
    sc.add_language_model(lm)
    sc.add_error_tables(confusion_matrices)

    # Input sentence
    sentence = "She defenitely has a good sence of humor"
    fixed = sc.spell_check(sentence, alpha=0.9)

    print(f"Original: {sentence}")
    print(f"Fixed   : {fixed}")


if __name__ == "__main__":
    main()
