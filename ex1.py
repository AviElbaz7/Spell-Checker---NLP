import collections
import math
import string
from nltk.metrics import edit_distance
import re


class Spell_Checker:
    """
    A context-sensitive spell checker using a noisy channel model and a language model.

    Attributes:
        lm (Language_Model): The language model used to evaluate and generate language.
        error_tables (dict): Confusion matrices for spelling errors.
        vocab (set): Set of all known vocabulary words from the language model.
    """

    def __init__(self, lm=None):
        """
        Initializes the spell checker with an optional language model.

        Args:
            lm (Language_Model, optional): An optional language model to initialize with.
        """
        self.lm = lm
        self.error_tables = {}
        self.vocab = set()

    def add_language_model(self, lm):
        """
        Adds or replaces the language model used by the spell checker.

        Args:
            lm (Language_Model): A language model object.
        """
        self.lm = lm
        model_dict = lm.get_model_dictionary()
        self.vocab = set()
        for ngram in model_dict:
            for token in ngram:
                self.vocab.add(token)

    def add_error_tables(self, error_tables):
        """
        Adds the spelling error confusion matrices.

        Args:
            error_tables (dict): Dictionary of confusion matrices.
        """
        self.error_tables = error_tables

    def evaluate_text(self, text):
        """
        Evaluates the log-likelihood of a given text based on the language model.

        Args:
            text (str): The input text.

        Returns:
            float: Log-probability of the sentence.
        """
        if not self.lm:
            return float('-inf')
        return self.lm.evaluate_text(text)

    def spell_check(self, text, alpha):
        """
        Corrects the input sentence using a context-sensitive spell checker.

        Args:
            text (str): Input sentence.
            alpha (float): Probability of keeping a word as-is.

        Returns:
            str: The corrected sentence.
        """
        text = text.lower()
        tokens = text.strip().split()
        best_sentence = tokens[:]
        original_score = self.evaluate_text(' '.join(tokens))

        for i, word in enumerate(tokens):
            candidates = self._generate_candidates(word)

            if candidates:
                min_dist = min(self._edit_distance(word, c) for c in candidates)
                candidates = [c for c in candidates if self._edit_distance(word, c) == min_dist]

            best_word = word
            best_score = float('-inf')

            for cand in candidates:
                if cand == word:
                    edit_prob = math.log(alpha)
                else:
                    edit_prob_raw = self._edit_probability(word, cand)
                    edit_prob = math.log((1 - alpha) * edit_prob_raw + 1e-12)

                new_tokens = tokens[:i] + [cand] + tokens[i + 1:]
                context_text = ' '.join(new_tokens)
                context_score = self.evaluate_text(context_text)

                unigram_bonus = self._unigram_score(cand)

                total_score = (
                        4.5 * context_score +
                        0.3 * unigram_bonus +
                        0.2 * edit_prob
                )

                if total_score > best_score:
                    best_score = total_score
                    best_word = cand
                elif abs(total_score - best_score) < 0.2:
                    if self._unigram_score(cand) > self._unigram_score(best_word):
                        best_score = total_score
                        best_word = cand

            best_sentence[i] = best_word

        result = ' '.join(best_sentence)
        return result

    def _generate_candidates(self, word):
        """
        Generates possible correction candidates for a given word.

        Args:
            word (str): The word to generate candidates for.

        Returns:
            list[str]: List of candidate words.
        """
        if word in self.vocab:
            return [word]
        edits1 = self._edits1(word)
        candidates = [w for w in edits1 if w in self.vocab]
        return candidates if candidates else [word]

    def _get_edit_operation(self, source, target):
        """
        Identifies the edit type and key between a source and target word.

        Args:
            source (str): The original word.
            target (str): The candidate word.

        Returns:
            Tuple[str, str]: The error type and key or (None, None).
        """
        if abs(len(source) - len(target)) > 1:
            return None, None

        if source == target:
            return None, None

        if len(target) == len(source) + 1:
            for i in range(len(target)):
                if source[:i] + target[i] + source[i:] == target:
                    key = source[i - 1] + target[i] if i > 0 else "#" + target[i]
                    return "insertion", key

        if len(source) == len(target) + 1:
            for i in range(len(source)):
                if source[:i] + source[i + 1:] == target:
                    key = source[i - 1] + source[i] if i > 0 else "#" + source[i]
                    return "deletion", key

        if len(source) == len(target):
            for i in range(len(source)):
                if source[i] != target[i]:
                    key = source[i] + target[i]
                    return "substitution", key

        for i in range(len(source) - 1):
            if (source[i] == target[i + 1] and
                    source[i + 1] == target[i] and
                    source[i + 2:] == target[i + 2:]):
                key = source[i] + source[i + 1]
                return "transposition", key

        return None, None

    def _get_edit_denominator(self, key, error_type):
        """
        Returns the total occurrences of the left-character (context)
        for normalizing the error probability.

        Args:
            key (str): The confusion matrix key.
            error_type (str): Type of edit.

        Returns:
            int: Sum of occurrences starting with the same context.
        """
        if not key:
            return 0

        context_char = key[0]
        return sum(count for k, count in self.error_tables.get(error_type, {}).items() if k.startswith(context_char))

    def _edit_probability(self, source, target):
        """
        Calculates the probability of an edit given source and target words.

        Args:
            source (str): Original word.
            target (str): Candidate word.

        Returns:
            float: Probability of the edit.
        """
        if source == target:
            return 1.0

        error_type, error_key = self._get_edit_operation(source, target)

        if error_type is None or error_key is None:
            return 1e-9

        numerator = self.error_tables.get(error_type, {}).get(error_key, 0)
        denominator = self._get_edit_denominator(error_key, error_type)
        total_keys = len(self.error_tables[error_type])

        return (numerator + 1) / (denominator + total_keys)

    def _unigram_bonus(self, word):
        """
        Returns log-probability bonus based on word frequency.

        Args:
            word (str): Word to score.

        Returns:
            float: Log-probability bonus.
        """
        count = self.lm.unigram_counts.get(word, 0)
        total = self.lm.total_unigrams
        return math.log(count + 1) - math.log(total + 1)

    def _unigram_score(self, word):
        """
        Returns unigram log-probability.

        Args:
            word (str): Word to score.

        Returns:
            float: Log-probability.
        """
        return self._unigram_bonus(word)

    @staticmethod
    def _edit_distance(w1, w2):
        """Returns edit distance between two words."""
        return edit_distance(w1, w2)

    @staticmethod
    def _edits1(word):
        """
        Generates all strings one edit away from the original word.

        Args:
            word (str): Input word.

        Returns:
            set[str]: Set of edited words.
        """
        letters = string.ascii_lowercase
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]

        return set(deletes + transposes + replaces + inserts)

    class Language_Model:
        """
        A Markov language model that supports n-gram evaluation and generation.
        """

        def __init__(self, n=3, chars=False):
            """
            Initializes the language model.

            Args:
                n (int): n-gram size.
                chars (bool): Whether to use character-level modeling.
            """
            self.n = n
            self.chars = chars
            self.model_dict = None
            self.unigram_counts = collections.defaultdict(int)
            self.total_unigrams = 0

        def build_model(self, text):
            """
            Builds an n-gram model from the input text.

            Args:
                text (str): The training corpus.
            """
            from collections import defaultdict
            self.model_dict = defaultdict(int)

            tokens = list(text) if self.chars else text.split()

            for token in tokens:
                self.unigram_counts[token] += 1
                self.total_unigrams += 1

            tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']

            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                self.model_dict[ngram] += 1

        def get_model_dictionary(self):
            """
            Returns the model's n-gram dictionary.

            Returns:
                dict: n-gram frequency dictionary.
            """
            return self.model_dict

        def get_model_window_size(self):
            """
            Returns the context window size (n).

            Returns:
                int: The n-gram size.
            """
            return self.n

        def evaluate_text(self, text):
            """
            Evaluates the log-likelihood of the input text.

            Args:
                text (str): Input text.

            Returns:
                float: Log-likelihood.
            """
            tokens = list(text) if self.chars else text.split()
            tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']

            log_prob = 0
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                log_prob += math.log(self.smooth(ngram))

            return log_prob

        def smooth(self, ngram):
            """
            Applies Laplace smoothing for a given n-gram.

            Args:
                ngram (tuple): An n-gram tuple.

            Returns:
                float: Smoothed probability.
            """
            prefix = ngram[:-1]
            model = self.model_dict
            vocab_size = len(set(k[-1] for k in model)) + 1
            count_ngram = model.get(ngram, 0)
            count_prefix = sum(v for k, v in model.items() if k[:-1] == prefix)
            return (count_ngram + 1) / (count_prefix + vocab_size)


def normalize_text(text):
    """
    Normalizes input text (removes punctuation and lowercases).

    Args:
        text (str): The input string.

    Returns:
        str: Normalized text.
    """
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower().strip()


