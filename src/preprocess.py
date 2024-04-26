import re
import string
import nltk
from argparse import ArgumentParser
from tqdm import tqdm


def remove_strings(text):
    pool = [
        "Thermo Fisher Scientific",
        "Sigma-Aldrich",
        "Bio-Rad Laboratories",
        "Millipore Sigma",
        "Qiagen",
        "Promega",
        "New England Biolabs",
        "Agilent Technologies",
        "Enzo Life Sciences",
        "Tocris Bioscience",
        "Santa Cruz Biotechnology",
        "Abcam",
        "Meridian Life Science",
        "Cayman Chemical",
        "Alfa Aesar",
        "BD Biosciences",
        "Zymo Research",
        "Molecular Probes",
        "Invitrogen",
        "Roche Applied Science",
        "EMD Millipore",
        "Worthington Biochemical Corporation",
        "VWR International",
        "MP Biomedicals",
        "GE Healthcare Life Sciences",
        "WTA2",
    ]

    for term in pool:
        if term.lower() in text.lower():
            text = re.sub(rf'{term}', '', text, flags=re.IGNORECASE)
    return text


def remove_files(text):
    text = re.sub(r"\s+", " ", text)
    words = []
    for word in text.split(" "):
        if (not re.search(r'[^\.]+\.[a-zA-Z]+', word)) and (not re.search(r'[^\.]+\.[a-zA-Z]+\.[a-zA-Z]+', word)):
            words.append(word)
    text_without_filename = " ".join(words)
    return text_without_filename


def remove_url(text):
    """
    Remove url from text
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text_without_urls = url_pattern.sub('', text)

    return text_without_urls


def remove_unencoded_text(text):
    """
    Removes characters that are not UTF-8 encodable.
    """
    return "".join([i if (32 <= ord(i) < 128) else "" for i in text])


def contains_numbers(text):
    """
    Parses text using a regular expression and returns a boolean value
    designating whether that string contains any numbers.
    """
    return bool(re.search(r"^\d+$", text))


def is_allowed_word(word, stopwords, remove_numbers, min_word_len, max_word_len):
    """
    Checks if word is allowed based on inclusion in stopwords, presence of
    numbers, and length.
    """
    stopwords_allowed = word not in stopwords
    numbers_allowed = not (remove_numbers and contains_numbers(word))
    length_allowed = min_word_len <= len(word) <= max_word_len
    return stopwords_allowed and numbers_allowed and length_allowed


def preprocess(text, stopwords=set(nltk.corpus.stopwords.words("english")),
               stem=False, lemmatize=True, keep_alt_forms=False,
               remove_numbers=True, min_word_len=2, max_word_len=20):
    '''
    Standardized preprocessing of a line of text. Made by Anna Yannakopoulos
    2020. Added by NTH on 21 Jan 2020.
    '''

    # remove non utf-8 characters
    text = remove_unencoded_text(text)

    # remove predefined string
    text = remove_strings(text)

    # remove url
    text = remove_url(text)

    # remove file names
    text = remove_files(text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # lowercase the input
    text = text.lower()

    # convert all whitespace to spaces for splitting
    whitespace_pattern = re.compile(r"\s+")
    text = re.sub(whitespace_pattern, " ", text)

    # split into words
    words = text.split(" ")

    # stem and/or lemmatize words
    # filtering stopwords, numbers, and word lengths as required
    stemmer = nltk.stem.porter.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    if stem and lemmatize:
        words = [
            [word, stemmer.stem(word), lemmatizer.lemmatize(word)]
            for word in words if is_allowed_word(
                word, stopwords, remove_numbers, min_word_len, max_word_len)]
    elif stem:
        words = [
            [word, stemmer.stem(word)]
            for word in words if is_allowed_word(
                word, stopwords, remove_numbers, min_word_len, max_word_len)]
    elif lemmatize:
        words = [
            [word, lemmatizer.lemmatize(word)]
            for word in words if is_allowed_word(
                word, stopwords, remove_numbers, min_word_len, max_word_len)]
    else:
        words = [
            word for word in words if is_allowed_word(
                word, stopwords, remove_numbers, min_word_len, max_word_len)]

    if len(words) > 0:
        if stem or lemmatize:
            if keep_alt_forms:
                # return both original and stemmed/lemmatized words
                # as long as stems/lemmas are unique
                words = [w for word in words for w in set(word)]
            else:
                # return only requested stems/lemmas
                # if both stemming and lemmatizing, return only lemmas
                words = list(zip(*words))[-1]

        # remove processed words which length shorter than min_word_len
        words = [word for word in words if min_word_len <= len(word) <= max_word_len]

        return " ".join(words)
    else:
        return ""


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-input",
                        help="input text, one instance per line",
                        required=True,
                        type=str)
    parser.add_argument("-out",
                        help="processed text",
                        required=True,
                        type=str)
    args = parser.parse_args()

    # preprocess
    processed = []
    with open(args.input) as f:
        lines = f.readlines()
        for line in tqdm(lines, total=len(lines), desc='preprocessing text'):
            processed.append(preprocess(line.rstrip(), lemmatize=False))

    # output
    with open(args.out, 'w') as f:
        for i in processed:
            f.write(f'{i}\n')
