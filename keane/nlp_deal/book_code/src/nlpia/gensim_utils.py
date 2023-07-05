# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import (bytes, dict, int, list, object, range, str,  # noqa
    ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict,

# from gensim.models import KeyedVectors
from gensim import corpora
from gensim import utils

from nlpia.constants import logging
logger = logging.getLogger(__name__)


def tokens2ngrams(tokens, n=2):
    tokens = list(tokens)
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(' '.join(tokens[i:i + n]))
    return ngrams


def passthrough(*args, **kwargs):
    return args[0] if len(args) else list(kwargs.values())[0]


def return_false(*args, **kwargs):
    return False


def return_true(*args, **kwargs):
    return True


def noop(*args, **kwargs):
    pass


def return_none(*args, **kwargs):
    pass


def to_unicode(sorb, allow_eval=False):
    r"""Ensure that strings are unicode (UTF-8 encoded).

    Evaluate bytes literals that are sometimes accidentally created by str(b'whatever')

    >>> to_unicode(b'whatever')
    'whatever'
    >>> to_unicode(b'b"whatever"')
    'whatever'
    >>> to_unicode(repr(b'b"whatever"'))
    'whatever'
    >>> to_unicode(str(b'b"whatever"'))
    'whatever'
    >>> to_unicode(str(str(b'whatever')))
    'whatever'
    >>> to_unicode(bytes(u'whatever', 'utf-8'))
    'whatever'
    >>> to_unicode(b'u"whatever"')
    'whatever'
    >>> to_unicode(u'b"whatever"')
    'whatever'

    There seems to be a bug in python3 core:
    >>> str(b'whatever')  # user intended str.decode(b'whatever') (str coercion) rather than python code repr
    "b'whatever'"
    >>> repr(str(b'whatever'))
    '"b\'whatever\'"'
    >>> str(repr(str(b'whatever')))
    '"b\'whatever\'"'
    >>> repr(str(repr(str(b'whatever'))))
    '\'"b\\\'whatever\\\'"\''
    >>> repr(repr(b'whatever'))
    '"b\'whatever\'"'
    >>> str(str(b'whatever'))
    "b'whatever'"
    >>> str(repr(b'whatever'))
    "b'whatever'"
    """
    if sorb is None:
        return sorb
    if isinstance(sorb, bytes):
        sorb = sorb.decode()
    for i, s in enumerate(["b'", 'b"', "u'", 'u"']):
        if (sorb.startswith(s) and sorb.endswith(s[-1])):
            # print(i)
            return to_unicode(eval(sorb, {'__builtins__': None}, {}))
    return sorb


class TweetCorpus(corpora.TextCorpus):
    ignore_matcher = return_none   # compiled regular expression for token matches to skip/ignore
    num_grams = 2
    case_normalizer = str
    tokenizer = None
    mask = None

    def get_texts(self):
        """ Parse documents from a .txt file assuming 1 document per line, yielding lists of filtered tokens """
        with self.getstream() as text_stream:
            for i, line in enumerate(text_stream):
                line = to_unicode(line)
                line = (TweetCorpus.case_normalizer or passthrough)(line)
                # line = self.case_normalizer(line)
                if self.mask is not None and not self.mask[i]:
                    continue
                ngrams = []
                for ng in tokens2ngrams((TweetCorpus.tokenizer or str.split)(line), n=self.num_grams):
                    if self.ignore_matcher(ng):
                        continue
                    ngrams += [ng]
                if not (i % 1000):
                    print(line)
                    print(ngrams)
                yield ngrams

    def __len__(self):
        """ Enables `len(corpus)` """
        if 'length' not in self.__dict__:
            logger.info("Computing the number of lines in the corpus size (calculating number of documents)")
            self.length = sum(1 for doc in self.getstream())
        return self.length


class SMSCorpus(corpora.TextCorpus):
    ignore_matcher = return_none   # compiled regular expression for token matches to skip/ignore
    num_grams = 2
    case_normalizer = utils.to_unicode
    tokenizer = str.split
    mask = None

    def get_texts(self):
        """ Parse documents from a .txt file assuming 1 document per line, yielding lists of filtered tokens """
        with self.getstream() as text_stream:
            for i, line in enumerate(text_stream):
                line = SMSCorpus.case_normalizer(line)
                if self.mask is not None and not self.mask[i]:
                    continue
                ngrams = []
                for ng in tokens2ngrams(self.tokenizer(line)):
                    if SMSCorpus.ignore_matcher(ng):
                        continue
                    ngrams += [ng]
                yield ngrams

    def __len__(self):
        """ Enables `len(corpus)` """
        if 'length' not in self.__dict__:
            logger.info("Computing the number of lines in the corpus size (calculating number of documents)")
            self.length = sum(1 for doc in self.getstream())
        return self.length
