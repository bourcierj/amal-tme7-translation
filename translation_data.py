"""Utilities for handling GSD dataset from datamaestro"""

from pathlib import Path
import unicodedata

import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TranslationVocabulary:
    """Helper class to manage a vocabulary.
    Does not handle out-of-vocabulary (OOV) words. (maybe future #@todo?)
    """

    PAD_ID = 0  # not a word (padding) code
    SOS_ID = 1  # start-of-sequence code
    EOS_ID = 2  # end-of-sequence code
    # OOV_ID = 3  # out of vocabulary code

    def __init__(self):

        self.word2id = {'<pad>': self.PAD_ID, '<eos>': self.EOS_ID, '<sos>': self.SOS_ID}
        self.id2word = {self.PAD_ID: '', self.EOS_ID: '<eos>', self.SOS_ID: '<sos>'}
        self.chars = set()  # set of all encountered characters

    def get(self, word: str, adding=False):
        """Maps a word to its id.
        Args:
            word (str): word for which to get the id
            adding (bool): if True, adds word to vocabulary if it was not
                encountered before
        """
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word[wordid] = word
                self.chars.update(word)
                return wordid
            raise

    def get_sentence(self, sen, adding=False):
        """Encodes a sentence.
        Args:
            sen (list): a list of string tokens
            adding (bool): if True, adds unseen words to vocabulary
        Return:
            torch.Tensor: a tensor of ids
        """
        return torch.LongTensor([self.SOS_ID]+[self.get(w, adding) for w in sen]
                                +[self.EOS_ID])

    def __len__(self):
        return len(self.word2id)

    def decode(self, sen):
        """Decodes a sentence.
        Args:
            sen(list): a list of Tensor of ids
        Returns:
            list: a list of string tokens
        """
        if isinstance(sen, torch.Tensor):
            sen = sen.tolist()
        return [self.id2word[w] for w in sen]


def normalize(text: str):
    """Normalizes text as unicode characters"""
    #@todo: fix '\u202' characters showing up in some french sentences.

    return ''.join(c for c in unicodedata.normalize('NFD', text.lower().strip().replace(u'\u202f', ' ')))

def tokenize_on_space(text: str, lang=None):
    """Tokenizes text simply by separating on spaces"""
    return text.split(' ')

def tokenize_spacy(text: str, spacy_lang='en'):
    """Tokenizes text using a spaCy model"""
    return [token.text for token in spacy_lang.tokenizer(text)]


def process(datapath, tokenize, max_len):
    """Pre-processes the Anki dataset of sentence pairs (https://www.manythings.org/anki/).
    The steps:
        - Get origin and destination sentences from every line in the file;
        - Normalize characters as unicode;
        - Tokenize each sentence with language-specific tokenization from spaCy.
        - Get encoded sentences from vocabs.
    Args:
        datapath (path-like): path to dataset file
        max_len (int): filter pairs by the maximum length of the origin sentence. Short
            sentences are easier to translate.
    Returns:
        list: list of tokenized pairs of sentences
    """
    if tokenize == tokenize_spacy:  # load spaCy models
        orig_lang = spacy.load('fr')
        dest_lang = spacy.load('en')
    else:
        orig_lang = dest_lang = None

    sentences = list()
    with open(datapath, 'r') as fp:
        for line in fp:
            if len(line) < 1:
                continue
            # get sentences and normalize them
            orig, dest = map(normalize, line.split('\t')[:2])
            if len(orig) > max_len:
                continue
            # tokenize sentences
            orig, dest = map(tokenize, (orig, dest), (orig_lang, dest_lang))
            sentences.append((orig, dest))
    return sentences


class TranslationDataset(Dataset):
    """Dataset for Anki Translation sentence pairs (https://www.manythings.org/anki/)"""
    #@todo:
    # Later, when we'll need to use spaCy models, maybe save processed data to avoid
    # loading models again.
    def __init__(self, datapath, vocab_orig, vocab_dest, tokenize=tokenize_on_space,
                 max_len=10):

        datapath = Path(datapath)
        processed = process(datapath, tokenize, max_len)
        # build vocabs and numericalize
        sentences = [(vocab_orig.get_sentence(orig, adding=True),
                      vocab_dest.get_sentence(dest, adding=True))
                     for orig, dest in processed]
        self.sentences = sentences

    def __getitem__(self, idx):
        return self.sentences[idx]

    def __len__(self):
        return len(self.sentences)

    @staticmethod
    def collate(batch):
        """Collate function (pass to DataLoader's collate_fn arg).
        Args:
            batch (list): list of examples returned by __getitem__
        Returns:
            tuple: Three tensors: batch of padded origin sequences, lengths of origin
                sequences, and batch of padded destination sequences (the targets).
        """
        data, target = list(zip(*batch))
        lengths = torch.LongTensor([len(s) for s in data])
        return (pad_sequence(data), lengths, pad_sequence(target))


def get_dataloader_and_vocabs(batch_size, tokenize, max_len):

    orig_vocab = TranslationVocabulary()
    dest_vocab = TranslationVocabulary()

    #@todo: Here we cheat for simplicity: add all the words to the vocabularies in either train,
    # test or val set.
    dataset = TranslationDataset('./data/fra.txt', orig_vocab, dest_vocab, tokenize,
                                 max_len)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=dataset.collate,
                        pin_memory=torch.cuda.is_available(),
                        num_workers=torch.multiprocessing.cpu_count())
    return loader, orig_vocab, dest_vocab

    # # partition data between train, test and validation set
    # dev_len = int(len(dataset)*0.7)
    # test_len = len(dataset) - dev_len  # use 30% for testing

    # dev_ds, test_ds = torch.utils.data.random_split(dataset, [dev_len, test_len])
    # train_len = int(len(dev_ds)*0.9)
    # val_len = len(dev_ds) - train_len
    # train_ds, val_ds = torch.utils.data.random_split(dev_ds, [train_len, val_len])

    # kwargs = dict(collate_fn=TranslationDataset.collate,
    #               pin_memory=(torch.cuda.is_available()),
    #               num_workers=torch.multiprocessing.cpu_count())

    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, **kwargs)
    # test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, **kwargs)

    # return train_loader, val_loader, test_loader, orig_vocab, dest_vocab


if __name__ == '__main__':

    # Test data processing, print some stats
    orig_vocab = TranslationVocabulary()
    dest_vocab = TranslationVocabulary()

    dataset = TranslationDataset('./data/fra.txt', orig_vocab, dest_vocab, max_len=10)
    print("Total number of sentences:", len(dataset))

    print("Origin vocabulary size:", len(orig_vocab))
    print("Destination vocabulary size:", len(dest_vocab))
    print("Characters:", ''.join(orig_vocab.chars | dest_vocab.chars))

    print("Train samples:")
    for i in range(10):
        data, target = dataset[i]
        print(f"  Input: {orig_vocab.decode(data)}")
        print(f"  Target: {dest_vocab.decode(target)}")

    # Test dataloading
    train_loader, orig_vocab, dest_vocab = \
        get_dataloader_and_vocabs(batch_size=16, tokenize=tokenize_on_space,
                                  max_len=10)

    print("Train batch:")
    data, lengths, target = next(iter(train_loader))
    print(f"Input {tuple(data.size())}:\n{data}")
    print(f"Lenghts:\n{lengths}")
    print(f"Target {tuple(target.size())}:\n{target}")
