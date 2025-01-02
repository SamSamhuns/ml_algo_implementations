# Load required libs, name-to-language data and util funcs
import os
import glob
import string
import unicodedata
from io import open as io_open

import torch


# character to consider
ALL_LETTERS = string.ascii_letters + " .,;'"
# length of one hot char encodiing vector
N_LETTERS = len(ALL_LETTERS)


def unicode_to_ascii(string):
    """convert unicode string to plain ascii, removing any accents"""
    return ''.join(c for c in unicodedata.normalize('NFD', string)
                   if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS)


def file_list(path):
    return glob.glob(path)


def read_lines_in_file(filename):
    """get list of lines from file
    """
    lines = io_open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


def letter_to_index(letter):
    return ALL_LETTERS.find(letter)


def letter_to_tensor(letter):
    letter_idx = letter_to_index(letter)
    letter_tensor = torch.zeros((1, N_LETTERS), dtype=torch.float32)
    letter_tensor[0][letter_idx] = 1
    return letter_tensor


def line_to_tensor(line):
    line_tensor = torch.zeros((len(line), 1, N_LETTERS), dtype=torch.uint8)
    for i, char in enumerate(line):
        vec_i = letter_to_index(char)
        line_tensor[i][0][vec_i] = 1
    return line_tensor


def get_category_names_and_lines(data_txt_path="../data/names/*.txt"):
    # Load name-to-category/language data
    # category_lines: dictionary, a list of names per language
    # all_categories: list of category names
    category_names = []
    category_lines = {}

    # populate the categories
    for class_file_path in file_list(data_txt_path):
        category_name = os.path.basename(class_file_path).split('.')[
            0]  # language name
        category_lines[category_name] = read_lines_in_file(class_file_path)
        category_names.append(category_name)

    return category_names, category_lines
