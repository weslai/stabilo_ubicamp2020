import Levenshtein as Lev
import torch

LETTER_DICT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
                'X': 23, 'Y': 24, 'Z': 25, 'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30, 'f': 31, 'g': 32, 'h': 33,
               'i': 34, 'j': 35, 'k': 36, 'l': 37, 'm': 38, 'n': 39, 'o': 40, 'p': 41, 'q': 42, 'r': 43, 's': 44,
               't': 45, 'u': 46, 'v': 47, 'w': 48, 'x': 49, 'y': 50, 'z': 51, '0': 52} ## '0' blank
KEY_LIST = list(LETTER_DICT.keys())
VALUE_LIST = list(LETTER_DICT.values())

def GreedyDecoder(matrix):
    ### matrix shape [batch, seq_len, num_class]
    # matrix = matrix[:, :, :-1] ## not included the blank 0 class 52
    idx_topk = torch.topk(matrix.cpu().detach(), k=1, dim=2)[1]
    seqs = torch.squeeze(idx_topk, dim=2)
    return seqs


def remove_dups_and_blanks(seq):
    res = [seq[0]]
    n = len(seq)
    for i in range(1,n):
        if seq[i] != res[-1]:
            res += [seq[i]]
    i = 0
    while i < len(res):
        if res[i] == 52:
            res.pop(i)    ### throw out the token
        else:
            i += 1
    return res


def post_process(seqs):
    seqs = [remove_dups_and_blanks(seq) for seq in seqs]
    return seqs

def convert_to_chars(indices):
    ## indices is a tensor in list
    ## convert the number to characters
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().detach()
    res = ''
    length = len(indices)
    num = 1
    for idx in indices: # list seq of classes
        if isinstance(idx, torch.Tensor):
            idx = int(idx.numpy())
        if idx != 52 and num != length: ## token 0
            res += KEY_LIST[VALUE_LIST.index(idx)]
            res += ' '
        elif idx != 52:
            res += KEY_LIST[VALUE_LIST.index(idx)]
        num += 1
    return res


def wer(s1, s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    print('s1:', s1)
    print('s2:', s2)
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]
    return Lev.distance(''.join(w1), ''.join(w2))

def cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Lev.distance(s1, s2)

if __name__ == '__main__':
    mat = torch.ones((4, 10, 5))
    mat[:, :5, 3] = 10
    mat[:, 6, 2] = 11
    mat[:, 8, 1] = 14
    seq = GreedyDecoder(mat)
    seq_out = post_process(seq)

    char = convert_to_chars(seq_out)
    s1 = "a b c d e f"
    s2 = "a h j d e f f"
    s3 = s1.split()
    s4 = s2.split()
    s5 = char.split()
    distance = wer(s1, s2)
    cer_error = cer(s1, s2)
    print(distance)