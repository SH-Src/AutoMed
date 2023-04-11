from collections import namedtuple

Genotype = namedtuple('Genotype', 'time, ehr, fuse, select')

PRIMITIVES = [
    'identity',
    'conv',
    'attention',
    'rnn',
    'ffn',
    'zero'
]


