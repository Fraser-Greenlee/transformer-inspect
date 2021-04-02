"""Simple Math Sums"""

from __future__ import absolute_import, division, print_function

import random
from sys import maxsize
from typing import Any, Optional, List, Tuple
import operator

import datasets
from datasets.utils.filelock import logger
logger = logger()


_DESCRIPTION = """\
Dataset of simple math sums.
Data is generated based on the configuration given.
Used to inspect reasoning in transformers.

Formatted as a language modelling problem.
{
    'text': '3+2=5'
}
"""


random.seed(1)

OPERATOR_TO_STR = {
    operator.add: 'a',
    operator.sub: 's',
    operator.mul: 'm',
    operator.pow: 'p',
}


def subsets(a_list: list):
    for i in range(1, len(a_list)):
        yield a_list[:i]


CONFIG_PARAMS = []
for operators in subsets(list(OPERATOR_TO_STR.keys())):
    for a_range in [(0, 10), (-10, 10), (-100, 100, (-1000, 1000)), (-10_000, 10_000)]:
        input_range = a_range
        output_range = a_range
        for max_size in [1_000, 100_000, 1_000_000, 10_000_000]:
            CONFIG_PARAMS.append((
                operators,
                input_range,
                output_range,
                max_size
            ))


class MathConfig(datasets.BuilderConfig):
    """BuilderConfig for MathConfig."""

    def __init__(self, operators: Optional[List[Any]] = [operator.add], input_range: Tuple[int] = (0, 9), output_range: Tuple[int] = (0, 9), max_size: Optional[int] = 10_000, **kwargs):
        """BuilderConfig for MathConfig.
        Args:
          operators: list of python operator functions
          input_range: min & max input value
          output_range: min & max output value
          max_size: max number of elements allowed in the dataset
          **kwargs: keyword arguments forwarded to super.
        """
        self.operators = operators
        self.input_range = input_range
        self.output_range = output_range
        num_possibilities = len(operators) * (input_range[1] - input_range[0]) * (output_range[1] - output_range[0])
        self.max_size = min(max_size, num_possibilities)
        if self.max_size < max_size:
            logger.warn(f'Max size of dataset is {self.max_size}, smaller than the config size {max_size}, use more epochs to get more training steps.')
        super(MathConfig, self).__init__(**kwargs)


class Math(datasets.GeneratorBasedBuilder):
    """Small math sums."""

    previous_selections = set()
    BUILDER_CONFIGS = [
        MathConfig(
            name=f"{','.join([OPERATOR_TO_STR[op] for op in operators])}_{input_range[0]},{input_range[1]}_{output_range[0]},{output_range[1]}_{max_size}",
            description=f"Math problems using operators: {operator} with input range {input_range} and output_range {output_range} with max {max_size} rows.",
            operators=operators,
            input_range=input_range,
            output_range=output_range,
            max_size=max_size,
        )
        for operators, input_range, output_range, max_size in CONFIG_PARAMS
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'text': datasets.Value("string"),
                }
            ),
            homepage="https://github.com/Fraser-Greenlee/transformer-vae",
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION
            ),
        ]

    def _generate_examples(self):
        """Generate examples."""
        for id_ in range(self.config.max_size):
            in_a, in_b, operator = random.randint(*self.config.input_range), random.randint(*self.config.input_range), random.choice(self.config.operators)
            result = operator(in_a, in_b)
            yield id_, {'text': str(in_a) + ' ' + OPERATOR_TO_STR[operator] + f' {in_b} = {result}'}
