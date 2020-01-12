
from typing import List, Dict, Set, Tuple
from common import Instance, Sentence
from collections import defaultdict
from config.eval import Span
from config.reader import Reader
from config.utils import use_iobes
import random

def get_spans(output, type:str):
    """
    Remember the label needs to be IOBES encoding.
    :param output:
    :return:
    """
    output_spans = set()
    start = -1
    for i in range(len(output)):
        if output[i].startswith("B-") and type in output[i]:
            start = i
        if output[i].startswith("E-") and type in output[i]:
            end = i
            output_spans.add(Span(start, end, output[i][2:]))
        if output[i].startswith("S-") and type in output[i]:
            output_spans.add(Span(i, i, output[i][2:]))
    return output_spans

def extract_dictionary(insts: List[Instance], target_type:str, ratio:float = 0.2, ratio_for_new_type_data: float= 0.0)-> Tuple[List[Instance], List[Instance], List[str]]:
    """
    Remove the instances with new entity type
    Return the pruned instances
    Extract out the dictionary
    :param insts:
    :param target_type:
    :return: labeled instance, unlabeled instance, and the set of dictionary to be used
    """

    results = []
    unlabels = []
    ent2count = defaultdict(int)
    for inst in insts:
        output = inst.output
        words = inst.input.words
        has_new_type = False
        for label in output:
            if target_type in label:
                has_new_type = True

        # spans = get_spans(output=output, type=target_type)
        # for span in spans:
        #     ent2count[' '.join(words[span.left:span.right+1])] += 1
        if not has_new_type:
            results.append(inst)
        else:
            unlabels.append(inst)
    random.shuffle(unlabels)
    num_new_type_insts = int(len(unlabels) * ratio_for_new_type_data)
    results = results + unlabels[:num_new_type_insts]
    unlabels = unlabels[num_new_type_insts:]

    for inst in unlabels:
        output = inst.output
        spans = get_spans(output=output, type=target_type)
        words = inst.input.words
        for span in spans:
            ent2count[' '.join(words[span.left:span.right+1])] += 1

    print(f"number of remaining instances: {len(results)}")
    print(f"number of unique total entities: {len(ent2count)}")
    dictionary = sorted(ent2count.items(), key=lambda x: x[1], reverse=True)[:int(len(ent2count)*ratio)]
    dictionary = [x[0] for x in dictionary]
    print(f"number of entities in the dictionary: {len(dictionary)}")
    return results, unlabels, dictionary


def convert_dictionary_into_instances(dictionary: List[str], target_type: str) -> List[Instance]:

    insts = []
    for entity in dictionary:

        words = entity.split()
        output = []
        for i, word in enumerate(words):
            if i == 0:
                output.append(f"B-{target_type}")
            else:
                output.append(f"I-{target_type}")
        insts.append(Instance(Sentence(words), output))

    return insts
# reader = Reader(False)
# trains = reader.read_conll("data/conll2003/train.sd.conllx")
# use_iobes(trains)

# extract_dictionary(trains, target_type="PER")