import argparse
import random
import numpy as np
from config import Reader, Config, ContextEmb, lr_decay, simple_batching, evaluate_batch_insts, get_optimizer, write_results, batching_list_instances
import time
from model.neuralcrf import NNCRF
import torch
from typing import List
from common import Instance
from termcolor import colored
import os
from config.utils import load_elmo_vec, get_metric
from data_utils import extract_dictionary, convert_dictionary_into_instances
import pickle
from collections import Counter, defaultdict
import tarfile

def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=True,
                        help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="conll2003")
    parser.add_argument('--embedding_file', type=str, default="data/glove.6B.100d.txt",
                        help="we will be using random embeddings if file do not exist")
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=0.01)  ##only for sgd now
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10, help="default batch size is 10 (works well)")
    parser.add_argument('--num_epochs', type=int, default=100, help="Usually we set to 10.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")

    ##model hyperparameter
    parser.add_argument('--model_folder', type=str, default="english_model", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--context_emb', type=str, default="none", choices=["none", "elmo"],
                        help="contextual word embedding")

    ## arguments for extracting
    parser.add_argument('--extraction', type=int, default=0, choices=[0,1], help="whether the mode for extracting entities for dictionary")
    parser.add_argument('--target_type', type=str, default="ORG", help="target entity type which is the new entity type")
    parser.add_argument('--dict_ratio', type=float, default=0.0, help="The ratio of extracting dictionary from the training data")
    parser.add_argument('--inst_ratio', type=float, default=0.2, help="The ratio of extracting dictionary from the training data")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train_model(config: Config, epoch: int, train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance], unlabeled_insts: List[Instance] = None):
    model:NNCRF = NNCRF(config)
    optimizer = get_optimizer(config, model)
    train_num = len(train_insts)
    print("number of instances: %d" % (train_num))
    print(colored("[Shuffled] Shuffle the training instance ids", "red"))
    random.shuffle(train_insts)

    batched_data = batching_list_instances(config, train_insts)
    batched_unlabeled = batching_list_instances(config, unlabeled_insts) if unlabeled_insts else None
    dev_batches = batching_list_instances(config, dev_insts)
    test_batches = batching_list_instances(config, test_insts)

    best_dev = [-1, 0]
    best_test = [-1, 0]

    model_folder = config.model_folder
    res_folder = "results"
    # if os.path.exists(model_folder):
    #     raise FileExistsError(f"The folder {model_folder} exists. Please either delete it or create a new one "
    #                           f"to avoid override.")
    model_name = f"model_files/{model_folder}/lstm_crf.m"
    config_name = f"model_files/{model_folder}/config.conf"
    res_name = res_folder + "/lstm_crf.results".format()
    print("[Info] The model will be saved to: %s.tar.gz" % (model_folder))
    if not os.path.exists("model_files"):
        os.makedirs("model_files")
    if not os.path.exists(f"model_files/{model_folder}"):
        os.makedirs(f"model_files/{model_folder}")
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    for i in range(1, epoch + 1):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        if config.optimizer.lower() == "sgd":
            optimizer = lr_decay(config, optimizer, i)

        labeled_data_batch_idxs = np.random.permutation(len(batched_data))
        unlabeled_data_batch_idxs = np.random.permutation(len(batched_unlabeled)) if unlabeled_insts else []
        final_idxs = []
        k = 0
        j = 0
        while len(final_idxs) < len(labeled_data_batch_idxs) + len(unlabeled_data_batch_idxs):
            if k < len(labeled_data_batch_idxs):
                final_idxs.append((1, labeled_data_batch_idxs[k]))
            if j < len(unlabeled_data_batch_idxs):
                final_idxs.append((0, unlabeled_data_batch_idxs[j]))
            k+=1
            j+=1

        for is_labeled, index in final_idxs:
            model.train()
            if not is_labeled:
                _, batch_max_ids = model.decode(batched_data[index])
                words, word_len, context_emb, char_seq, char_seq_len, _ = batched_data[index]
                batch_size = words.size(0)
                for batch_idx in range(batch_size):
                    batch_max_ids[batch_idx, :word_len[batch_idx]] = batch_max_ids[batch_idx, :word_len[batch_idx]].flip(0)
                for type in ["forward", "backward", "future", "past"]:
                    loss = model(words, word_len, context_emb, char_seq, char_seq_len, batch_max_ids, forward_type = type)
                    epoch_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    model.zero_grad()
            else:
                loss = model(*batched_data[index])
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                model.zero_grad()

        end_time = time.time()
        print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)

        model.eval()
        dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
        test_metrics = evaluate_model(config, model, test_batches, "test", test_insts)
        if dev_metrics[2] > best_dev[0]:
            print("saving the best model...")
            best_dev[0] = dev_metrics[2]
            best_dev[1] = i
            best_test[0] = test_metrics[2]
            best_test[1] = i
            torch.save(model.state_dict(), model_name)
            # Save the corresponding config as well.
            f = open(config_name, 'wb')
            pickle.dump(config, f)
            f.close()
            write_results(res_name, test_insts)
        model.zero_grad()

    # print("Archiving the best Model...")
    # with tarfile.open(f"model_files/{model_folder}/{model_folder}.tar.gz", "w:gz") as tar:
    #     tar.add(model_folder, arcname=os.path.basename(model_folder))
    # print("Finished archiving the models")

    print("The best dev: %.2f" % (best_dev[0]))
    print("The corresponding test: %.2f" % (best_test[0]))
    print("Final testing.")
    model.load_state_dict(torch.load(model_name))
    model.eval()
    evaluate_model(config, model, test_batches, "test", test_insts)
    write_results(res_name, test_insts)


def evaluate_model(config: Config, model: NNCRF, batch_insts_ids, name: str, insts: List[Instance]):
    ## evaluation
    p_dict, total_predict_dict, total_entity_dict = Counter(), Counter(), Counter()
    batch_id = 0
    batch_size = config.batch_size
    for batch in batch_insts_ids:
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batch_max_scores, batch_max_ids = model.decode(batch)
        batch_p , batch_predict, batch_total = evaluate_batch_insts(one_batch_insts, batch_max_ids, batch[-1], batch[1], config.idx2labels)
        p_dict += batch_p
        total_predict_dict += batch_predict
        total_entity_dict += batch_total
        batch_id += 1

    for key in sorted(total_entity_dict.keys()):
        precision_key, recall_key, fscore_key = get_metric(p_dict[key], total_entity_dict[key], total_predict_dict[key])
        print(f"[{key}] Prec.: {precision_key:.2f}, Rec.: {recall_key:.2f}, F1: {fscore_key:.2f}")

    total_p = sum(list(p_dict.values()))
    total_predict = sum(list(total_predict_dict.values()))
    total_entity = sum(list(total_entity_dict.values()))
    precision, recall, fscore = get_metric(total_p, total_entity, total_predict)
    print(f"[Total] Prec.: {precision:.2f}, Rec.: {recall:.2f}, F1: {fscore:.2f}")
    return [precision, recall, fscore]


def main():
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    conf = Config(opt)

    reader = Reader(conf.digit2zero)
    set_seed(opt, conf.seed)

    trains = reader.read_conll(conf.train_file, conf.train_num)
    devs = reader.read_conll(conf.dev_file, conf.dev_num)
    tests = reader.read_conll(conf.test_file, conf.test_num)

    if conf.context_emb != ContextEmb.none:
        print('Loading the ELMo vectors for all datasets.')
        conf.context_emb_size = load_elmo_vec(conf.train_file + "." + conf.context_emb.name + ".vec", trains)
        load_elmo_vec(conf.dev_file + "." + conf.context_emb.name + ".vec", devs)
        load_elmo_vec(conf.test_file + "." + conf.context_emb.name + ".vec", tests)

    conf.use_iobes(trains + devs + tests)
    unlabeled_insts = []
    if conf.extraction:
        labeled_insts, unlabeled_insts, ent_dict = extract_dictionary(trains, target_type=conf.target_type, ratio=conf.dict_ratio, ratio_for_new_type_data=conf.inst_ratio)
        dict_insts = convert_dictionary_into_instances(ent_dict, conf.target_type)
        conf.use_iobes(dict_insts)
        trains = labeled_insts + dict_insts

    conf.build_label_idx(trains + devs + tests + unlabeled_insts)
    conf.build_word_idx(trains + devs + tests + unlabeled_insts)
    conf.build_emb_table()
    conf.map_insts_ids(trains + devs + tests + unlabeled_insts)
    print("num chars: " + str(conf.num_char))
    # print(str(config.char2idx))

    print("num words: " + str(len(conf.word2idx)))
    # print(config.word2idx)

    if conf.extraction:
        train_model(conf, conf.num_epochs, trains, devs, tests, unlabeled_insts=unlabeled_insts)

    else:
        train_model(conf, conf.num_epochs, trains, devs, tests)


if __name__ == "__main__":
    main()
