from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pickle
from absl import flags, logging
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
import metrics
import argparse
import preprocess as pr
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--max_seq_length', type=int, default=128, help='train data source')
parser.add_argument('--data_dir', type=str, default='./data', help='train data source')
parser.add_argument('--process', type=str, default='train', help='train, eval or predict')
parser.add_argument('--middle_output', type=str, default='middle_data', help='Dir was used to store middle data!')
parser.add_argument('--bert_config_file', type=str, default='./cased_L-12_H-768_A-12/bert_config.json', help='The config json file corresponding to the pre-trained BERT model.')
parser.add_argument('--batch_size', type=int, default=1, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=1, help='#epoch of training')
parser.add_argument('--vocab_file', type=str, default='./cased_L-12_H-768_A-12/vocab.txt', help="The vocabulary file that the BERT model was trained on.")
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for.')
parser.add_argument('--save_checkpoints_steps', type=int, default=1000, help='How often to save the model checkpoint.')
parser.add_argument('--iterations_per_loop', type=int, default=1000, help='How many steps to make in each estimator call.')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--init_checkpoint', type=str, default='./cased_L-12_H-768_A-12/bert_model.ckpt', help='train/test/demo')
parser.add_argument('--output_dir', type=str, default='./output/result_dir', help="The output directory where the model checkpoints will be written.")
#传递参数送入模型中
args = parser.parse_args()
def hidden2tag(hiddenlayer, numclass):
    linear = tf.keras.layers.Dense(numclass, activation=None)
    return linear(hiddenlayer)


def crf_loss(logits, labels, mask, num_labels, mask2len):
    # TODO
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
            "transition",
            shape=[num_labels, num_labels],
            initializer=tf.contrib.layers.xavier_initializer()
        )

    log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(logits, labels, transition_params=trans,
                                                                   sequence_lengths=mask2len)
    loss = tf.math.reduce_mean(-log_likelihood)

    return loss, transition


def softmax_layer(logits, labels, num_labels, mask):
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask, dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12  # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    return loss, predict


def create_model(bert_config, is_training, input_ids, mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()
    # output_layer shape is
    if is_training:
        output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)
    logits = hidden2tag(output_layer, num_labels)
    # TODO test shape
    logits = tf.reshape(logits, [-1, args.max_seq_length, num_labels])
    if args.CRF:
        mask2len = tf.reduce_sum(mask, axis=1)
        loss, trans = crf_loss(logits, labels, mask, num_labels, mask2len)
        predict, viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
        return (loss, logits, predict)
    else:
        loss, predict = softmax_layer(logits, labels, num_labels, mask)

        return (loss, logits, predict)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        mask = features["mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, logits, predicts) = create_model(bert_config, is_training, input_ids,
                                                      mask, segment_ids, label_ids, num_labels,
                                                      use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        initialized_variable_names = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                         init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps,
                                                     False)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            predictions = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
            #label_ids = tf.reshape(label_ids,[128])
            #predictions = tf.reshape(predictions, [128])
            #mask = tf.reshape(mask, [128])
            acc = tf.metrics.accuracy(label_ids, predictions, weights=mask)
            rec = tf.metrics.recall(label_ids, predictions, weights=mask)
            pre = tf.metrics.precision(label_ids, predictions, weights=mask)
            eval_op={"acc" : acc,
                     "rec" : rec,
                     "pre" : pre}
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_op,
                scaffold=scaffold_fn)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predicts, scaffold=scaffold_fn
            )
        return output_spec
    return model_fn


def _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i):
    token = batch_tokens[i]
    predict = id2label[prediction]
    true_l = id2label[batch_labels[i]]
    if token != "[PAD]" and token != "[CLS]" and true_l != "X":
        if predict == "X" and not predict.startswith("##"):
            predict = "O"
        line = "{}\t{}\t{}\n".format(token, true_l, predict)
        wf.write(line)


def Writer(output_predict_file, result, batch_tokens, batch_labels, id2label):
    with open(output_predict_file, 'w') as wf:

        if args.CRF:
            predictions = []
            for m, pred in enumerate(result):
                predictions.extend(pred)
            for i, prediction in enumerate(predictions):
                _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i)

        else:
            for i, prediction in enumerate(result):
                _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i)


def main(_):
    logging.set_verbosity(logging.INFO)
    processors = {"ner": pr.NerProcessor}
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
    processor = processors['ner']()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=False)
    run_config = tf.estimator.RunConfig(
        model_dir=args.output_dir,
        save_checkpoints_steps=args.save_checkpoints_steps)
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if args.process=='train':
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.batch_size * args.epoch)
        num_warmup_steps = int(num_train_steps * args.warmup_proportion)
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=args.init_checkpoint,
        learning_rate=args.lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_one_hot_embeddings=False)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    if args.process=='train':
        train_file = os.path.join(args.output_dir, "train.tf_record")
        _, _ = pr.filed_based_convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, train_file)
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", args.batch_size)
        logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = pr.file_based_input_fn_builder(
            input_file=train_file,
            seq_length=args.max_seq_length,
            is_training=True,
            drop_remainder=True,
            batch_size=args.batch_size)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    if args.process=="eval":
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_file = os.path.join(args.output_dir, "eval.tf_record")
        batch_tokens, batch_labels = pr.filed_based_convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, eval_file)

        logging.info("***** Running evaluation *****")
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", args.batch_size)
        eval_input_fn = pr.file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=args.max_seq_length,
            is_training=False,
            drop_remainder=False,
            batch_size=args.batch_size)
        result = estimator.evaluate(input_fn=eval_input_fn)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as wf:
            logging.info("***** Eval results *****")
            f = result["acc"]
            r = result["rec"]
            p = result["pre"]
            logging.info("***********************************************")
            logging.info("********************P = %s*********************", str(p))
            logging.info("********************R = %s*********************", str(r))
            logging.info("********************F = %s*********************", str(f))
            logging.info("***********************************************")
    if args.process=="predict":
        with open(args.middle_output + '/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(args.data_dir)

        predict_file = os.path.join(args.output_dir, "predict.tf_record")
        batch_tokens, batch_labels = pr.filed_based_convert_examples_to_features(predict_examples, label_list,
                                                                              args.max_seq_length, tokenizer,
                                                                              predict_file)

        logging.info("***** Running prediction*****")
        logging.info("  Num examples = %d", len(predict_examples))
        logging.info("  Batch size = %d", args.batch_size)

        predict_input_fn = pr.file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=args.max_seq_length,
            is_training=False,
            drop_remainder=False,
            batch_size=args.batch_size)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(args.output_dir, "label_test.txt")
        Writer(output_predict_file, result, batch_tokens, batch_labels, id2label)


if __name__ == "__main__":
    tf.app.run()
