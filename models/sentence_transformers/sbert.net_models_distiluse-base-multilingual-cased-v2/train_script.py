#CUDA_VISIBLE_DEVICES=1 python make_multilingual_mUSE.py parallel-sentences/*-train.tsv.gz monolingual-sentences/merged_sentences.txt.gz --dev parallel-sentences/*-dev.tsv.gz
from sentence_transformers import SentenceTransformer, LoggingHandler, models, evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import ParallelSentencesDataset
from datetime import datetime

import os
import logging
import gzip
import numpy as np
import sys
import zipfile
import io
from shutil import copyfile
import csv
import sys
import torch.multiprocessing as mp
from USEModel import USEModel
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import tensorflow as tf
from Normalize import Normalize

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    teacher_model_name = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"  # Our monolingual teacher model, we want to convert to multiple languages
    student_model_name = 'input_models/distilmuse-no-tatoeba'  # Multilingual base model we use to imitate the teacher model

    max_seq_length = 128  # Student model max. lengths for inputs (number of word pieces)
    train_batch_size = 64  # Batch size for training
    inference_batch_size = 64  # Batch size at inference
    max_sentences_per_language = 1000000  # Maximum number of  parallel sentences for training
    train_max_sentence_length = 250  # Maximum length (characters) for parallel training sentences

    num_epochs = 5  # Train for x epochs
    num_warmup_steps = 10000  # Warumup steps

    num_evaluation_steps = 50000  # Evaluate performance after every xxxx steps

    output_path = "output/make-multilingual-large-muse-" + teacher_model_name.split("/")[-2] + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Write self to path
    os.makedirs(output_path, exist_ok=True)

    train_script_path = os.path.join(output_path, 'train_script.py')
    copyfile(__file__, train_script_path)
    with open(train_script_path, 'a') as fOut:
        fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

    # Read passed arguments

    train_files = []
    dev_files = []
    is_dev_file = False
    for arg in sys.argv[1:]:
        if arg.lower() == '--dev':
            is_dev_file = True
        else:
            if not os.path.exists(arg):
                print("File could not be found:", arg)
                exit()

            if is_dev_file:
                dev_files.append(arg)
            else:
                train_files.append(arg)

    if len(train_files) == 0:
        print("Please pass at least some train files")
        print("python make_multilingual_sys.py file1.tsv.gz file2.tsv.gz --dev dev1.tsv.gz dev2.tsv.gz")
        exit()

    logging.info("Train files: {}".format(", ".join(train_files)))
    logging.info("Dev files: {}".format(", ".join(dev_files)))

    ######## Start the extension of the teacher model to multiple languages ########
    logging.info("Load teacher model")
    teacher_model = USEModel(model_name=teacher_model_name)


    logging.info("Create student model from scratch")
    #word_embedding_model = models.Transformer(student_model_name, max_seq_length=max_seq_length)
    #pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    #dan = models.Dense(in_features=768, out_features=512)
    #norm = Normalize()
    #student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dan, norm])

    student_model = SentenceTransformer(student_model_name)

    ###### Read Parallel Sentences Dataset ######
    train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model, batch_size=inference_batch_size, use_embedding_cache=False)
    for train_file in train_files:
        train_data.load_data(train_file, max_sentences=max_sentences_per_language, max_sentence_length=train_max_sentence_length)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size, num_workers=0)
    train_loss = losses.MSELoss(model=student_model)

    #### Evaluate cross-lingual performance on different tasks #####
    mse_evaluators = []  # evaluators has a list of different evaluator classes we call periodically
    trans_evaluator = []

    for dev_file in dev_files:
        logging.info("Create evaluator for " + dev_file)
        src_sentences = []
        trg_sentences = []
        with gzip.open(dev_file, 'rt', encoding='utf8') if dev_file.endswith('.gz') else open(dev_file, encoding='utf8') as fIn:
            for line in fIn:
                splits = line.strip().split('\t')
                if splits[0] != "" and splits[1] != "":
                    src_sentences.append(splits[0])
                    trg_sentences.append(splits[1])

        # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
        dev_mse = evaluation.MSEEvaluator(src_sentences, trg_sentences, name=os.path.basename(dev_file), teacher_model=teacher_model, batch_size=inference_batch_size)
        mse_evaluators.append(dev_mse)

        # TranslationEvaluator computes the embeddings for all parallel sentences. It then check if the embedding of source[i] is the closest to target[i] out of all available target sentences
        dev_trans_acc = evaluation.TranslationEvaluator(src_sentences, trg_sentences, name=os.path.basename(dev_file), batch_size=inference_batch_size)
        trans_evaluator.append(dev_trans_acc)

    # Open the ZIP File of STS2017-extended.zip and check for which language combinations we have STS data
    sts_data = {}
    sts_evaluators = []
    with zipfile.ZipFile("datasets/STS2017-extended.zip") as zip:
        filelist = zip.namelist()
        sts_files = []

        for filepath in filelist:
            filename = os.path.basename(filepath)
            if filename.startswith('STS'):
                sts_data[filename] = {'sentences1': [], 'sentences2': [], 'scores': []}

                fIn = zip.open(filepath)
                for line in io.TextIOWrapper(fIn, 'utf8'):
                    sent1, sent2, score = line.strip().split("\t")
                    score = float(score)
                    sts_data[filename]['sentences1'].append(sent1)
                    sts_data[filename]['sentences2'].append(sent2)
                    sts_data[filename]['scores'].append(score)

    for filename, data in sts_data.items():
        test_evaluator = evaluation.EmbeddingSimilarityEvaluator(data['sentences1'], data['sentences2'], data['scores'], batch_size=inference_batch_size, name=filename, show_progress_bar=False)
        sts_evaluators.append(test_evaluator)

    # Train the model
    student_model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluation.SequentialEvaluator(mse_evaluators + trans_evaluator + sts_evaluators, main_score_function=lambda scores: np.mean(scores[0:len(mse_evaluators)])),
                      epochs=num_epochs,
                      warmup_steps=num_warmup_steps,
                      evaluation_steps=num_evaluation_steps,
                      output_path=output_path,
                      save_best_model=True,
                      optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}, use_amp=True, output_path_ignore_not_empty=True
                      )


# Script was called via:
#python make_multilingual_mUSE.py parallel-sentences/Europarl-en-bg-train.tsv.gz parallel-sentences/Europarl-en-cs-train.tsv.gz parallel-sentences/Europarl-en-da-train.tsv.gz parallel-sentences/Europarl-en-de-train.tsv.gz parallel-sentences/Europarl-en-el-train.tsv.gz parallel-sentences/Europarl-en-es-train.tsv.gz parallel-sentences/Europarl-en-et-train.tsv.gz parallel-sentences/Europarl-en-fi-train.tsv.gz parallel-sentences/Europarl-en-fr-train.tsv.gz parallel-sentences/Europarl-en-hu-train.tsv.gz parallel-sentences/Europarl-en-it-train.tsv.gz parallel-sentences/Europarl-en-lt-train.tsv.gz parallel-sentences/Europarl-en-lv-train.tsv.gz parallel-sentences/Europarl-en-nl-train.tsv.gz parallel-sentences/Europarl-en-pl-train.tsv.gz parallel-sentences/Europarl-en-pt-train.tsv.gz parallel-sentences/Europarl-en-ro-train.tsv.gz parallel-sentences/Europarl-en-sk-train.tsv.gz parallel-sentences/Europarl-en-sl-train.tsv.gz parallel-sentences/Europarl-en-sv-train.tsv.gz parallel-sentences/GlobalVoices-en-ar-train.tsv.gz parallel-sentences/GlobalVoices-en-bg-train.tsv.gz parallel-sentences/GlobalVoices-en-ca-train.tsv.gz parallel-sentences/GlobalVoices-en-cs-train.tsv.gz parallel-sentences/GlobalVoices-en-da-train.tsv.gz parallel-sentences/GlobalVoices-en-de-train.tsv.gz parallel-sentences/GlobalVoices-en-el-train.tsv.gz parallel-sentences/GlobalVoices-en-es-train.tsv.gz parallel-sentences/GlobalVoices-en-fa-train.tsv.gz parallel-sentences/GlobalVoices-en-fr-train.tsv.gz parallel-sentences/GlobalVoices-en-he-train.tsv.gz parallel-sentences/GlobalVoices-en-hi-train.tsv.gz parallel-sentences/GlobalVoices-en-hu-train.tsv.gz parallel-sentences/GlobalVoices-en-id-train.tsv.gz parallel-sentences/GlobalVoices-en-it-train.tsv.gz parallel-sentences/GlobalVoices-en-ko-train.tsv.gz parallel-sentences/GlobalVoices-en-mk-train.tsv.gz parallel-sentences/GlobalVoices-en-my-train.tsv.gz parallel-sentences/GlobalVoices-en-nl-train.tsv.gz parallel-sentences/GlobalVoices-en-pl-train.tsv.gz parallel-sentences/GlobalVoices-en-pt-train.tsv.gz parallel-sentences/GlobalVoices-en-ro-train.tsv.gz parallel-sentences/GlobalVoices-en-ru-train.tsv.gz parallel-sentences/GlobalVoices-en-sq-train.tsv.gz parallel-sentences/GlobalVoices-en-sr-train.tsv.gz parallel-sentences/GlobalVoices-en-sv-train.tsv.gz parallel-sentences/GlobalVoices-en-tr-train.tsv.gz parallel-sentences/GlobalVoices-en-ur-train.tsv.gz parallel-sentences/JW300-en-ar-train.tsv.gz parallel-sentences/JW300-en-bg-train.tsv.gz parallel-sentences/JW300-en-cs-train.tsv.gz parallel-sentences/JW300-en-da-train.tsv.gz parallel-sentences/JW300-en-de-train.tsv.gz parallel-sentences/JW300-en-el-train.tsv.gz parallel-sentences/JW300-en-es-train.tsv.gz parallel-sentences/JW300-en-et-train.tsv.gz parallel-sentences/JW300-en-fa-train.tsv.gz parallel-sentences/JW300-en-fi-train.tsv.gz parallel-sentences/JW300-en-fr-train.tsv.gz parallel-sentences/JW300-en-gu-train.tsv.gz parallel-sentences/JW300-en-he-train.tsv.gz parallel-sentences/JW300-en-hi-train.tsv.gz parallel-sentences/JW300-en-hr-train.tsv.gz parallel-sentences/JW300-en-hu-train.tsv.gz parallel-sentences/JW300-en-hy-train.tsv.gz parallel-sentences/JW300-en-id-train.tsv.gz parallel-sentences/JW300-en-it-train.tsv.gz parallel-sentences/JW300-en-ja-train.tsv.gz parallel-sentences/JW300-en-ka-train.tsv.gz parallel-sentences/JW300-en-ko-train.tsv.gz parallel-sentences/JW300-en-lt-train.tsv.gz parallel-sentences/JW300-en-lv-train.tsv.gz parallel-sentences/JW300-en-mk-train.tsv.gz parallel-sentences/JW300-en-mn-train.tsv.gz parallel-sentences/JW300-en-mr-train.tsv.gz parallel-sentences/JW300-en-my-train.tsv.gz parallel-sentences/JW300-en-nl-train.tsv.gz parallel-sentences/JW300-en-pl-train.tsv.gz parallel-sentences/JW300-en-pt-train.tsv.gz parallel-sentences/JW300-en-ro-train.tsv.gz parallel-sentences/JW300-en-ru-train.tsv.gz parallel-sentences/JW300-en-sk-train.tsv.gz parallel-sentences/JW300-en-sl-train.tsv.gz parallel-sentences/JW300-en-sq-train.tsv.gz parallel-sentences/JW300-en-sv-train.tsv.gz parallel-sentences/JW300-en-th-train.tsv.gz parallel-sentences/JW300-en-tr-train.tsv.gz parallel-sentences/JW300-en-uk-train.tsv.gz parallel-sentences/JW300-en-ur-train.tsv.gz parallel-sentences/JW300-en-vi-train.tsv.gz parallel-sentences/News-Commentary-en-ar-train.tsv.gz parallel-sentences/News-Commentary-en-cs-train.tsv.gz parallel-sentences/News-Commentary-en-de-train.tsv.gz parallel-sentences/News-Commentary-en-es-train.tsv.gz parallel-sentences/News-Commentary-en-fr-train.tsv.gz parallel-sentences/News-Commentary-en-it-train.tsv.gz parallel-sentences/News-Commentary-en-ja-train.tsv.gz parallel-sentences/News-Commentary-en-nl-train.tsv.gz parallel-sentences/News-Commentary-en-pt-train.tsv.gz parallel-sentences/News-Commentary-en-ru-train.tsv.gz parallel-sentences/OpenSubtitles-en-ar-train.tsv.gz parallel-sentences/OpenSubtitles-en-bg-train.tsv.gz parallel-sentences/OpenSubtitles-en-ca-train.tsv.gz parallel-sentences/OpenSubtitles-en-cs-train.tsv.gz parallel-sentences/OpenSubtitles-en-da-train.tsv.gz parallel-sentences/OpenSubtitles-en-de-train.tsv.gz parallel-sentences/OpenSubtitles-en-el-train.tsv.gz parallel-sentences/OpenSubtitles-en-es-train.tsv.gz parallel-sentences/OpenSubtitles-en-et-train.tsv.gz parallel-sentences/OpenSubtitles-en-fa-train.tsv.gz parallel-sentences/OpenSubtitles-en-fi-train.tsv.gz parallel-sentences/OpenSubtitles-en-fr-train.tsv.gz parallel-sentences/OpenSubtitles-en-gl-train.tsv.gz parallel-sentences/OpenSubtitles-en-he-train.tsv.gz parallel-sentences/OpenSubtitles-en-hi-train.tsv.gz parallel-sentences/OpenSubtitles-en-hr-train.tsv.gz parallel-sentences/OpenSubtitles-en-hu-train.tsv.gz parallel-sentences/OpenSubtitles-en-hy-train.tsv.gz parallel-sentences/OpenSubtitles-en-id-train.tsv.gz parallel-sentences/OpenSubtitles-en-it-train.tsv.gz parallel-sentences/OpenSubtitles-en-ja-train.tsv.gz parallel-sentences/OpenSubtitles-en-ka-train.tsv.gz parallel-sentences/OpenSubtitles-en-ko-train.tsv.gz parallel-sentences/OpenSubtitles-en-lt-train.tsv.gz parallel-sentences/OpenSubtitles-en-lv-train.tsv.gz parallel-sentences/OpenSubtitles-en-mk-train.tsv.gz parallel-sentences/OpenSubtitles-en-ms-train.tsv.gz parallel-sentences/OpenSubtitles-en-nl-train.tsv.gz parallel-sentences/OpenSubtitles-en-pl-train.tsv.gz parallel-sentences/OpenSubtitles-en-pt-train.tsv.gz parallel-sentences/OpenSubtitles-en-ro-train.tsv.gz parallel-sentences/OpenSubtitles-en-ru-train.tsv.gz parallel-sentences/OpenSubtitles-en-sk-train.tsv.gz parallel-sentences/OpenSubtitles-en-sl-train.tsv.gz parallel-sentences/OpenSubtitles-en-sq-train.tsv.gz parallel-sentences/OpenSubtitles-en-sr-train.tsv.gz parallel-sentences/OpenSubtitles-en-sv-train.tsv.gz parallel-sentences/OpenSubtitles-en-th-train.tsv.gz parallel-sentences/OpenSubtitles-en-tr-train.tsv.gz parallel-sentences/OpenSubtitles-en-uk-train.tsv.gz parallel-sentences/OpenSubtitles-en-ur-train.tsv.gz parallel-sentences/OpenSubtitles-en-vi-train.tsv.gz parallel-sentences/OpenSubtitles-en-zh_cn-train.tsv.gz parallel-sentences/Tatoeba-eng-ara-train.tsv.gz parallel-sentences/Tatoeba-eng-bul-train.tsv.gz parallel-sentences/Tatoeba-eng-cat-train.tsv.gz parallel-sentences/Tatoeba-eng-ces-train.tsv.gz parallel-sentences/Tatoeba-eng-cmn-train.tsv.gz parallel-sentences/Tatoeba-eng-dan-train.tsv.gz parallel-sentences/Tatoeba-eng-deu-train.tsv.gz parallel-sentences/Tatoeba-eng-ell-train.tsv.gz parallel-sentences/Tatoeba-eng-est-train.tsv.gz parallel-sentences/Tatoeba-eng-fin-train.tsv.gz parallel-sentences/Tatoeba-eng-fra-train.tsv.gz parallel-sentences/Tatoeba-eng-glg-train.tsv.gz parallel-sentences/Tatoeba-eng-guj-train.tsv.gz parallel-sentences/Tatoeba-eng-heb-train.tsv.gz parallel-sentences/Tatoeba-eng-hin-train.tsv.gz parallel-sentences/Tatoeba-eng-hrv-train.tsv.gz parallel-sentences/Tatoeba-eng-hun-train.tsv.gz parallel-sentences/Tatoeba-eng-hye-train.tsv.gz parallel-sentences/Tatoeba-eng-ind-train.tsv.gz parallel-sentences/Tatoeba-eng-ita-train.tsv.gz parallel-sentences/Tatoeba-eng-jpn-train.tsv.gz parallel-sentences/Tatoeba-eng-kat-train.tsv.gz parallel-sentences/Tatoeba-eng-kor-train.tsv.gz parallel-sentences/Tatoeba-eng-kur-train.tsv.gz parallel-sentences/Tatoeba-eng-lit-train.tsv.gz parallel-sentences/Tatoeba-eng-lvs-train.tsv.gz parallel-sentences/Tatoeba-eng-mar-train.tsv.gz parallel-sentences/Tatoeba-eng-mkd-train.tsv.gz parallel-sentences/Tatoeba-eng-mon-train.tsv.gz parallel-sentences/Tatoeba-eng-mya-train.tsv.gz parallel-sentences/Tatoeba-eng-nld-train.tsv.gz parallel-sentences/Tatoeba-eng-nob-train.tsv.gz parallel-sentences/Tatoeba-eng-pes-train.tsv.gz parallel-sentences/Tatoeba-eng-pol-train.tsv.gz parallel-sentences/Tatoeba-eng-por-train.tsv.gz parallel-sentences/Tatoeba-eng-ron-train.tsv.gz parallel-sentences/Tatoeba-eng-rus-train.tsv.gz parallel-sentences/Tatoeba-eng-slk-train.tsv.gz parallel-sentences/Tatoeba-eng-slv-train.tsv.gz parallel-sentences/Tatoeba-eng-spa-train.tsv.gz parallel-sentences/Tatoeba-eng-sqi-train.tsv.gz parallel-sentences/Tatoeba-eng-srp-train.tsv.gz parallel-sentences/Tatoeba-eng-swe-train.tsv.gz parallel-sentences/Tatoeba-eng-tha-train.tsv.gz parallel-sentences/Tatoeba-eng-tur-train.tsv.gz parallel-sentences/Tatoeba-eng-ukr-train.tsv.gz parallel-sentences/Tatoeba-eng-urd-train.tsv.gz parallel-sentences/Tatoeba-eng-vie-train.tsv.gz parallel-sentences/Tatoeba-eng-zsm-train.tsv.gz parallel-sentences/TED2020-en-ar-train.tsv.gz parallel-sentences/TED2020-en-bg-train.tsv.gz parallel-sentences/TED2020-en-ca-train.tsv.gz parallel-sentences/TED2020-en-cs-train.tsv.gz parallel-sentences/TED2020-en-da-train.tsv.gz parallel-sentences/TED2020-en-de-train.tsv.gz parallel-sentences/TED2020-en-el-train.tsv.gz parallel-sentences/TED2020-en-es-train.tsv.gz parallel-sentences/TED2020-en-et-train.tsv.gz parallel-sentences/TED2020-en-fa-train.tsv.gz parallel-sentences/TED2020-en-fi-train.tsv.gz parallel-sentences/TED2020-en-fr-ca-train.tsv.gz parallel-sentences/TED2020-en-fr-train.tsv.gz parallel-sentences/TED2020-en-gl-train.tsv.gz parallel-sentences/TED2020-en-gu-train.tsv.gz parallel-sentences/TED2020-en-he-train.tsv.gz parallel-sentences/TED2020-en-hi-train.tsv.gz parallel-sentences/TED2020-en-hr-train.tsv.gz parallel-sentences/TED2020-en-hu-train.tsv.gz parallel-sentences/TED2020-en-hy-train.tsv.gz parallel-sentences/TED2020-en-id-train.tsv.gz parallel-sentences/TED2020-en-it-train.tsv.gz parallel-sentences/TED2020-en-ja-train.tsv.gz parallel-sentences/TED2020-en-ka-train.tsv.gz parallel-sentences/TED2020-en-ko-train.tsv.gz parallel-sentences/TED2020-en-ku-train.tsv.gz parallel-sentences/TED2020-en-lt-train.tsv.gz parallel-sentences/TED2020-en-lv-train.tsv.gz parallel-sentences/TED2020-en-mk-train.tsv.gz parallel-sentences/TED2020-en-mn-train.tsv.gz parallel-sentences/TED2020-en-mr-train.tsv.gz parallel-sentences/TED2020-en-ms-train.tsv.gz parallel-sentences/TED2020-en-my-train.tsv.gz parallel-sentences/TED2020-en-nb-train.tsv.gz parallel-sentences/TED2020-en-nl-train.tsv.gz parallel-sentences/TED2020-en-pl-train.tsv.gz parallel-sentences/TED2020-en-pt-br-train.tsv.gz parallel-sentences/TED2020-en-pt-train.tsv.gz parallel-sentences/TED2020-en-ro-train.tsv.gz parallel-sentences/TED2020-en-ru-train.tsv.gz parallel-sentences/TED2020-en-sk-train.tsv.gz parallel-sentences/TED2020-en-sl-train.tsv.gz parallel-sentences/TED2020-en-sq-train.tsv.gz parallel-sentences/TED2020-en-sr-train.tsv.gz parallel-sentences/TED2020-en-sv-train.tsv.gz parallel-sentences/TED2020-en-th-train.tsv.gz parallel-sentences/TED2020-en-tr-train.tsv.gz parallel-sentences/TED2020-en-uk-train.tsv.gz parallel-sentences/TED2020-en-ur-train.tsv.gz parallel-sentences/TED2020-en-vi-train.tsv.gz parallel-sentences/TED2020-en-zh-cn-train.tsv.gz parallel-sentences/TED2020-en-zh-tw-train.tsv.gz parallel-sentences/WikiMatrix-en-ar-train.tsv.gz parallel-sentences/WikiMatrix-en-bg-train.tsv.gz parallel-sentences/WikiMatrix-en-ca-train.tsv.gz parallel-sentences/WikiMatrix-en-cs-train.tsv.gz parallel-sentences/WikiMatrix-en-da-train.tsv.gz parallel-sentences/WikiMatrix-en-de-train.tsv.gz parallel-sentences/WikiMatrix-en-el-train.tsv.gz parallel-sentences/WikiMatrix-en-es-train.tsv.gz parallel-sentences/WikiMatrix-en-et-train.tsv.gz parallel-sentences/WikiMatrix-en-fa-train.tsv.gz parallel-sentences/WikiMatrix-en-fi-train.tsv.gz parallel-sentences/WikiMatrix-en-fr-train.tsv.gz parallel-sentences/WikiMatrix-en-gl-train.tsv.gz parallel-sentences/WikiMatrix-en-he-train.tsv.gz parallel-sentences/WikiMatrix-en-hi-train.tsv.gz parallel-sentences/WikiMatrix-en-hr-train.tsv.gz parallel-sentences/WikiMatrix-en-hu-train.tsv.gz parallel-sentences/WikiMatrix-en-id-train.tsv.gz parallel-sentences/WikiMatrix-en-it-train.tsv.gz parallel-sentences/WikiMatrix-en-ja-train.tsv.gz parallel-sentences/WikiMatrix-en-ka-train.tsv.gz parallel-sentences/WikiMatrix-en-ko-train.tsv.gz parallel-sentences/WikiMatrix-en-lt-train.tsv.gz parallel-sentences/WikiMatrix-en-mk-train.tsv.gz parallel-sentences/WikiMatrix-en-mr-train.tsv.gz parallel-sentences/WikiMatrix-en-nl-train.tsv.gz parallel-sentences/WikiMatrix-en-pl-train.tsv.gz parallel-sentences/WikiMatrix-en-pt-train.tsv.gz parallel-sentences/WikiMatrix-en-ro-train.tsv.gz parallel-sentences/WikiMatrix-en-ru-train.tsv.gz parallel-sentences/WikiMatrix-en-sk-train.tsv.gz parallel-sentences/WikiMatrix-en-sl-train.tsv.gz parallel-sentences/WikiMatrix-en-sq-train.tsv.gz parallel-sentences/WikiMatrix-en-sr-train.tsv.gz parallel-sentences/WikiMatrix-en-sv-train.tsv.gz parallel-sentences/WikiMatrix-en-tr-train.tsv.gz parallel-sentences/WikiMatrix-en-uk-train.tsv.gz parallel-sentences/WikiMatrix-en-vi-train.tsv.gz parallel-sentences/WikiMatrix-en-zh-train.tsv.gz monolingual-sentences/merged_sentences.txt.gz --dev parallel-sentences/Tatoeba-eng-ara-dev.tsv.gz parallel-sentences/Tatoeba-eng-bul-dev.tsv.gz parallel-sentences/Tatoeba-eng-ces-dev.tsv.gz parallel-sentences/Tatoeba-eng-cmn-dev.tsv.gz parallel-sentences/Tatoeba-eng-dan-dev.tsv.gz parallel-sentences/Tatoeba-eng-deu-dev.tsv.gz parallel-sentences/Tatoeba-eng-ell-dev.tsv.gz parallel-sentences/Tatoeba-eng-fin-dev.tsv.gz parallel-sentences/Tatoeba-eng-fra-dev.tsv.gz parallel-sentences/Tatoeba-eng-heb-dev.tsv.gz parallel-sentences/Tatoeba-eng-hun-dev.tsv.gz parallel-sentences/Tatoeba-eng-ita-dev.tsv.gz parallel-sentences/Tatoeba-eng-jpn-dev.tsv.gz parallel-sentences/Tatoeba-eng-mar-dev.tsv.gz parallel-sentences/Tatoeba-eng-mkd-dev.tsv.gz parallel-sentences/Tatoeba-eng-nld-dev.tsv.gz parallel-sentences/Tatoeba-eng-pol-dev.tsv.gz parallel-sentences/Tatoeba-eng-por-dev.tsv.gz parallel-sentences/Tatoeba-eng-ron-dev.tsv.gz parallel-sentences/Tatoeba-eng-rus-dev.tsv.gz parallel-sentences/Tatoeba-eng-spa-dev.tsv.gz parallel-sentences/Tatoeba-eng-srp-dev.tsv.gz parallel-sentences/Tatoeba-eng-swe-dev.tsv.gz parallel-sentences/Tatoeba-eng-tur-dev.tsv.gz parallel-sentences/Tatoeba-eng-ukr-dev.tsv.gz parallel-sentences/TED2020-en-ar-dev.tsv.gz parallel-sentences/TED2020-en-bg-dev.tsv.gz parallel-sentences/TED2020-en-ca-dev.tsv.gz parallel-sentences/TED2020-en-cs-dev.tsv.gz parallel-sentences/TED2020-en-da-dev.tsv.gz parallel-sentences/TED2020-en-de-dev.tsv.gz parallel-sentences/TED2020-en-el-dev.tsv.gz parallel-sentences/TED2020-en-es-dev.tsv.gz parallel-sentences/TED2020-en-et-dev.tsv.gz parallel-sentences/TED2020-en-fa-dev.tsv.gz parallel-sentences/TED2020-en-fi-dev.tsv.gz parallel-sentences/TED2020-en-fr-ca-dev.tsv.gz parallel-sentences/TED2020-en-fr-dev.tsv.gz parallel-sentences/TED2020-en-gl-dev.tsv.gz parallel-sentences/TED2020-en-gu-dev.tsv.gz parallel-sentences/TED2020-en-he-dev.tsv.gz parallel-sentences/TED2020-en-hi-dev.tsv.gz parallel-sentences/TED2020-en-hr-dev.tsv.gz parallel-sentences/TED2020-en-hu-dev.tsv.gz parallel-sentences/TED2020-en-hy-dev.tsv.gz parallel-sentences/TED2020-en-id-dev.tsv.gz parallel-sentences/TED2020-en-it-dev.tsv.gz parallel-sentences/TED2020-en-ja-dev.tsv.gz parallel-sentences/TED2020-en-ka-dev.tsv.gz parallel-sentences/TED2020-en-ko-dev.tsv.gz parallel-sentences/TED2020-en-ku-dev.tsv.gz parallel-sentences/TED2020-en-lt-dev.tsv.gz parallel-sentences/TED2020-en-lv-dev.tsv.gz parallel-sentences/TED2020-en-mk-dev.tsv.gz parallel-sentences/TED2020-en-mn-dev.tsv.gz parallel-sentences/TED2020-en-mr-dev.tsv.gz parallel-sentences/TED2020-en-ms-dev.tsv.gz parallel-sentences/TED2020-en-my-dev.tsv.gz parallel-sentences/TED2020-en-nb-dev.tsv.gz parallel-sentences/TED2020-en-nl-dev.tsv.gz parallel-sentences/TED2020-en-pl-dev.tsv.gz parallel-sentences/TED2020-en-pt-br-dev.tsv.gz parallel-sentences/TED2020-en-pt-dev.tsv.gz parallel-sentences/TED2020-en-ro-dev.tsv.gz parallel-sentences/TED2020-en-ru-dev.tsv.gz parallel-sentences/TED2020-en-sk-dev.tsv.gz parallel-sentences/TED2020-en-sl-dev.tsv.gz parallel-sentences/TED2020-en-sq-dev.tsv.gz parallel-sentences/TED2020-en-sr-dev.tsv.gz parallel-sentences/TED2020-en-sv-dev.tsv.gz parallel-sentences/TED2020-en-th-dev.tsv.gz parallel-sentences/TED2020-en-tr-dev.tsv.gz parallel-sentences/TED2020-en-uk-dev.tsv.gz parallel-sentences/TED2020-en-ur-dev.tsv.gz parallel-sentences/TED2020-en-vi-dev.tsv.gz parallel-sentences/TED2020-en-zh-cn-dev.tsv.gz parallel-sentences/TED2020-en-zh-tw-dev.tsv.gz parallel-sentences/WikiMatrix-en-ar-dev.tsv.gz parallel-sentences/WikiMatrix-en-bg-dev.tsv.gz parallel-sentences/WikiMatrix-en-ca-dev.tsv.gz parallel-sentences/WikiMatrix-en-cs-dev.tsv.gz parallel-sentences/WikiMatrix-en-da-dev.tsv.gz parallel-sentences/WikiMatrix-en-de-dev.tsv.gz parallel-sentences/WikiMatrix-en-el-dev.tsv.gz parallel-sentences/WikiMatrix-en-es-dev.tsv.gz parallel-sentences/WikiMatrix-en-et-dev.tsv.gz parallel-sentences/WikiMatrix-en-fa-dev.tsv.gz parallel-sentences/WikiMatrix-en-fi-dev.tsv.gz parallel-sentences/WikiMatrix-en-fr-dev.tsv.gz parallel-sentences/WikiMatrix-en-gl-dev.tsv.gz parallel-sentences/WikiMatrix-en-he-dev.tsv.gz parallel-sentences/WikiMatrix-en-hi-dev.tsv.gz parallel-sentences/WikiMatrix-en-hr-dev.tsv.gz parallel-sentences/WikiMatrix-en-hu-dev.tsv.gz parallel-sentences/WikiMatrix-en-id-dev.tsv.gz parallel-sentences/WikiMatrix-en-it-dev.tsv.gz parallel-sentences/WikiMatrix-en-ja-dev.tsv.gz parallel-sentences/WikiMatrix-en-ka-dev.tsv.gz parallel-sentences/WikiMatrix-en-ko-dev.tsv.gz parallel-sentences/WikiMatrix-en-lt-dev.tsv.gz parallel-sentences/WikiMatrix-en-mk-dev.tsv.gz parallel-sentences/WikiMatrix-en-mr-dev.tsv.gz parallel-sentences/WikiMatrix-en-nl-dev.tsv.gz parallel-sentences/WikiMatrix-en-pl-dev.tsv.gz parallel-sentences/WikiMatrix-en-pt-dev.tsv.gz parallel-sentences/WikiMatrix-en-ro-dev.tsv.gz parallel-sentences/WikiMatrix-en-ru-dev.tsv.gz parallel-sentences/WikiMatrix-en-sk-dev.tsv.gz parallel-sentences/WikiMatrix-en-sl-dev.tsv.gz parallel-sentences/WikiMatrix-en-sq-dev.tsv.gz parallel-sentences/WikiMatrix-en-sr-dev.tsv.gz parallel-sentences/WikiMatrix-en-sv-dev.tsv.gz parallel-sentences/WikiMatrix-en-tr-dev.tsv.gz parallel-sentences/WikiMatrix-en-uk-dev.tsv.gz parallel-sentences/WikiMatrix-en-vi-dev.tsv.gz parallel-sentences/WikiMatrix-en-zh-dev.tsv.gz