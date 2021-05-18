from transformers import AutoTokenizer, AutoConfig
from model import BertReranker
from dataset import MsmarcoPassageCrossEncoderTrainSet
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from tools.pl_callbacks import CheckpointEveryNSteps
import torch
import argparse

model_type = "bert-large-uncased"
cache_dir = ".cache"
tokenizer = AutoTokenizer.from_pretrained(model_type, cache_dir=cache_dir)


def collate_fn(batch):
    queries = []
    passages = []
    labels = []
    for query, passage, label in batch:
        queries.append(query)
        passages.append(passage)
        labels.append(label)
    inputs = tokenizer(queries, passages, return_tensors="pt", padding=True, truncation=True)

    return inputs, torch.stack(labels)


def main(args):
    seed_everything(313)
    torch.multiprocessing.set_sharing_strategy('file_system')
    log_name = f"bert_reranker_{args.training_setting}_bs64_gpu2_RepBERT_top1000_adam_lr3e-6_warm5000"
    dataset_path = "data/runs/DPR/msmarco/RepBert_msmarco_train_top1000_results.txt"
    path_to_qrel = "data/qrels/passage/qrels.train.tsv"
    path_to_query = "data/queries/passage/queries.train.tsv"
    collection_path = "data/collection/passage/collection.tsv"

    batch_size = 64
    lr = 3e-6
    warm_up_steps = 5000
    optimizer = 'adam'
    save_path = "ckpts/CE/BertReranker/{}/{}".format(model_type, log_name)
    gpus_per_node = 2
    num_nodes = 1
    num_epochs = 4  #
    num_neg_per_pos = 4
    num_gpus = gpus_per_node * num_nodes
    save_step_frequency = 5000

    if args.training_setting == 'typos_aware':
        insert_typo = True
    else:
        insert_typo = False

    tb_logger = pl_loggers.TensorBoardLogger('logs/', name=log_name)

    # tokenizer = AutoTokenizer.from_pretrained(model_type, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(model_type, cache_dir=cache_dir)

    gradient_checkpointing = True  # trade-off training speed for batch_size per GPU
    config.gradient_checkpointing = gradient_checkpointing

    train_set = MsmarcoPassageCrossEncoderTrainSet(dataset_path,
                                                   path_to_qrel,
                                                   path_to_query,
                                                   collection_path,
                                                   tokenizer,
                                                   num_epochs=num_epochs,
                                                   num_neg_per_pos=num_neg_per_pos,
                                                   insert_typo=insert_typo)

    print("Training set size:", len(train_set))

    callbacks = [CheckpointEveryNSteps(save_step_frequency, save_path),
                 LearningRateMonitor(logging_interval='step')]


    model = BertReranker(encoder_name_or_dir=model_type,
                         encoder_config=config,
                         cache_dir=cache_dir,
                         optimizer=optimizer,
                         lr=lr,
                         warm_up_steps=warm_up_steps,
                         num_gpus=num_gpus,
                         batch_size=batch_size,
                         train_set_size=len(train_set),
                         num_epochs=1,  # this is fake num_epoch.
                         num_neg_per_pos=num_neg_per_pos,
                         )

    train_dataloader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  pin_memory=True,
                                  shuffle=True,
                                  num_workers=10,
                                  collate_fn=collate_fn
                                  )

    trainer = Trainer(max_epochs=1,
                      gpus=gpus_per_node,
                      num_nodes=num_nodes,
                      checkpoint_callback=False,
                      logger=tb_logger,
                      # amp_backend='apex',
                      # amp_level='O1',
                      accelerator="ddp",
                      plugins='ddp_sharded',
                      log_every_n_steps=10,
                      callbacks=callbacks,
                      )

    trainer.fit(model, train_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--training_setting",
                        choices=["standard", "typos_aware"],
                        required=True)
    args = parser.parse_args()

    main(args)

