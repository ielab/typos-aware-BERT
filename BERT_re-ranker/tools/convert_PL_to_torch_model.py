from model import BertReranker, LCEreranker
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    model_name = "CE_LCE_ernie_large_bs64_gpu4_DE_random1000_adam_1e5_step3000"
    model_type = "nghuyong/ernie-2.0-large-en"
    # model_type = "bert-base-uncased"
    ckpt_path = "../ckpts/ernie_large/adam/{}.ckpt".format(model_name)
    model = LCEreranker.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.encoder.save_pretrained("encoder")
    torch.save(model.project, "encoder/project")


if __name__ == '__main__':
    main()

