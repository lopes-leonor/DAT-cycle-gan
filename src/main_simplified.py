from argparse import ArgumentParser
from src import settings
from src.training.train import train


def main():
    args = parse_args()
    train(args)





def parse_args():
    parser = ArgumentParser()

    # Directory args
    parser.add_argument("--project_dir", default=settings.PROJ_DIR)
    parser.add_argument("--train_dir", default=settings.TRAIN_DIR)
    parser.add_argument("--test_dir", default=settings.TRAIN_DIR)

    # Data args
    parser.add_argument("--train_pet_csv", default=settings.TRAIN_PET_CSV)
    parser.add_argument("--train_spect_csv", default=settings.TRAIN_SPECT_CSV)

    # Training args
    parser.add_argument("--gpus", type=str, default=settings.GPUS)
    parser.add_argument("--epochs", type=int, default=settings.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--img_shape",  default=settings.IMG_SHAPE)
    parser.add_argument("--loss_list", default=settings.LOSS_LIST)
    parser.add_argument("--loss_weights", default=settings.LOSS_WEIGHTS)
    parser.add_argument("--mask_file", default=settings.MASK_FILE)
    parser.add_argument("--paired_by_label", type=bool, default=settings.PAIRED_BY_LABEL)
    parser.add_argument("--gen_disc_update", default=settings.GEN_DISC_UPDATE)


    args = parser.parse_args()
    return args






if __name__ == '__main__':
    main()