import argparse

import pytorch_lightning as pl

from config.parser import parse_args
from memfof.model_lit import MEMFOFLit, DataModule


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )
    args = parse_args(parser)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
    )

    datamodule = DataModule(args)
    model = MEMFOFLit(args)
    trainer.validate(model, datamodule)
