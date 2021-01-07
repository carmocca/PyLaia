import tempfile

import numpy as np
import pytorch_lightning as pl
import torch

import laia.common.logging as log
from laia.callbacks import ProgressBar
from laia.common.arguments import CommonArgs, CreateCRNNArgs, OptimizerArgs
from laia.common.loader import ModelLoader
from laia.dummies import DummyMNISTLines
from laia.engine import Compose, DataModule, EngineModule, ImageFeeder, ItemFeeder
from laia.losses import CTCLoss
from laia.scripts.htr.create_model import run as model
from laia.utils import SymbolsTable

train_path = tempfile.mkdtemp()
log.config()
pl.seed_everything(123)

# ==========
# Data setup
# ==========

print("Generating fake data...")
n = 10 ** 3
dummy_module = DummyMNISTLines(tr_n=n, va_n=int(0.1 * n), samples_per_space=5)
dummy_module.prepare_data()

syms = train_path + "/syms"
syms_table = SymbolsTable()
for k, v in dummy_module.syms.items():
    syms_table.add(v, k)
syms_table.save(syms)

model(
    syms,
    fixed_input_height=28,
    save_model=True,
    common=CommonArgs(train_path=train_path),
    crnn=CreateCRNNArgs(
        cnn_num_features=[16, 32, 48, 64],
        rnn_units=32,
        rnn_layers=1,
        rnn_dropout=0,
    ),
)

# ==============
# Training setup
# ==============

loader = ModelLoader(train_path, device="cpu")
model = loader.load()


class TestEngineModule(EngineModule):
    # parent throws exception if NaN or ± inf
    def compute_loss(self, batch, batch_y_hat, batch_y):
        return self.criterion(batch_y_hat, batch_y)


class TestCTCLoss(CTCLoss):
    # invalid_return = None
    invalid_return = "inf"
    # invalid_return = "nan"

    def forward(self, *args, **kwargs):
        loss = super().forward(*args, **kwargs)
        # weighted coin flip
        if np.random.binomial(1, 0.1):
            if self.invalid_return is None:
                loss = None
            else:
                loss.data = torch.tensor(float(self.invalid_return), device=loss.device)
        return loss


engine_module = TestEngineModule(
    model,
    TestCTCLoss(),
    batch_input_fn=Compose([ItemFeeder("img"), ImageFeeder()]),
    batch_target_fn=ItemFeeder("txt"),
    optimizer=OptimizerArgs(learning_rate=0.001),
)

data_module = DataModule(
    syms=syms_table,
    img_dirs=[str(dummy_module.root / p) for p in ("tr", "va")],
    tr_txt_table=str(dummy_module.root / "tr.gt"),
    va_txt_table=str(dummy_module.root / "va.gt"),
    batch_size=8,
    stage="fit",
)

trainer = pl.Trainer(
    default_root_dir=train_path,
    callbacks=[ProgressBar()],
    gpus=1,
    accelerator="ddp",
)

trainer.fit(engine_module, datamodule=data_module)
