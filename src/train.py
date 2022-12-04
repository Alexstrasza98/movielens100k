import neptune.new as neptune
from tqdm import tqdm

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.data import DataLoader

from src.utils import AverageMeter
from src.data_preprocessing import prepare_data, title2bert
from src.data_loader import Rating_Dataset
from src.model import MovieRater_Simple, MovieRater_Embedding
from src.evaluate import evaluate_model
from src.configs import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Learner:
    def __init__(self, train_config, run):
        """
        train_config: a dict of {'model': (required) model to use,
                                 'loss_fn': (required) loss function to use,
                                 'optim': (required) optimizer to use,
                                 'scheduler': (required) lr scheduler to use,
                                 'epochs': (required) how many epochs to train on,
                                 'train_dl': (required) training data loader,
                                 'val_dl': (optional) validation data loader,
                                 }
        """

        self.run = run
        self.config = train_config
        self.model = train_config["model"]
        self.model_name = self.model.model_name
        self.model.to(device)
        self.criterion = train_config["loss_fn"]
        self.optimizer = train_config["optim"]
        self.scheduler = train_config["scheduler"]
        self.train_dl = train_config["train_dl"]
        self.val_dl = None
        if "val_dl" in train_config:
            self.val_dl = train_config["val_dl"]

        # initialization of record variables
        self.train_metric_all = []
        if self.val_dl is not None:
            self.val_ndcg_all = []
        self.best_metric = 0.0
        self.model_path = f'./model/{self.model_name}_best.pth'
        self.run["model"].track_files(self.model_path)

    def train(self, verbose=True):
        cudnn.benchmark = True
        epochs = self.config['epochs']

        for epoch in range(epochs):
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))

            # train steps
            self.model.train()
            train_metric = self.train_step(epoch, verbose)
            self.train_metric_all.append(train_metric)
            self.run["train/epoch/loss"].log(train_metric)

            # validation
            if self.val_dl is not None:
                self.model.eval()
                val_ndcg, val_loss = evaluate_model(self.val_dl, self.model, self.criterion)
                self.val_ndcg_all.append(val_ndcg)
                self.run["val/epoch/ndcg"].log(val_ndcg)
                self.run["val/epoch/loss"].log(val_loss)

                if verbose:
                    print('Epoch[{}] *Validation*: LOSS {loss:.4f}  NDCG@1 {ndcg:.4f}'.format(epoch,
                                                                                              loss=val_loss,
                                                                                              ndcg=val_ndcg))
                if val_ndcg > self.best_metric:
                    self.save_model()
                    self.best_metric = val_ndcg

            self.scheduler.step()

        return self.val_ndcg_all if self.val_dl is not None else self.train_metric_all

    def train_step(self, epoch, verbose=True):
        """
        Run one train epoch
        """
        losses = AverageMeter()

        for i, (samples, labels, _) in enumerate(self.train_dl):

            # fetch batch data
            labels = labels.to(device)
            samples = samples.to(device)

            # compute output
            logits = self.model(samples)
            loss = self.criterion(logits, labels)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure accuracy and record loss
            losses.update(loss.item(), labels.size(0))
            self.run["train/batch/loss"].log(loss.item())

            # measure elapsed time
            if i % 100 == 0 and verbose:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.avg:.4f}\t'.format(
                    epoch, i, len(self.train_dl), loss=losses))

        return losses.avg

    def validate(self, epoch, verbose=True):
        """
        Run evaluation
        """
        losses = AverageMeter()

        with torch.no_grad():
            for samples, labels, _ in self.val_dl:
                # fetch batch data
                labels = labels.to(device)
                samples = samples.to(device)

                # compute output
                logits = self.model(samples)

                # measure accuracy and record loss
                loss = self.criterion(logits, labels)
                losses.update(loss.item(), labels.size(0))
        if verbose:
            print('Epoch[{}] *Validation*: LOSS {top1.avg:.4f}'.format(epoch, top1=losses))

        return losses.avg

    def save_model(self):
        print(f"Best model saved at {self.model_path}!")
        torch.save(self.model.state_dict(), self.model_path)

    def check_point(self):
        # not implemented yet
        pass

    def logging(self):
        # not implemented yet
        pass


def main(params, build_bert=False, with_title=True):
    run = neptune.init(
        project="alexyannn/movielens100k",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNjA4ZTkyNi02OGQwLTRlNTAtYmZhZi1mNjQ0OTc5Y2Q0MzcifQ==",
        name=NEPTUNE_NAME,
        description=DESCRIPTION,
        mode=MODE
    )

    # Prepare Dataset
    if build_bert:
        title2bert(movie_path="./data/u.item")
    run["parameters"] = params

    data = prepare_data(rating_path_train=TRAIN_SET_PATH,
                        rating_path_test=TEST_SET_PATH,
                        movie_path="./data/u.item",
                        user_path="./data/u.user",
                        validation=VALIDATION)
    train_input, train_labels, train_ids = data["train"]
    val_input, val_labels, val_ids = data["val"]
    test_input, test_labels, test_ids = data["test"]

    train_ds = Rating_Dataset(train_input, train_labels, train_ids, with_title)
    train_dl = DataLoader(train_ds, batch_size=params["bs"], shuffle=True)
    val_ds = Rating_Dataset(val_input, val_labels, val_ids, with_title)
    val_dl = DataLoader(val_ds, batch_size=params["bs"], shuffle=True)

    # Training
    input_dim = train_ds[0][0].shape[0]
    if with_title:
        data["feature_tags"]["numeric"].extend(range(input_dim - 768, input_dim))

    # model = MovieRater_Simple(input_dim, params["hidden_size"], model_name="movierater_simple")
    model = MovieRater_Embedding(input_dim, params["hidden_size"], data["feature_tags"])
    optimizer = optim.SGD(model.parameters(), lr=params["lr"], momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, (params["epochs"] // 3) * 2)
    loss_fn = nn.MSELoss()
    epochs = params["epochs"]

    train_config = {"model": model,
                    "loss_fn": loss_fn,
                    "optim": optimizer,
                    "scheduler": scheduler,
                    "epochs": epochs,
                    "train_dl": train_dl,
                    "val_dl": val_dl}

    learner = Learner(train_config, run)
    train_history = learner.train()

    # Final Evaluation
    test_ds = Rating_Dataset(test_input, test_labels, test_ids)
    test_dl = DataLoader(test_ds, batch_size=params["bs"], shuffle=False)
    test_ndcg, test_loss = evaluate_model(test_dl, learner.model, loss_fn)

    run["test/ndcg"] = test_ndcg
    run["test/loss"] = test_loss
    run["final_model"].upload(learner.model_path)

    run.stop()

    return train_history, test_ndcg, test_loss
