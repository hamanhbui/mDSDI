import logging
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from algorithms.ERM.src.dataloaders import dataloader_factory
from algorithms.ERM.src.models import model_factory
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Classifier(nn.Module):
    def __init__(self, feature_dim, classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, z):
        y = self.classifier(z)
        return y


def set_tr_val_samples_labels(meta_filenames, val_size):
    sample_tr_paths, class_tr_labels, sample_val_paths, class_val_labels = [], [], [], []

    for idx_domain, meta_filename in enumerate(meta_filenames):
        column_names = ["filename", "class_label"]
        data_frame = pd.read_csv(meta_filename, header=None, names=column_names, sep="\s+")
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)

        split_idx = int(len(data_frame) * (1 - val_size))
        sample_tr_paths.append(data_frame["filename"][:split_idx])
        class_tr_labels.append(data_frame["class_label"][:split_idx])

        sample_val_paths.extend(data_frame["filename"][split_idx:])
        class_val_labels.extend(data_frame["class_label"][split_idx:])

    return sample_tr_paths, class_tr_labels, sample_val_paths, class_val_labels


def set_test_samples_labels(meta_filenames):
    sample_paths, class_labels = [], []
    for idx_domain, meta_filename in enumerate(meta_filenames):
        column_names = ["filename", "class_label"]
        data_frame = pd.read_csv(meta_filename, header=None, names=column_names, sep="\s+")
        sample_paths.extend(data_frame["filename"])
        class_labels.extend(data_frame["class_label"])

    return sample_paths, class_labels


class Trainer_ERM:
    def __init__(self, args, device, exp_idx):
        self.args = args
        self.device = device
        self.writer = self.set_writer(
            log_dir="algorithms/"
            + self.args.algorithm
            + "/results/tensorboards/"
            + self.args.exp_name
            + "_"
            + exp_idx
            + "/"
        )
        self.checkpoint_name = (
            "algorithms/" + self.args.algorithm + "/results/checkpoints/" + self.args.exp_name + "_" + exp_idx
        )
        self.plot_dir = (
            "algorithms/" + self.args.algorithm + "/results/plots/" + self.args.exp_name + "_" + exp_idx + "/"
        )

        (
            src_tr_sample_paths,
            src_tr_class_labels,
            src_val_sample_paths,
            src_val_class_labels,
        ) = set_tr_val_samples_labels(self.args.src_train_meta_filenames, self.args.val_size)
        test_sample_paths, test_class_labels = set_test_samples_labels(self.args.target_test_meta_filenames)
        self.train_loaders = []
        for i in range(self.args.n_domain_classes):
            self.train_loaders.append(
                DataLoader(
                    dataloader_factory.get_train_dataloader(self.args.dataset)(
                        src_path=self.args.src_data_path,
                        sample_paths=src_tr_sample_paths[i],
                        class_labels=src_tr_class_labels[i],
                        domain_label=i,
                    ),
                    batch_size=self.args.batch_size,
                    shuffle=True,
                )
            )

        if self.args.val_size != 0:
            self.val_loader = DataLoader(
                dataloader_factory.get_test_dataloader(self.args.dataset)(
                    src_path=self.args.src_data_path,
                    sample_paths=src_val_sample_paths,
                    class_labels=src_val_class_labels,
                ),
                batch_size=self.args.batch_size,
                shuffle=True,
            )
        else:
            self.val_loader = DataLoader(
                dataloader_factory.get_test_dataloader(self.args.dataset)(
                    src_path=self.args.src_data_path, sample_paths=test_sample_paths, class_labels=test_class_labels
                ),
                batch_size=self.args.batch_size,
                shuffle=True,
            )

        self.test_loader = DataLoader(
            dataloader_factory.get_test_dataloader(self.args.dataset)(
                src_path=self.args.src_data_path, sample_paths=test_sample_paths, class_labels=test_class_labels
            ),
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        self.model = model_factory.get_model(self.args.model)().to(self.device)
        self.classifier = Classifier(self.args.feature_dim, self.args.n_classes).to(self.device)

        optimizer_params = list(self.model.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.Adam(optimizer_params, lr=self.args.learning_rate)

        self.criterion = nn.CrossEntropyLoss()
        self.val_loss_min = np.Inf
        self.val_acc_max = 0

    def set_writer(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        shutil.rmtree(log_dir)
        return SummaryWriter(log_dir)

    def save_plot(self):
        checkpoint = torch.load(self.checkpoint_name + ".pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.model.eval()
        self.classifier.eval()

        Z_out, Y_out, Y_domain_out = [], [], []
        Z_test, Y_test, Y_domain_test = [], [], []

        with torch.no_grad():
            self.train_iter_loaders = []
            for train_loader in self.train_loaders:
                self.train_iter_loaders.append(iter(train_loader))

            for d_idx in range(len(self.train_iter_loaders)):
                train_loader = self.train_iter_loaders[d_idx]
                for idx in range(len(train_loader)):
                    samples, labels, domain_labels = train_loader.next()
                    samples = samples.to(self.device)
                    labels = labels.to(self.device)
                    domain_labels = domain_labels.to(self.device)
                    z = self.model(samples)
                    Z_out += z.tolist()
                    Y_out += labels.tolist()
                    Y_domain_out += domain_labels.tolist()

            for iteration, (samples, labels, domain_labels) in enumerate(self.test_loader):
                samples, labels = samples.to(self.device), labels.to(self.device)
                z = self.model(samples)
                Z_test += z.tolist()
                Y_test += labels.tolist()
                Y_domain_test += domain_labels.tolist()

        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)
        with open(self.plot_dir + "Z_out.pkl", "wb") as fp:
            pickle.dump(Z_out, fp)
        with open(self.plot_dir + "Y_out.pkl", "wb") as fp:
            pickle.dump(Y_out, fp)
        with open(self.plot_dir + "Y_domain_out.pkl", "wb") as fp:
            pickle.dump(Y_domain_out, fp)

        with open(self.plot_dir + "Z_test.pkl", "wb") as fp:
            pickle.dump(Z_test, fp)
        with open(self.plot_dir + "Y_test.pkl", "wb") as fp:
            pickle.dump(Y_test, fp)
        with open(self.plot_dir + "Y_domain_test.pkl", "wb") as fp:
            pickle.dump(Y_domain_test, fp)

    def train(self):
        self.model.train()
        self.classifier.train()

        n_class_corrected = 0
        total_classification_loss = 0
        total_samples = 0
        self.train_iter_loaders = []
        for train_loader in self.train_loaders:
            self.train_iter_loaders.append(iter(train_loader))

        for iteration in range(self.args.iterations):
            samples, labels = [], []

            for idx in range(len(self.train_iter_loaders)):
                if (iteration % len(self.train_iter_loaders[idx])) == 0:
                    self.train_iter_loaders[idx] = iter(self.train_loaders[idx])
                train_loader = self.train_iter_loaders[idx]

                itr_samples, itr_labels, itr_domain_labels = train_loader.next()
                samples.append(itr_samples)
                labels.append(itr_labels)

            samples = torch.cat(samples, dim=0).to(self.device)
            labels = torch.cat(labels, dim=0).to(self.device)

            predicted_classes = self.classifier(self.model(samples))
            classification_loss = self.criterion(predicted_classes, labels)
            total_classification_loss += classification_loss.item()

            _, predicted_classes = torch.max(predicted_classes, 1)
            n_class_corrected += (predicted_classes == labels).sum().item()
            total_samples += len(samples)

            self.optimizer.zero_grad()
            classification_loss.backward()
            self.optimizer.step()

            if iteration % self.args.step_eval == 0:
                self.writer.add_scalar("Accuracy/train", 100.0 * n_class_corrected / total_samples, iteration)
                self.writer.add_scalar("Loss/train", total_classification_loss / total_samples, iteration)
                logging.info(
                    "Train set: Iteration: [{}/{}]\tAccuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}".format(
                        iteration,
                        self.args.iterations,
                        n_class_corrected,
                        total_samples,
                        100.0 * n_class_corrected / total_samples,
                        total_classification_loss / total_samples,
                    )
                )
                self.evaluate(iteration)

            n_class_corrected = 0
            total_classification_loss = 0
            total_samples = 0

    def evaluate(self, n_iter):
        self.model.eval()
        self.classifier.eval()

        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.val_loader):
                samples, labels = samples.to(self.device), labels.to(self.device)
                predicted_classes = self.classifier(self.model(samples))
                classification_loss = self.criterion(predicted_classes, labels)
                total_classification_loss += classification_loss.item()

                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()

        self.writer.add_scalar("Accuracy/validate", 100.0 * n_class_corrected / len(self.val_loader.dataset), n_iter)
        self.writer.add_scalar("Loss/validate", total_classification_loss / len(self.val_loader.dataset), n_iter)
        logging.info(
            "Val set: Accuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}".format(
                n_class_corrected,
                len(self.val_loader.dataset),
                100.0 * n_class_corrected / len(self.val_loader.dataset),
                total_classification_loss / len(self.val_loader.dataset),
            )
        )

        val_acc = n_class_corrected / len(self.val_loader.dataset)
        val_loss = total_classification_loss / len(self.val_loader.dataset)

        self.model.train()
        self.classifier.train()
        if self.args.val_size != 0:
            if self.val_loss_min > val_loss:
                self.val_loss_min = val_loss
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "classifier_state_dict": self.classifier.state_dict(),
                    },
                    self.checkpoint_name + ".pt",
                )
        else:
            if self.val_acc_max < val_acc:
                self.val_acc_max = val_acc
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "classifier_state_dict": self.classifier.state_dict(),
                    },
                    self.checkpoint_name + ".pt",
                )

    def test(self):
        checkpoint = torch.load(self.checkpoint_name + ".pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.model.eval()
        self.classifier.eval()

        n_class_corrected = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.test_loader):
                samples, labels = samples.to(self.device), labels.to(self.device)
                predicted_classes = self.classifier(self.model(samples))
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()

        logging.info(
            "Test set: Accuracy: {}/{} ({:.2f}%)".format(
                n_class_corrected,
                len(self.test_loader.dataset),
                100.0 * n_class_corrected / len(self.test_loader.dataset),
            )
        )
