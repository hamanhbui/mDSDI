import copy
import logging
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from algorithms.mDSDI.src.dataloaders import dataloader_factory
from algorithms.mDSDI.src.models import model_factory
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class Domain_Discriminator(nn.Module):
    def __init__(self, feature_dim, domain_classes):
        super(Domain_Discriminator, self).__init__()
        self.class_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, domain_classes),
        )

    def forward(self, di_z):
        y = self.class_classifier(GradReverse.apply(di_z))
        return y


class Classifier(nn.Module):
    def __init__(self, feature_dim, classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(int(feature_dim * 2), classes)

    def forward(self, di_z, ds_z):
        z = torch.cat((di_z, ds_z), dim=1)
        y = self.classifier(z)
        return y


class ZS_Domain_Classifier(nn.Module):
    def __init__(self, feature_dim, domain_classes):
        super(ZS_Domain_Classifier, self).__init__()
        self.class_classifier = nn.Sequential(nn.Linear(feature_dim, domain_classes))

    def forward(self, ds_z):
        y = self.class_classifier(ds_z)
        return y


def random_pairs_of_minibatches(samples, labels):
    perm = torch.randperm(len(samples)).tolist()
    pairs = []

    for j in range(len(samples)):
        xi, yi = [], []
        for i in range(len(samples)):
            if i != j:
                xi += samples[perm[i]]
                yi += labels[perm[i]]

        xi = torch.stack(xi)
        yi = torch.stack(yi)
        xj, yj = samples[perm[j]], labels[perm[j]]

        pairs.append(((xi, yi), (xj, yj)))

    return pairs


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


class Trainer_mDSDI:
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
        self.zi_model = model_factory.get_model(self.args.model)().to(self.device)
        self.zs_model = model_factory.get_model(self.args.model)().to(self.device)

        self.classifier = Classifier(feature_dim=self.args.feature_dim, classes=self.args.n_classes).to(self.device)
        self.zs_domain_classifier = ZS_Domain_Classifier(
            feature_dim=self.args.feature_dim, domain_classes=self.args.n_domain_classes
        ).to(self.device)
        self.domain_discriminator = Domain_Discriminator(
            feature_dim=self.args.feature_dim, domain_classes=self.args.n_domain_classes
        ).to(self.device)

        optimizer_params = (
            list(self.zi_model.parameters())
            + list(self.zs_model.parameters())
            + list(self.classifier.parameters())
            + list(self.domain_discriminator.parameters())
            + list(self.zs_domain_classifier.parameters())
        )
        self.optimizer = torch.optim.Adam(optimizer_params, lr=self.args.learning_rate)

        meta_optimizer_params = list(self.zs_model.parameters()) + list(self.classifier.parameters())
        self.meta_optimizer = torch.optim.Adam(meta_optimizer_params, lr=self.args.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.reconstruction_criterion = nn.MSELoss()
        self.val_loss_min = np.Inf
        self.val_acc_max = 0

    def set_writer(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        shutil.rmtree(log_dir)
        return SummaryWriter(log_dir)

    def save_plot(self):
        checkpoint = torch.load(self.checkpoint_name + ".pt")
        self.zi_model.load_state_dict(checkpoint["zi_model_state_dict"])
        self.zs_model.load_state_dict(checkpoint["zs_model_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.zs_domain_classifier.load_state_dict(checkpoint["zs_domain_classifier_state_dict"])
        self.domain_discriminator.load_state_dict(checkpoint["domain_discriminator_state_dict"])

        self.zi_model.eval()
        self.zs_model.eval()
        self.classifier.eval()
        self.zs_domain_classifier.eval()
        self.domain_discriminator.eval()

        Zi_out, Zs_out, Y_out, Y_domain_out = [], [], [], []
        Zi_test, Zs_test, Y_test, Y_domain_test = [], [], [], []

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
                    di_z, ds_z = self.zi_model(samples), self.zs_model(samples)

                    Zi_out += di_z.tolist()
                    Zs_out += ds_z.tolist()
                    Y_out += labels.tolist()
                    Y_domain_out += domain_labels.tolist()

            for iteration, (samples, labels, domain_labels) in enumerate(self.test_loader):
                samples, labels = samples.to(self.device), labels.to(self.device)
                di_z, ds_z = self.zi_model(samples), self.zs_model(samples)
                Zi_test += di_z.tolist()
                Zs_test += ds_z.tolist()
                Y_test += labels.tolist()
                Y_domain_test += domain_labels.tolist()

        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)
        with open(self.plot_dir + "Zi_out.pkl", "wb") as fp:
            pickle.dump(Zi_out, fp)
        with open(self.plot_dir + "Zs_out.pkl", "wb") as fp:
            pickle.dump(Zs_out, fp)
        with open(self.plot_dir + "Y_out.pkl", "wb") as fp:
            pickle.dump(Y_out, fp)
        with open(self.plot_dir + "Y_domain_out.pkl", "wb") as fp:
            pickle.dump(Y_domain_out, fp)

        with open(self.plot_dir + "Zi_test.pkl", "wb") as fp:
            pickle.dump(Zi_test, fp)
        with open(self.plot_dir + "Zs_test.pkl", "wb") as fp:
            pickle.dump(Zs_test, fp)
        with open(self.plot_dir + "Y_test.pkl", "wb") as fp:
            pickle.dump(Y_test, fp)
        with open(self.plot_dir + "Y_domain_test.pkl", "wb") as fp:
            pickle.dump(Y_domain_test, fp)

    def train(self):
        self.zi_model.train()
        self.zs_model.train()
        self.classifier.train()
        self.zs_domain_classifier.train()
        self.domain_discriminator.train()

        n_class_corrected = 0
        n_domain_class_corrected = 0
        n_zs_domain_class_corrected = 0

        total_classification_loss = 0
        total_dc_loss = 0
        total_zsc_loss = 0
        total_disentangle_loss = 0
        total_samples = 0
        total_meta_samples = 0
        self.train_iter_loaders = []
        for train_loader in self.train_loaders:
            self.train_iter_loaders.append(iter(train_loader))

        for iteration in range(self.args.iterations):
            samples, labels, domain_labels = [], [], []

            for idx in range(len(self.train_iter_loaders)):
                if (iteration % len(self.train_iter_loaders[idx])) == 0:
                    self.train_iter_loaders[idx] = iter(self.train_loaders[idx])
                train_loader = self.train_iter_loaders[idx]

                itr_samples, itr_labels, itr_domain_labels = train_loader.next()

                samples.append(itr_samples)
                labels.append(itr_labels)
                domain_labels.append(itr_domain_labels)

            tr_samples = torch.cat(samples, dim=0).to(self.device)
            tr_labels = torch.cat(labels, dim=0).to(self.device)
            tr_domain_labels = torch.cat(domain_labels, dim=0).to(self.device)

            di_z, ds_z = self.zi_model(tr_samples), self.zs_model(tr_samples)

            # Distangle by Covariance Matrix
            mdi_z = torch.mean(di_z, 0)
            mds_z = torch.mean(ds_z, 0)

            di_z_n = di_z - mdi_z[None, :]
            ds_z_n = ds_z - mds_z[None, :]
            C = di_z_n[:, :, None] * ds_z_n[:, None, :]

            target_cr = torch.zeros(C.shape[0], C.shape[1], C.shape[2]).to(self.device)
            disentangle_loss = nn.MSELoss()(C, target_cr)
            total_disentangle_loss += disentangle_loss.item()

            di_predicted_domain = self.domain_discriminator(di_z)
            predicted_domain_di_loss = self.criterion(di_predicted_domain, tr_domain_labels)
            total_dc_loss += predicted_domain_di_loss.item()

            ds_predicted_classes = self.zs_domain_classifier(ds_z)
            predicted_domain_ds_loss = self.criterion(ds_predicted_classes, tr_domain_labels)
            total_zsc_loss += predicted_domain_ds_loss.item()

            predicted_classes = self.classifier(di_z, ds_z)
            classification_loss = self.criterion(predicted_classes, tr_labels)
            total_classification_loss += classification_loss.item()

            total_loss = classification_loss + predicted_domain_di_loss + predicted_domain_ds_loss + disentangle_loss

            _, ds_predicted_classes = torch.max(ds_predicted_classes, 1)
            n_zs_domain_class_corrected += (ds_predicted_classes == tr_domain_labels).sum().item()
            _, di_predicted_domain = torch.max(di_predicted_domain, 1)
            n_domain_class_corrected += (di_predicted_domain == tr_domain_labels).sum().item()
            _, predicted_classes = torch.max(predicted_classes, 1)
            n_class_corrected += (predicted_classes == tr_labels).sum().item()

            total_samples += len(tr_samples)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Meta-training
            self.meta_optimizer.zero_grad()
            self_param = list(self.zs_model.parameters()) + list(self.classifier.parameters())
            for p in self_param:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            for (mtr_samples, mtr_labels), (mte_samples, mte_labels) in random_pairs_of_minibatches(samples, labels):
                mtr_samples = mtr_samples.to(self.device)
                mtr_labels = mtr_labels.to(self.device)
                mte_samples = mte_samples.to(self.device)
                mte_labels = mte_labels.to(self.device)

                inner_zs_model = copy.deepcopy(self.zs_model)
                inner_classifier = copy.deepcopy(self.classifier)

                inner_param = list(inner_zs_model.parameters()) + list(inner_classifier.parameters())

                inner_opt = torch.optim.Adam(inner_param, lr=self.args.learning_rate)

                di_z, ds_z = self.zi_model(mtr_samples), inner_zs_model(mtr_samples)
                predicted_classes = inner_classifier(di_z, ds_z)
                inner_obj = self.criterion(predicted_classes, mtr_labels)
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == mtr_labels).sum().item()

                inner_opt.zero_grad()
                inner_obj.backward()
                inner_opt.step()

                for p_tgt, p_src in zip(self_param, inner_param):
                    if p_src.grad is not None:
                        p_tgt.grad.data.add_(p_src.grad.data / self.args.n_domain_classes)

                total_classification_loss += inner_obj.item()

                di_z, ds_z = self.zi_model(mte_samples), inner_zs_model(mte_samples)
                predicted_classes = inner_classifier(di_z, ds_z)
                loss_inner_j = self.criterion(predicted_classes, mte_labels)
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == mte_labels).sum().item()

                grad_inner_j = torch.autograd.grad(loss_inner_j, inner_param, allow_unused=True)

                total_classification_loss += (1.0 * loss_inner_j).item()

                for p, g_j in zip(self_param, grad_inner_j):
                    if g_j is not None:
                        p.grad.data.add_(1.0 * g_j.data / self.args.n_domain_classes)

                total_meta_samples += len(mtr_samples)
                total_meta_samples += len(mte_samples)

            total_class_samples = total_samples + total_meta_samples
            self.meta_optimizer.step()

            if iteration % self.args.step_eval == 0:
                self.writer.add_scalar("Accuracy/train", 100.0 * n_class_corrected / total_class_samples, iteration)
                self.writer.add_scalar(
                    "Accuracy/domainAT_train", 100.0 * n_domain_class_corrected / total_samples, iteration
                )
                self.writer.add_scalar(
                    "Accuracy/domainZS_train", 100.0 * n_zs_domain_class_corrected / total_samples, iteration
                )
                self.writer.add_scalar("Loss/train", total_classification_loss / total_class_samples, iteration)
                self.writer.add_scalar("Loss/domainAT_train", total_dc_loss / total_samples, iteration)
                self.writer.add_scalar("Loss/domainZS_train", total_zsc_loss / total_samples, iteration)
                self.writer.add_scalar("Loss/disentangle", total_disentangle_loss / total_samples, iteration)
                logging.info(
                    "Train set: Iteration: [{}/{}]\tAccuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}".format(
                        iteration,
                        self.args.iterations,
                        n_class_corrected,
                        total_class_samples,
                        100.0 * n_class_corrected / total_class_samples,
                        total_classification_loss / total_class_samples,
                    )
                )
                self.evaluate(iteration)

            n_class_corrected = 0
            n_domain_class_corrected = 0
            n_zs_domain_class_corrected = 0
            total_dc_loss = 0
            total_classification_loss = 0
            total_zsc_loss = 0
            total_disentangle_loss = 0
            total_samples = 0
            total_meta_samples = 0

    def evaluate(self, n_iter):
        self.zi_model.eval()
        self.zs_model.eval()
        self.classifier.eval()
        self.zs_domain_classifier.eval()
        self.domain_discriminator.eval()

        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.val_loader):
                samples, labels = samples.to(self.device), labels.to(self.device)
                di_z, ds_z = self.zi_model(samples), self.zs_model(samples)
                predicted_classes = self.classifier(di_z, ds_z)
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

        self.zi_model.train()
        self.zs_model.train()
        self.classifier.train()
        self.zs_domain_classifier.train()
        self.domain_discriminator.train()

        if self.args.val_size != 0:
            if self.val_loss_min > val_loss:
                self.val_loss_min = val_loss
                torch.save(
                    {
                        "zi_model_state_dict": self.zi_model.state_dict(),
                        "zs_model_state_dict": self.zs_model.state_dict(),
                        "classifier_state_dict": self.classifier.state_dict(),
                        "zs_domain_classifier_state_dict": self.zs_domain_classifier.state_dict(),
                        "domain_discriminator_state_dict": self.domain_discriminator.state_dict(),
                    },
                    self.checkpoint_name + ".pt",
                )
        else:
            if self.val_acc_max < val_acc:
                self.val_acc_max = val_acc
                torch.save(
                    {
                        "zi_model_state_dict": self.zi_model.state_dict(),
                        "zs_model_state_dict": self.zs_model.state_dict(),
                        "classifier_state_dict": self.classifier.state_dict(),
                        "zs_domain_classifier_state_dict": self.zs_domain_classifier.state_dict(),
                        "domain_discriminator_state_dict": self.domain_discriminator.state_dict(),
                    },
                    self.checkpoint_name + ".pt",
                )

    def test(self):
        checkpoint = torch.load(self.checkpoint_name + ".pt")
        self.zi_model.load_state_dict(checkpoint["zi_model_state_dict"])
        self.zs_model.load_state_dict(checkpoint["zs_model_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.zs_domain_classifier.load_state_dict(checkpoint["zs_domain_classifier_state_dict"])
        self.domain_discriminator.load_state_dict(checkpoint["domain_discriminator_state_dict"])
        self.zi_model.eval()
        self.zs_model.eval()
        self.classifier.eval()
        self.zs_domain_classifier.eval()
        self.domain_discriminator.eval()

        n_class_corrected = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.test_loader):
                samples, labels = samples.to(self.device), labels.to(self.device)
                di_z, ds_z = self.zi_model(samples), self.zs_model(samples)
                predicted_classes = self.classifier(di_z, ds_z)

                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()

        logging.info(
            "Test set: Accuracy: {}/{} ({:.2f}%)".format(
                n_class_corrected,
                len(self.test_loader.dataset),
                100.0 * n_class_corrected / len(self.test_loader.dataset),
            )
        )
