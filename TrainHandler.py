import argparse
from matplotlib import pyplot as plt
import os
import sys

if "-umap" in sys.argv:
    import umap
from mod.nets import *
import isy_minerl.segm.utils as utils
import copy
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import pickle
import minerl
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from matplotlib import colors
from matplotlib import cm
import cv2
from sklearn.cluster import KMeans
import logging as L
import gzip
import math
from PIL import Image, ImageDraw, ImageFont
from isy_minerl.segm.PatchEmbedder import PatchEmbedder
from sklearn.mixture import GaussianMixture as GMM
import copy
from isy_minerl.segm.td3 import *
import ffmpeg


def get_moving_avg(x, n=10):
    cumsum = np.cumsum(x)
    return (cumsum[n:] - cumsum[:-n]) / n


def make_plotbar(ph, pw, values):
    pred_mean = np.mean(values)
    plotvalues = values - np.min(values)
    max = plotvalues.max()
    plotvalues = plotvalues / ((max * 1.01) if max else 1)
    # print(plotvalues)
    plotvalues = ph - 1 - np.floor(plotvalues * ph).astype(np.int)
    plotbar = np.zeros((ph, pw + len(values) - 1, 3))
    for x, y in enumerate(plotvalues):
        plotbar[y, x + pw // 2] = (255, 255, 255)
    return plotbar


def vidwrite(fn, images, framerate=32, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n, height, width, channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                   s='{}x{}'.format(width, height), r=framerate)
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()


class Handler():
    def __init__(self, args: argparse.Namespace):
        self.args = args
        argdict = args.__dict__
        self.font = ImageFont.truetype("./isy_minerl/segm/etc/Ubuntu-R.ttf", 10)
        self.device = "cuda" if T.cuda.is_available() else "cpu"
        print("device:", self.device)
        self.models = dict()
        self.criticname = "critic"
        self.maskername = "masker"

        self.ious = 0, 0
        self.bestepoch = 0

        # INIT MODELS
        # self.critic = NewCritic(bottleneck=args.neck, chfak=args.chfak)
        # self.masker = UnetDecoder(bottleneck=args.neck, chfak=args.chfak)
        self.reset_models()
        self.models[self.criticname] = self.critic
        self.models[self.maskername] = self.masker
        self.critic_args = "-".join([f"{a}={argdict[a]}" for a in
                                     ["rewidx", "cepochs", "datamode", "datasize", "threshrew", "shift", "chfak",
                                      "dropout"] if argdict[a]])
        self.masker_args = "-".join([f"{a}={argdict[a]}" for a in
                                     ["mepochs", "L1", "L2", "inject"]
                                     if argdict[a]])

        # SETUP PATHS
        self.path = f"train/CriticMasker/{args.name}/"
        self.train_path = self.path + "train/"
        self.result_path = self.path + "results/"
        self.save_path = self.path + "saves/"
        self.data_path = "train/data/straight/"
        self.save_paths = {
            self.criticname: f"train/CriticMasker/critic-{self.critic_args}.pt",
            self.maskername: f"{self.save_path}masker-{self.masker_args}.pt"
        }

        # L.basicConfig(filename=f'./logs/{args.name}.log', format='%(asctime)s %(levelname)s %(name)s %(message)s', level=L.INFO)

    def __init__old(self, args):
        # os.environ["MINERL_DATA_ROOT"] = "data"
        # self.data = minerl.data.make('MineRLTreechopVectorObf-v0')

        self.args = args
        self.device = "cuda" if T.cuda.is_available() else "cpu"
        # self.device = "cpu"
        print("device:", self.device)
        self.models = dict()
        self.criticname = "critic" + ("+5" if args.clustercritic else "")

        # CRITIC
        """
        if args.resnet:
            self.critic = ResNetCritic().to(self.device)
            args.color = "RGB"
        else:
            if args.small:
                self.critic = SmallCritic(end=[] if not args.sigmoid else [nn.Sigmoid()],
                                      colorchs= args.clustercritic+3 if args.clustercritic else 3,
                                      chfak=args.chfak).to(self.device)
            else:
                self.critic = Critic(end=[] if not args.sigmoid else [nn.Sigmoid()],
                                          colorchs=args.clustercritic + 3 if args.clustercritic else 3,
                                          chfak=args.chfak,
                                     pool="max" if args.pool else "").to(self.device)
        """
        if not args.ucritic:
            if args.vgg:
                self.critic = VGGCritic()
            else:
                self.critic = Critic(end=[nn.Sigmoid()],
                                     colorchs=args.clustercritic + 3 if args.clustercritic else 3,
                                     chfak=args.chfak,
                                     pool="max" if args.pool else "").to(self.device)
        else:
            self.critic = nn.Identity()

        # UNET
        if args.grounded:
            self.unet = GroundedUnet().to(self.device)
        else:
            if args.small:
                self.unet = SmallUnet(chfak=args.chfak).to(self.device)
            else:
                self.unet = Unet(chfak=args.chfak, pool="max",
                                 bottleneck=args.neck, upsample=not args.transpose,
                                 pure=args.pure).to(self.device)

        # DATA ARGUMENTS
        if args.discounted:
            sarsa = "sarsa"  # "discount"
            # self.data_args = f"{sarsa}-{self.args.color}-ds{args.datasize}-cons{self.args.cons}-delay{self.args.delay}-gam{self.args.gamma}-revgam{self.args.revgamma}-chunk{self.args.chunksize}"
            self.data_args = f"{sarsa}-{self.args.color}-ds{args.datasize}-fskip{args.fskip}-chunk{self.args.chunksize}"
            if args.integrated:
                self.data_path = f"./train/{sarsa}/tree-chop/{self.data_args}/"
            else:
                self.data_path = f"./train/{sarsa}/tree-chop/{self.data_args}/"
        else:
            self.data_args = f"split-{self.args.color}-ds{args.datasize}-wait{args.wait}-delay{self.args.delay}-warmup{self.args.warmup}-chunk{self.args.chunksize}"
            if args.integrated:
                self.data_path = f"./train/segm/data/split/tree-chop/{self.data_args}/"
            else:
                self.data_path = f"./train/split/tree-chop/{self.data_args}/"
        self.arg_path = f"{'-grounded' if args.grounded else ''}{'-resnet' if args.resnet else ''}" \
                        f"{'-blur' + str(args.blur) if args.blur else ''}" \
                        f"{'-L1_' + str(args.L1) if not args.L2 else '-L2_' + str(args.L2)}" \
                        f"{'-chfak' + str(args.chfak) if args.chfak > 1 else ''}" \
                        f"{'-pooled' if args.pool else '-strided'}" + \
                        f"{'-shifted' if args.shift else ''}" + \
                        f"{'-trans' if args.transpose else ''}" + \
                        f"{'-ucritic' if args.ucritic else ''}" + \
                        f"{'-pure' if args.pure else ''}" + \
                        f"{'-copy' if args.copy else ''}" + \
                        f"{'-navneg' if args.navneg else ''}" + \
                        f"{'-neck' + str(args.neck)}" + \
                        f"{'-static' if args.staticnorm else ''}" + \
                        (f"-live{args.clossfak}" if args.live else "") + \
                        f"{'-inject' if args.inject else ''}" + \
                        "/"
        # self.data_args
        print("model path:", self.arg_path)

        # RESULT PATHS
        if args.integrated:
            self.result_path = f"./train/segm/results/Critic/" + args.name + self.arg_path
        else:
            self.result_path = f"./train/NewCritic/" + args.name + self.arg_path
        print("viz path:", self.result_path)

        # EMBED PATHS AND ARGS
        self.embed_data_args = f"cl{args.embed_cluster}-dim{args.embed_dim}-ds{args.embed_train_samples}-" \
                               + f"dl{args.delay}-th{args.embed_pos_threshold}-pw{args.embed_patch_width}" \
                               + f"{'-hue' if args.hue else ('-hsv' if args.hsv else '-hs')}" \
                               + f"-{self.args.embed_norm}"
        self.embed_data_args_specific = self.embed_data_args
        if args.final or args.integrated:
            self.unetname = f"unet"
            self.embed_data_path = f"./train/"
            self.embed_data_args = "embed-data"
            self.save_path = f"./train/"
            self.font = ImageFont.truetype("./isy_minerl/segm/etc/Ubuntu-R.ttf", 10)
        else:
            self.unetname = f"unet-l2_{args.L2}-l1_{args.L1}"
            self.save_path = self.result_path
            self.font = ImageFont.truetype("./isy_minerl/segm/etc/Ubuntu-R.ttf", 10)
            self.embed_data_path = f"train/patchembed/"

        if args.water:
            self.water_name = "water_disc"
            self.water_disc = Critic(end=[], colorchs=3, chfak=args.chfak).to(self.device)
            self.models[self.water_name] = self.water_disc
        self.models[self.criticname] = self.critic
        self.models[self.unetname] = self.unet

        # L.basicConfig(filename=f'./logs/{args.name}.log', format='%(asctime)s %(levelname)s %(name)s %(message)s', level=L.INFO)

    def reset_models(self):
        args = self.args
        self.critic = NewCritic(bottleneck=args.neck, chfak=args.chfak, dropout=args.dropout)
        self.masker = UnetDecoder(bottleneck=args.neck, chfak=args.chfak)
        if self.args.separate:
            self.sepcrit = NewCritic(bottleneck=args.neck, chfak=args.chfak, dropout=args.dropout)

    def load_data_old(self, batch_size=64):
        self.batchsize = 64
        args = self.args
        wait = self.args.wait
        data_size = self.args.datasize
        test_size = self.args.testsize
        data_path = self.data_path
        file_name = "data.pickle"
        # data_collector = self.collect_discounted_dataset if self.args.discounted else self.collect_split_dataset
        data_collector = self.collect_data()

        if args.proof:
            pics = datasets.ImageFolder("debug/proof-dataset", transform=transforms.ToTensor())
            X = [rgb_to_hsv(pics.__getitem__(i)[0].permute(1, 2, 0)) for i in range(pics.__len__())]
            Y = [(pics.__getitem__(i)[1], 0, 0, 0) for i in range(pics.__len__())]
            X = (np.stack(X) * 255).astype(np.uint8)
            Y = np.array(Y)
            self.X, self.Y = np.stack([X, X], axis=1), np.stack([Y, Y], axis=1)
            self.XX, self.YY = self.X, self.Y
            print(self.X.shape, self.Y.shape)
            # print(X)
            self.dataloader = T.utils.data.DataLoader(
                T.utils.data.TensorDataset(T.from_numpy(self.X)
                                           , T.from_numpy(self.Y),
                                           T.arange(self.X.shape[0], dtype=T.uint8)),
                batch_size=batch_size, shuffle=True)
            self.testdataloader = T.utils.data.DataLoader(
                T.utils.data.TensorDataset(T.from_numpy(self.XX),
                                           T.from_numpy(self.YY),
                                           T.arange(self.XX.shape[0],
                                                    dtype=T.uint8)),
                batch_size=self.args.testsize, shuffle=False)
            self.testsize = self.XX.shape[0]
            self.trainsize = self.X.shape[0]
            return

        if args.blur:
            # blur = nn.AvgPool2d(args.blur, 1, 1)
            if args.blur == 3:
                blurkernel = T.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]] * 1] * 3).float() / 16
                blur = lambda x: F.conv2d(x, blurkernel, padding=1, groups=3)
            if args.blur == 5:
                blurkernel = T.tensor([[[[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4],
                                         [1, 4, 6, 4, 1]]] * 1] * 3).float() / 256
                blur = lambda x: F.conv2d(x, blurkernel, padding=2, groups=3)

        print("loading data:", data_path)
        # TRAIN
        if not os.path.exists(data_path + file_name):
            print("train set...")
            os.makedirs(data_path, exist_ok=True)
            data_collector(data_path + file_name, size=data_size, datadir=data_path)
        # TEST
        testfilename = data_path + "test-" + file_name
        if not os.path.exists(testfilename):
            print("collecting test set...")
            os.makedirs(data_path, exist_ok=True)
            data_collector(testfilename, size=test_size, datadir=data_path, test=1)

        if args.clippify:
            utils.data_set_to_vid(testfilename, HSV=args.color == "HSV")

        # TRAIN data
        with gzip.open(data_path + file_name, "rb") as fp:
            self.X, self.Y = pickle.load(fp)
            if args.blur:
                self.X = np.swapaxes(self.X, 1, -1)
                self.X = blur(T.from_numpy(self.X).float()).numpy().astype(np.uint8)
                self.X = np.swapaxes(self.X, 1, -1)
                print(self.X.shape)
        print(f"loaded train set with {len(self.X)}")
        # TEST data
        with gzip.open(testfilename, "rb") as fp:
            self.XX, self.YY = pickle.load(fp)
            if args.blur:
                self.XX = np.swapaxes(self.XX, 1, -1)
                self.XX = blur(T.from_numpy(self.XX).float()).numpy().astype(np.uint8)
                self.XX = np.swapaxes(self.XX, 1, -1)
                print(self.XX.shape)
        print(f"loaded test set with {len(self.XX)}")

        if args.withnav or args.patchembed:
            # LOAD NAV DATA TRAIN
            navdatadir = f"./train/navigate/train/{args.navdatasize}"
            if not os.path.exists(navdatadir + "data.pickle"):
                self.collect_navigation_dataset(datadir=navdatadir, nperchunk=10)
            with gzip.open(navdatadir + "data.pickle", 'rb') as fp:
                self.NX, self.NY = pickle.load(fp)
                # self.NY = NY[:, 0]
                # self.NX = NX / 255
            print("loaded navigation train data:", self.NX.shape, self.NY.shape)
            self.NX = np.stack([self.NX] * 10, axis=1)
            self.NY = np.repeat(np.stack([self.NY] * 10, axis=1), self.Y.shape[2], axis=2)
            print("after frameskip refactor:", self.NX.shape, self.NY.shape)
            self.X = np.concatenate((self.X, self.NX), axis=0)
            self.Y = np.concatenate((self.Y, self.NY), axis=0)

            navdatadir = f"./train/navigate/test/{args.testsize}"
            if not os.path.exists(navdatadir + "data.pickle"):
                self.collect_navigation_dataset(datadir=navdatadir, nperchunk=50)
            with gzip.open(navdatadir + "data.pickle", 'rb') as fp:
                self.NXX, self.NYY = pickle.load(fp)
                # self.NYY = NY[:, 0]
                # self.NXX = NX / 255
            print("loaded navigation test data:", self.NXX.shape, self.NYY.shape)

            print(self.Y.shape, self.X.shape)
            print("reward frames in dataset:", np.sum(self.Y[:, 0, 0]))
            # self.highrewmask = self.Y[:, args.rewidx]>args.high_rew_thresh
            # self.highrewmask = self.Y[:, 0, args.rewidx]>args.high_rew_thresh
            # self.HX = self.X[self.highrewmask]
            # self.HY = self.Y[self.highrewmask, args.rewidx]
            # self.HY = self.Y[self.highrewmask, :, args.rewidx]
            # self.highrewloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.from_numpy(self.HX),
            #    T.from_numpy(self.HY), T.arange(self.HX.shape[0], dtype=T.uint8)),
            #    batch_size=batch_size//2, shuffle=True)

            # self.lowrewmask = self.Y[:, 0, args.rewidx] < args.low_rew_thresh
            # self.LX = self.X[self.lowrewmask]
            # self.LY = self.Y[self.lowrewmask, :, args.rewidx]

        self.dataloader = T.utils.data.DataLoader(
            T.utils.data.TensorDataset(T.from_numpy(self.X)
                                       , T.from_numpy(self.Y),
                                       T.arange(self.X.shape[0], dtype=T.uint8)),
            batch_size=batch_size, shuffle=True)
        self.testdataloader = T.utils.data.DataLoader(
            T.utils.data.TensorDataset(T.from_numpy(self.XX),
                                       T.from_numpy(self.YY),
                                       T.arange(self.XX.shape[0],
                                                dtype=T.uint8)),
            batch_size=self.args.testsize, shuffle=False)
        self.testsize = self.XX.shape[0]
        self.trainsize = self.X.shape[0]

        if args.water:
            self.WX = self.collect_water()

    def load_data(self, batch_size=64):
        args = self.args
        X, Y, I = self.collect_data()
        train = slice(0, -args.testsize)
        test = slice(-args.testsize, None)
        self.X, self.Y, self.I = X[train], Y[:, train], I[train]
        self.XX, self.YY, self.II = X[test], Y[:, test], I[test]
        if args.threshrew:
            self.Y = (self.Y > args.threshrew).astype(np.float)
            self.YY = (self.YY > args.threshrew).astype(np.float)

        print("dataset shapes", X.shape, Y.shape, self.X.shape, self.Y.shape)
        self.train_loader = T.utils.data.DataLoader(
            T.utils.data.TensorDataset(T.from_numpy(self.X),
                                       T.from_numpy(self.Y).t(),
                                       T.arange(self.X.shape[0], dtype=T.int32)),
            batch_size=batch_size, shuffle=True)
        # self.test_loader = T.utils.data.DataLoader(
        #    T.utils.data.TensorDataset(T.from_numpy(self.XX),
        #                               T.from_numpy(self.YY),
        #                               T.arange(self.XX.shape[0], dtype=T.int32)),
        #        batch_size=batch_size, shuffle=True)

    def load_models(self, modelnames=[]):
        if not modelnames:
            modelnames = self.models.keys()
        for model in modelnames:
            save_path = self.save_paths[model]
            if not os.path.exists(save_path):
                print(f"{save_path} not found")
                return False
            print("loading:", save_path)
            self.models[model].load_state_dict(T.load(save_path, map_location=T.device(self.device)))
        return True

    def save_models(self, modelnames=[]):
        os.makedirs(self.save_path, exist_ok=True)
        if not modelnames:
            modelnames = self.models.keys()
        for model in modelnames:
            save_path = self.save_paths[model]
            print("saving:", save_path)
            T.save(self.models[model].state_dict(), save_path)

    def critic_pipe_old(self, mode="train", test=0):
        args = self.args
        testf = mode == "test"
        trainf = mode == "train"
        loader = self.train_loader

        if args.cload and self.load_models([self.criticname]):
            print("loaded critic, no new training")
            return

        # Setup save path and Logger
        result_path = self.path + "critic/"
        os.makedirs(result_path, exist_ok=True)
        log_file = open(result_path + "log.txt", "w")
        log_file.write(f"{self.args}\n\n")

        llog = []

        critic = self.critic
        critic = critic.to(self.device)
        opti = T.optim.Adam(critic.parameters())
        # Epoch and Batch Loops
        for epoch in range(int(testf) or self.args.cepochs):
            for b_idx, (X, Y, I) in enumerate(loader):
                # SHIFT
                if args.shift:
                    X = self.shift_batch(X)

                # FORMATING
                XP = X.permute(0, 3, 1, 2).float().to(self.device) / 255.0
                Y = Y[:, args.rewidx].float().to(self.device)
                pred = critic(XP).squeeze()
                if args.threshrew:
                    loss = F.binary_cross_entropy(pred, Y)
                else:
                    loss = F.mse_loss(pred, Y)
                print(f"critic e{epoch + 1} b{b_idx}", loss.item(), end="\r")
                opti.zero_grad()
                loss.backward()
                opti.step()
                llog.append(loss.item())
                # log_file.write(log_msg+"\n")

                # VIZ -----------------------------------
                if not b_idx % 100:  # VISUALIZE
                    vizs = []

                    if False:
                        order1 = YY.argsort(descending=True)
                        order2 = Y.argsort(descending=True)
                    # L.info(f"critic e{epoch} b{b_idx} loss: {loss.item()}")
                    viz = X.cpu().numpy()
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)

                    viz = np.concatenate(vizs, axis=0)
                    img = Image.fromarray(np.uint8(viz))
                    draw = ImageDraw.Draw(img)
                    for i, value in enumerate(Y.tolist()):
                        x, y = int(i * img.width / len(X)), 1
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                    for i, value in enumerate(pred.tolist()):
                        x, y = int(i * img.width / len(X)), int(1 + img.height / 2)
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

                    # plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path + f"e{epoch}_b{b_idx}.png")

            if not (epoch + 1) % args.saveevery:
                self.save_models(modelnames=[self.criticname])

            plt.clf()
            plt.plot(get_moving_avg(llog, 30), label="Train Loss")
            plt.ylim(0, plt.ylim()[1])
            plt.legend()
            plt.savefig(result_path + "_loss.png")

            if trainf:
                # critic.sched.step()
                pass

        print()

    def train_water_discriminator(self):
        args = self.args
        batchsize = 32

        # Setup save path and Logger
        result_path = self.result_path + "water/"
        os.makedirs(result_path, exist_ok=True)
        log_file = open(result_path + "log.txt", "w")
        log_file.write(f"{self.args}\n\n")

        # setup models and opti
        disc = self.water_disc
        opti = T.optim.Adam(disc.parameters())
        n_batches = int(np.ceil(len(self.WX) / batchsize))

        # Epoch and Batch Loops
        for epoch in range(args.epochs):
            for b_idx in range(n_batches):
                # break early
                wxidxs = np.random.choice(np.arange(len(self.WX)), batchsize)
                txidxs = np.random.choice(np.arange(len(self.X)), batchsize)
                WX = T.from_numpy(self.WX[wxidxs])
                TX = T.from_numpy(self.X[txidxs])

                # combine pos and neg
                Ybinary = T.cat((T.ones(batchsize), T.zeros(batchsize)))
                X = T.cat((WX, TX), dim=0)

                pred = disc(X.permute(0, 3, 1, 2).float()).squeeze()

                YY = T.sigmoid(pred)

                loss = F.binary_cross_entropy_with_logits(pred, Ybinary)
                loss_string = f"water disc e{epoch} b{b_idx} {loss.item()}"
                log_file.writelines([loss_string])
                if not b_idx % 10:
                    print(loss_string, end="\n")
                opti.zero_grad()
                loss.backward()
                opti.step()
                # log_file.write(log_msg+"\n")

                # VIZ -----------------------------------
                if not b_idx % 100:  # VISUALIZE
                    vizs = []
                    vizX = hsv_to_rgb(X.numpy() / 255) if self.args.color == "HSV" else X.numpy() / 255

                    vizs.append(np.concatenate(vizX, axis=1))

                    viz = np.concatenate(vizs, axis=0)
                    img = Image.fromarray(np.uint8(255 * viz))
                    draw = ImageDraw.Draw(img)
                    for i, value in enumerate(Ybinary.tolist()):
                        x, y = int(i * img.width / len(Ybinary)), 1
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                    for i, value in enumerate(YY.tolist()):
                        x, y = int(i * img.width / len(YY)), 10
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

                    # plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path + f"e{epoch}_b{b_idx}.png")

        self.save_models(modelnames=[self.water_name])
        print()

    def contrastive_critic_pipe(self, mode="train", test=0):
        args = self.args
        testf = mode == "test"
        trainf = mode == "train"
        loader = self.dataloader if trainf else self.testdataloader
        etha = args.etha
        fskip = 1

        if args.clustercritic:
            with gzip.open(self.data_path + f"{mode}-{str(self.args.clustercritic)}-cluster", 'rb') as fp:
                channels = pickle.load(fp)

        # Setup save path and Logger
        result_path = self.path + "critic/"
        os.makedirs(result_path, exist_ok=True)
        llog = []
        slog = []
        # log_file = open(result_path+"log.txt", "w")
        # log_file.write(f"{self.args}\n\n")

        # setup models and opti
        if args.ucritic:
            critic = lambda x: self.unet.forward(x, critic=True)
            opti = T.optim.Adam(self.unet.parameters(recurse=True), lr=3e-4)
        else:
            if args.vgg:
                critic = self.critic
                opti = T.optim.Adam(critic.head.parameters())
            else:
                critic = self.critic
                opti = T.optim.Adam(critic.parameters())

        """    
        if args.resnet:
            opti = T.optim.Adam(critic.head.parameters())
        else:
            opti = T.optim.Adam(critic.parameters())
        """

        # setup window slicing
        w = self.X.shape[2]
        ypad = (w - int(args.window.split('-')[1])) // 2
        xpad = (w - int(args.window.split('-')[0])) // 2
        yslice = slice(ypad, -ypad) if (xpad or ypad) else slice(None)
        xslice = slice(xpad, -xpad) if (xpad or ypad) else slice(None)
        print("xpad", xpad, "ypad", ypad)

        # Setup negs
        """
        if args.navneg:
            negs = self.NX
        else:
            negs = np.concatenate((self.LX, self.NX), axis=0)
            negsY = np.concatenate((self.LY, np.zeros(len(self.NY))), axis=0)
        """

        # Epoch and Batch Loops
        for epoch in range(args.epochs):
            for b_idx, (PX, PY, I) in enumerate(loader):
                PX = PX / 255.0
                PY = PY[:, :, args.rewidx]
                PX, PX2 = PX[:, 0], PX[:, fskip]
                PY, PY2 = PY[:, 0], PY[:, :fskip].sum(dim=1)
                # combine pos and neg
                # neg_idxs = np.random.choice(np.arange(len(negs)), len(PX))
                # NX = T.from_numpy(negs[neg_idxs])
                # Y = T.cat((PY, T.from_numpy(negsY[neg_idxs]))).float()
                # Ybinary = T.cat((T.ones(len(PX)), T.zeros(len(PX))))
                # X = T.cat((PX, NX), dim=0)

                # format data
                # XP = X.permute(0,3,1,2).float().to(self.device)
                XW = PX.permute(0, 3, 1, 2).float().to(self.device)
                XW2 = PX2.permute(0, 3, 1, 2).float().to(self.device)
                if args.shift:
                    XW = self.shift_batch(XW)
                    XW2 = self.shift_batch(XW2)

                # Shift test
                if False:
                    with T.no_grad():
                        XSHIFT = self.shift_batch(XW)
                        shiftpred = critic(XSHIFT).squeeze()
                        YSHIFT = T.sigmoid(shiftpred)
                        shift_loss = F.binary_cross_entropy_with_logits(shiftpred, Ybinary)
                        slog.append(shift_loss.item())

                pred = critic(XW).squeeze()
                pred2 = critic(XW2).squeeze()
                YY = pred

                # loss = F.binary_cross_entropy_with_logits(pred, Ybinary)
                V_target = PY.float() + etha * pred2.detach()
                loss = F.mse_loss(pred, V_target)
                loss_string = f"critic e{epoch} b{b_idx} {loss.item()}"
                # log_file.writelines([loss_string])
                llog.append(loss.item())
                if not b_idx % 10:
                    print(loss_string, end="\n")
                opti.zero_grad()
                loss.backward()
                opti.step()
                # log_file.write(log_msg+"\n")

                # VIZ -----------------------------------
                if not b_idx % args.visevery:  # VISUALIZE
                    if args.vgg:
                        salX = T.from_numpy(hsv_to_rgb(XW.permute(0, 2, 3, 1)))
                        salX.requires_grad = True
                        acti = critic(salX, hsv=False)
                    else:
                        salX = XW.permute(0, 2, 3, 1)
                        salX.requires_grad = True
                        acti = critic(salX.permute(0, 3, 1, 2))
                    acti.sum().backward()
                    # print("salience sum", salX.grad.sum())
                    salX = salX.grad.sum(dim=-1)
                    salXpos = salX * (salX > 0)
                    salXneg = (salX * (salX <= 0)).abs()
                    # print(salX.shape)
                    salXpos = salXpos / (salXpos.flatten(1).max(dim=1)[0][:, None, None] + 0.00001)
                    salXneg = salXneg / (salXneg.flatten(1).max(dim=1)[0][:, None, None] + 0.00001)
                    vizs = []
                    vizX = hsv_to_rgb(PX.numpy()) if self.args.color == "HSV" else X.numpy()
                    vizSalpos = T.stack([salXpos] * 3, dim=-1).numpy()
                    vizSalneg = T.stack([salXneg] * 3, dim=-1).numpy()
                    vizX2 = hsv_to_rgb(PX2.numpy()) if self.args.color == "HSV" else X.numpy()

                    vizs.append(np.concatenate(vizX, axis=1))
                    vizs.append(np.concatenate(vizSalpos * vizX, axis=1))
                    vizs.append(np.concatenate(vizSalneg * vizX, axis=1))
                    vizs.append(np.concatenate(vizX2, axis=1))
                    if False:
                        XSHIFT = XSHIFT.permute(0, 2, 3, 1)
                        vizShift = hsv_to_rgb(XSHIFT.numpy() / 255) if \
                            self.args.color == "HSV" else XSHIFT.numpy() / 255
                        vizs.append(np.concatenate(vizShift, axis=1))

                    viz = np.concatenate(vizs, axis=0)
                    img = Image.fromarray(np.uint8(255 * viz))
                    draw = ImageDraw.Draw(img)
                    for i, value in enumerate(PY.tolist()):
                        x, y = int(i * img.width / len(PY)), 1
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255))
                    for i, value in enumerate(pred.tolist()):
                        x, y = int(i * img.width / len(PY)), 10
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255))
                    for i, value in enumerate(PY2.tolist()):
                        x, y = int(i * img.width / len(PY)), 1 + (len(vizs) - 1) * XW.shape[2]
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255))
                    for i, value in enumerate(pred2.tolist()):
                        x, y = int(i * img.width / len(PY)), 10 + (len(vizs) - 1) * XW.shape[2]
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255))
                    if False:
                        for i, value in enumerate(YSHIFT.tolist()):
                            x, y = int(i * img.width / len(YSHIFT)), 1 + 2 * XW.shape[2]
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

                    # plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path + f"e{epoch}_b{b_idx}.png")

            plt.clf()
            if not args.shift:
                avg = get_moving_avg(llog, 30)
                plt.plot(avg, label="Train Loss")
                # plt.plot(get_moving_avg(slog, 30), label="Shift Loss")
            else:
                plt.plot(get_moving_avg(llog, 30), label="Train Loss")
            plt.ylim(0, plt.ylim()[1])
            plt.legend()
            plt.savefig(result_path + "_loss.png")

            if not (epoch + 1) % args.saveevery:
                self.save_models(modelnames=[self.criticname, self.unetname])

            if trainf:
                # critic.sched.step()
                pass

        print()

    def extract_contrastive_data(self):
        args = self.args
        critic = self.critic.to(self.device).eval()
        batchsize = 128

        if args.critic or args.cload:
            preds = []
            for bidx in range(math.ceil(len(self.X) / batchsize)):
                print("searching dataset for high and low values...", bidx / (len(self.X) / batchsize), end="\r")
                with T.no_grad():
                    batch = T.from_numpy(self.X[bidx * batchsize:(bidx + 1) * batchsize]).permute(0, 3, 1,
                                                                                                  2).float().to(
                        self.device) / 255.0
                    pred = critic(batch).squeeze()
                preds.append(pred)
            preds = T.cat(preds, dim=0).cpu()

            # PLOT HISTOGRAMM
            idx = args.rewidx
            plt.clf()
            plt.hist(preds.numpy())
            print("saving histogramm", self.path + f"pred_idx{idx}_hist.png")
            plt.savefig(self.path + f"pred_idx{idx}_hist.png")
            plt.clf()
            plt.hist(self.Y[args.rewidx])
            print("saving histogramm", self.path + f"GT_idx{idx}_hist.png")
            plt.savefig(self.path + f"GT_idx{idx}_hist.png")

            positives = preds > args.high_rew_thresh
            negatives = preds < args.low_rew_thresh
        else:
            print("no critic provided -> using random pos and neg frames")
            positives = T.rand(len(self.X)) > 0.5
            negatives = positives == False
            preds = T.cat((positives, negatives), dim=0)

        with open(self.path +
                  f"{positives.sum()}>{args.high_rew_thresh}__{negatives.sum()}<{args.low_rew_thresh}.txt", "w") as fp:
            fp.write("")

        print(f"\nallframes {len(preds)}  frames>{args.high_rew_thresh}", positives.sum(),
              f" frames<{args.low_rew_thresh}", negatives.sum())

        assert (sum(positives) >= 500 and sum(negatives) >= 500)
        # print("LENGTH", sum(positives))
        self.Xpos = self.X[positives]
        self.Ypos = self.Y[:, positives]
        self.Xneg = self.X[negatives]
        self.Yneg = self.Y[:, negatives]

        """
        newpreds = []
        for bidx in range(math.ceil(len(self.Xpos) / batchsize)):
           print("searching dataset for high and low values...", bidx / (len(self.Xpos) / batchsize), end="\r")
           with T.no_grad():
               batch = T.from_numpy(self.Xpos[bidx * batchsize:(bidx + 1) * batchsize]).permute(0, 3, 1, 2).float().to(
                   self.device) / 255.0
               pred = critic(batch).squeeze()
           newpreds.append(pred)
        newpreds = T.cat(newpreds, dim=0).cpu()"""
        # assert (preds[positives] == newpreds).all()

        print("n positives", self.Xpos.shape[0])
        print("positives:", self.Ypos)
        print("HIGH REW THRESH", args.high_rew_thresh)
        assert (preds[positives].mean()) > args.high_rew_thresh
        # assert np.mean(self.Ypos[args.rewidx]) > args.high_rew_thresh
        print("n negatives", self.Xneg.shape[0])

        self.XposIdxs = np.arange(len(self.Xpos))
        self.XnegIdxs = np.arange(len(self.Xneg))
        self.ContrastIdxs = np.arange(len(self.Xneg))
        self.contrastive_batchsize = 32
        self.get_contrastive_idxs = lambda: (np.random.choice(self.XposIdxs, self.contrastive_batchsize),
                                             np.random.choice(self.XnegIdxs, self.contrastive_batchsize),
                                             np.random.choice(self.ContrastIdxs, 2 * self.contrastive_batchsize))

    def segmentation_training(self):
        args = self.args
        self.extract_contrastive_data()
        # Setup save path and Logger
        train_path = self.path + "segment/"
        os.makedirs(train_path, exist_ok=True)
        log_file = open(train_path + "log.txt", "w")
        log_file.write(f"{self.args}\n\n")
        log = []

        # SETUP MODELS
        critic = self.critic.to(self.device).train()
        masker = self.masker.to(self.device).train()
        if args.separate:
            sepcrit = self.sepcrit.to(self.device).train()

        if args.live:
            opti = T.optim.Adam(
                chain(critic.parameters(), masker.parameters(), self.sepcrit.parameters() if args.separate else []))
        else:
            opti = T.optim.Adam(chain(masker.parameters(), self.sepcrit.parameters() if args.separate else []))

        # Epoch and Batch Loops
        for epoch in range(self.args.mepochs):
            # BATCHING THROUGH DATA
            for b_idx in range(math.ceil(self.Xpos.shape[0] / self.contrastive_batchsize)):
                loss_string = f"e{epoch} b{b_idx}"
                Hidx, Lidx, Cidx = self.get_contrastive_idxs()
                HX = T.from_numpy(self.Xpos[Hidx])
                HY = T.from_numpy(self.Ypos[args.rewidx, Hidx])
                LX = T.from_numpy(self.Xneg[Lidx])
                LY = T.from_numpy(self.Yneg[args.rewidx, Lidx])

                X = T.cat((HX, LX), dim=0)
                Y = T.cat((HY, LY), dim=0)
                CX = T.from_numpy(self.Xneg[Cidx])
                CY = T.from_numpy(self.Yneg[args.rewidx, Cidx])

                if args.shift:
                    X = self.shift_batch(X)
                    # CX = self.shift_batch(CX)

                # FORMATING
                A = X.permute(0, 3, 1, 2).float().to(self.device) / 255.0
                B = CX.permute(0, 3, 1, 2).float().to(self.device) / 255.0

                # get critic values
                pred, embeds = critic(A, collect=True)
                negpred = critic(B)
                pred = pred.squeeze()
                negpred = negpred.squeeze().detach()
                # print("pred negpred", pred.shape, negpred.shape)

                # pos_fails = 1 - (Y.view(pred.shape) - pred).abs()
                # neg_fails = 1 - (LY[negatives] - negpred).abs()
                # critic_fail_mask = 1  # (pos_fails * neg_fails).squeeze().detach()

                # LOSSES:
                loss = 0

                # LIVE CRITIC?
                if args.live:
                    # critic_loss = F.binary_cross_entropy_with_logits(rawpred, Ybinary)
                    if args.threshrew:
                        critic_loss = F.binary_cross_entropy(pred, Y.to(self.device).float())
                    else:
                        critic_loss = F.mse_loss(pred, Y.to(self.device).float())
                    loss = loss + args.lfak * critic_loss
                    loss_string += f"    live-critic {critic_loss.item()}"
                # pred = pred.detach()

                # SEGMENTATION
                if args.separate:
                    _, embeds = sepcrit(A, collect=True)
                Z = masker(A, embeds)
                merge_critic = critic

                # replace mask
                replaced = A * (1 - Z) + Z * B
                replacevalue = merge_critic(replaced).squeeze()
                # replacevalue = merge_critic(replaced).squeeze() * critic_fail_mask
                # replaceloss = F.binary_cross_entropy_with_logits(replacevalue, negpred.detach())
                # replacevalue = T.sigmoid(replacevalue)
                replaceloss = F.mse_loss(replacevalue, negpred.detach())
                loss = loss + replaceloss
                loss_string += f"   replace: {replaceloss.item()}"

                # inject mask
                if args.inject:
                    injected = B * (1 - Z) + Z * A
                    injectvalue = merge_critic(injected).squeeze()
                    # injectvalue = merge_critic(injected).squeeze() * critic_fail_mask
                    # injectloss = F.binary_cross_entropy_with_logits(injectvalue, pred.detach())
                    # injectvalue = T.sigmoid(injectvalue)
                    injectloss = F.mse_loss(injectvalue, pred.detach())
                    loss = loss + injectloss
                    loss_string += f"   inject: {injectloss.item()}"

                if args.staticnorm:
                    valuefak = 1  # * critic_fail_mask.view(-1, 1, 1, 1)
                else:
                    valuefak = 1 - pred.detach().view(-1, 1, 1, 1)  # * critic_fail_mask.view(-1, 1, 1, 1)
                    # valuefak = 1 - Ybinary.view(-1, 1, 1, 1)

                if args.L1:
                    normloss = args.L1 * F.l1_loss(valuefak * Z, T.zeros_like(Z))
                    # normloss = args.L1 * (valuefak*Z).mean()
                    loss = loss + normloss
                    loss_string += f"   L1: {normloss.item()}"
                if args.L2:
                    normloss = args.L2 * F.mse_loss(valuefak * Z, T.zeros_like(Z))
                    loss = loss + normloss
                    loss_string += f"   L2: {normloss.item()}"
                if False and args.distnorm:
                    mask = Z.cpu().detach()
                    w = X.shape[1]
                    b = X.shape[0]
                    xs = T.arange(w).repeat((b, 1, w, 1)).float() / w
                    ys = T.arange(w).repeat((b, 1, w, 1)).transpose(2, 3).float() / w
                    # print(xs[0], xs[1], xs.shape, ys.shape, mask.shape)
                    xvote = (xs * mask).flatten(start_dim=-2).mean(dim=-1).squeeze().view(b, 1, 1, 1)
                    yvote = (ys * mask).flatten(start_dim=-2).mean(dim=-1).squeeze().view(b, 1, 1, 1)
                    # print(xs.shape, xvote.shape)
                    xs -= xvote  # X Distance
                    ys -= yvote  # Y Distance
                    dist = (xs.pow(2) + xs.pow(2)).pow(0.5)
                    target = mask - dist
                    target[target < 0] = 0
                    distloss = 5 * F.mse_loss(Z, target.to(self.device))
                    loss = loss + distloss
                    loss_string = f"dist-norm: {distloss.item()}   " + loss_string

                # total_var = (Z[:,:,1:]-Z[:,:,:-1]).mean().abs() + \
                #            (Z[:,:,:,1:]-Z[:,:,:,:-1]).mean().abs()
                # loss = loss + 0.0001* total_var
                # loss_string += f"   total var: {round(total_var.item(), 10)}"

                if not b_idx % 10:
                    print((loss_string), end='\r')
                log.append((replaceloss.item(),
                            injectloss.item() if args.inject else 0,
                            normloss.item() if args.L1 or args.L2 else 0,
                            critic_loss.item() if args.live else 0))

                opti.zero_grad()
                loss.backward()
                opti.step()

                # VIZ -----------------------------------
                if not b_idx % args.visevery:  # VISUALIZE
                    vizs = []
                    # A = XWPAD
                    # B = XWPAD[negatives]
                    A = A.cpu().detach().permute(0, 2, 3, 1)
                    B = B.cpu().detach().permute(0, 2, 3, 1)
                    Z = Z.cpu().detach().permute(0, 2, 3, 1)
                    replaced = A * (1 - Z) + Z * B
                    injected = B * (1 - Z) + Z * A
                    # print(Z.shape, A.shape, B.shape, replaced.shape, injected.shape)
                    viz = A.numpy()
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(np.zeros_like(viz))
                    vizs.append(np.zeros_like(viz))
                    vizs.append(viz)
                    viz = B.numpy()
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)
                    viz = replaced.numpy()
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)
                    viz = injected.numpy()
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)
                    viz = T.cat((Z, Z, Z), dim=-1).cpu().numpy()
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)

                    viz = np.concatenate(vizs, axis=0)
                    # print("maxes", replaced.max(), injected.max(), A.max(), B.max(), Z.max(), Z.min())
                    img = Image.fromarray(np.uint8(255 * viz))
                    if True:
                        adder = 12
                        draw = ImageDraw.Draw(img)
                        for i, value in enumerate(Y.tolist()):
                            x, y = int(i * img.width / len(pred)), 0
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                        for i, value in enumerate(pred.tolist()):
                            x, y = int(i * img.width / len(pred)), adder
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                        for i, value in enumerate(negpred.tolist()):
                            x, y = int(i * img.width / len(Y)), 2 * adder
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                        for i, value in enumerate(replacevalue.tolist()):
                            x, y = int(i * img.width / len(Y)), 3 * adder
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                        if args.inject:
                            for i, value in enumerate(injectvalue.tolist()):
                                x, y = int(i * img.width / len(Y)), 4 * adder
                                draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

                        """
                        for i, value in enumerate(pos_fails.tolist()):
                            x, y = int(i * img.width / len(Y)), 5 * adder
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                        for i, value in enumerate(neg_fails.tolist()):
                            x, y = int(i * img.width / len(Y)), 6 * adder
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                        for i, value in enumerate(critic_fail_mask.tolist()):
                            x, y = int(i * img.width / len(Y)), 49
                            draw.text((x, y), str(round(value, 1)), fill=(255, 255, 255), font=self.font)"""

                    # plt.imsave(result_train_path+f"e{epoch}_b{b_idx}.png", viz)
                    # print("saving intermediate results in", f"e{epoch}_b{b_idx}.png")
                    img.save(train_path + f"e{epoch}_b{b_idx}.png")

                if False:  # VISUALIZE
                    viz1 = hsv_to_rgb(X.numpy() / 255) if self.args.color == "HSV" else A.numpy() / 255
                    viz1 = np.concatenate(viz1, axis=1)
                    Z = unet(XP)
                    Z = Z.detach().permute(0, 2, 3, 1)
                    seg = X.float() / 255
                    seg[:, :, :, 1] = Z.squeeze()
                    viz2 = hsv_to_rgb(seg.numpy()) if self.args.color == "HSV" else seg.numpy()
                    viz2 = np.concatenate(viz2, axis=1)
                    viz4 = T.cat((Z, Z, Z), dim=-1).cpu().numpy()
                    viz4 = np.concatenate(viz4, axis=1)

                    viz = np.concatenate((viz1, viz2, viz4), axis=0)
                    img = Image.fromarray(np.uint8(255 * viz))
                    draw = ImageDraw.Draw(img)
                    YY = T.sigmoid(critic(XP)).squeeze()
                    for i, value in enumerate(YY.tolist()):
                        x, y = int(i * img.width / len(YY)), 1
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

                    # plt.imsave(result_train_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(train_path + f"e{epoch}_b{b_idx}.png")

            # PLOT LOSS
            plt.clf()
            llog = np.array(log)
            loss_names = ["replace", "inject", "norm", "live-critic"]
            for i in range(len(llog[0])):
                plt.plot(get_moving_avg(llog[:, i], 30), label=loss_names[i])
            plt.legend()
            plt.savefig(train_path + "_loss.png")

            # SAVE MODEL
            if not (epoch + 1) % args.saveevery:
                self.save_models(modelnames=[self.maskername])

            ious = self.eval()
            if ious[0] > self.ious[0]:
                self.ious = ious
                self.bestepoch = epoch

        if False:
            log = np.array(log)
            end = len(log) // 10
            log1 = log[:10 * end, 0]
            log1 = log1.reshape((-1, 10))
            log1 = log1.mean(axis=-1)
            plt.plot(log1, label="value loss")
            log2 = log[:10 * end, 1]
            log2 = log2.reshape((-1, 10))
            log2 = log2.mean(axis=-1)
            plt.plot(log2, label="L2 norm")
            plt.legend()
            plt.savefig(result_path + f"loss.png")
        print()
        self.save_models(modelnames=[self.maskername])

    def contrastive_merge_segmentation_old(self, mode="train", test=0):
        args = self.args
        testf = mode == "test"
        trainf = mode == "train"
        unet = self.unet
        if args.freeze:
            opti = T.optim.Adam(chain(unet.dec_model.parameters()), lr=10e-4)
        else:
            opti = T.optim.Adam(unet.parameters(recurse=True), lr=2e-4)
        if args.ucritic:
            if args.live:
                critic = lambda x: self.unet.forward(x, critic=True)
                critic_copy = lambda x: copy.deepcopy(self.unet).forward(x, critic=True)
            else:
                critic_copy = copy.deepcopy(self.unet)
                critic = lambda x: critic_copy.forward(x, critic=True)
        else:
            critic = self.critic
            critic_opti = T.optim.Adam(critic.parameters())
            """
            if args.resnet:
                critic_opti = T.optim.Adam(critic.head.parameters(recurse=True), lr=1e-4)
            else:
                critic_opti = T.optim.Adam(critic.parameters(recurse=True), lr=1e-4)
            """

        # Setup save path and Logger
        result_path = self.result_path + "segment/" + mode + "/"
        os.makedirs(result_path, exist_ok=True)
        log_file = open(result_path + "log.txt", "w")
        log_file.write(f"{self.args}\n\n")
        if trainf:
            log = []

        # setup window slicing
        w = self.X.shape[2]
        ypad = (w - int(args.window.split('-')[1])) // 2
        xpad = (w - int(args.window.split('-')[0])) // 2
        yslice = slice(ypad, -ypad) if (xpad or ypad) else slice(None)
        xslice = slice(xpad, -xpad) if (xpad or ypad) else slice(None)
        print("window padding:", xpad, ypad)

        # Setup negs
        batchsize = 256
        preds = []
        for bidx in range(0, len(self.X), batchsize):
            print("searching dataset for high and low values...", bidx / len(self.X), end="\r")
            with T.no_grad():
                batch = T.from_numpy(self.X[bidx:bidx + batchsize, 0]).permute(0, 3, 1, 2).float()
                batch2 = T.from_numpy(self.X[bidx:bidx + batchsize, fskip]).permute(0, 3, 1, 2).float()
                # print(batch)
                pred = critic(batch).squeeze()
                pred2 = critic(batch2).squeeze()
            preds.append(T.stack((pred, pred2), dim=1))
        preds = T.cat(preds, dim=0)
        sorted = T.argsort(preds[:, 0])
        high = preds[sorted[8 * len(sorted) // 10], 0]
        low = preds[sorted[2 * len(sorted) // 10], 0]
        highsel = preds[:, 0] >= high
        lowsel = preds[:, 0] <= low
        HX = self.X[highsel]
        # HY =self.Y[highsel, :, args.rewidx]
        HY = preds[highsel]
        LX = self.X[lowsel]
        # NY =self.Y[lowsel, :, args.rewidx]
        LY = preds[lowsel]
        batch_size = 64
        print("test", high, low, HX.shape, LX.shape, HY.shape, LY.shape)
        assert low != high
        Hloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.from_numpy(HX),
                                                                     HY, T.arange(HX.shape[0], dtype=T.uint8)),
                                          batch_size=batch_size // 2, shuffle=True)
        Lloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.from_numpy(LX),
                                                                     LY, T.arange(LX.shape[0], dtype=T.uint8)),
                                          batch_size=batch_size // 2, shuffle=True)
        # Epoch and Batch Loops
        for epoch in range(int(testf) or self.args.epochs):
            # BATCHING THROUGH DATA
            for b_idx, ((HX, HY, HI), (LX, LY, LI)) in enumerate(zip(Hloader, Lloader)):
                loss_string = f"e{epoch} b{b_idx}"

                HX, HX2 = HX[:, 0], HX[:, 1]
                HY, HY2 = HY[:, 0], HY[:, 1]
                LX, LX2 = LX[:, 0], LX[:, 1]
                LY, LY2 = LY[:, 0], LY[:, 1]
                X = T.cat((HX, LX), dim=0)
                Y = T.cat((HY, LY), dim=0)
                X2 = T.cat((HX2, LX2), dim=0)
                Y2 = T.cat((HY2, LY2), dim=0)

                # format data
                XP = X.permute(0, 3, 1, 2).float().to(self.device)
                if args.shift:
                    XP = self.shift_batch(XP)

                # Get negative substitution windows
                negatives = np.random.choice(np.arange(len(LX)), len(XP))
                NXP = (LX[negatives]).permute(0, 3, 1, 2).float().to(self.device)

                # get critic values
                pred = critic(XP).squeeze()
                negpred = critic(NXP).squeeze()

                pos_fails = 1 - (Y.view(pred.shape) - pred).abs()
                neg_fails = 1 - (LY[negatives] - negpred).abs()
                critic_fail_mask = 1  # (pos_fails * neg_fails).squeeze().detach()

                # print(NXW.shape)
                # NXWPAD = T.zeros_like(XP)
                # NXWPAD[:, :, ypad:-ypad, xpad:-xpad] = NXW
                # NXWPAD = XWPAD.permute(0, 2, 3, 1)

                # FILTER in A and B
                # train critic if live
                # LOSSES:
                loss = 0
                if args.live:
                    # critic_loss = F.binary_cross_entropy_with_logits(rawpred, Ybinary)
                    critic_loss = F.mse_loss(pred, Y)
                    if not args.ucritic:
                        critic_opti.zero_grad()
                        critic_loss.backward()
                        critic_opti.step()
                    else:
                        loss = loss + args.clossfak * critic_loss
                    loss_string += f"    live-critic {critic_loss.item()}"

                A = XP
                B = NXP
                Z = unet(A)
                merge_critic = critic_copy if args.copy else critic

                # replace mask
                replaced = A * (1 - Z) + Z * B
                replacevalue = merge_critic(replaced).squeeze() * critic_fail_mask
                # replaceloss = F.binary_cross_entropy_with_logits(replacevalue, negpred.detach())
                # replacevalue = T.sigmoid(replacevalue)
                replaceloss = F.mse_loss(replacevalue, negpred.detach())
                loss = loss + args.lfak * replaceloss
                loss_string += f"   replace: {replaceloss.item()}"

                # inject mask
                if args.inject:
                    injected = B * (1 - Z) + Z * A
                    injectvalue = merge_critic(injected).squeeze() * critic_fail_mask
                    # injectloss = F.binary_cross_entropy_with_logits(injectvalue, pred.detach())
                    # injectvalue = T.sigmoid(injectvalue)
                    injectloss = F.mse_loss(injectvalue, pred.detach())
                    loss = loss + args.lfak * injectloss
                    loss_string += f"   inject: {injectloss.item()}"

                # invert mask
                # inverted = B * (1 - Z) + Z * A
                # invertavlue = critic(inverted).squeeze()
                # invertloss = F.binary_cross_entropy_with_logits(invertavlue, pred.detach())
                # invertvalue = T.sigmoid(invertvalue)
                # loss = loss + invertloss
                # loss_string += f"   invert: {invertloss.item()}"

                if args.staticnorm:
                    valuefak = 1  # * critic_fail_mask.view(-1, 1, 1, 1)
                else:
                    valuefak = 1 - pred.detach().view(-1, 1, 1, 1)  # * critic_fail_mask.view(-1, 1, 1, 1)
                    # valuefak = 1 - Ybinary.view(-1, 1, 1, 1)
                if not args.L2:
                    normloss = args.L1 * F.l1_loss(valuefak * Z, T.zeros_like(Z))
                    # normloss = args.L1 * (valuefak*Z).mean()
                    loss = loss + normloss
                    loss_string += f"   L1: {normloss.item()}"
                else:
                    normloss = args.L2 * F.mse_loss(valuefak * Z, T.zeros_like(Z))
                    loss = loss + normloss
                    loss_string += f"   L2: {normloss.item()}"
                if args.distnorm:
                    mask = Z.cpu().detach()
                    w = X.shape[1]
                    b = X.shape[0]
                    xs = T.arange(w).repeat((b, 1, w, 1)).float() / w
                    ys = T.arange(w).repeat((b, 1, w, 1)).transpose(2, 3).float() / w
                    # print(xs[0], xs[1], xs.shape, ys.shape, mask.shape)
                    xvote = (xs * mask).flatten(start_dim=-2).mean(dim=-1).squeeze().view(b, 1, 1, 1)
                    yvote = (ys * mask).flatten(start_dim=-2).mean(dim=-1).squeeze().view(b, 1, 1, 1)
                    # print(xs.shape, xvote.shape)
                    xs -= xvote  # X Distance
                    ys -= yvote  # Y Distance
                    dist = (xs.pow(2) + xs.pow(2)).pow(0.5)
                    target = mask - dist
                    target[target < 0] = 0
                    distloss = 5 * F.mse_loss(Z, target.to(self.device))
                    loss = loss + distloss
                    loss_string = f"dist-norm: {distloss.item()}   " + loss_string

                # total_var = (Z[:,:,1:]-Z[:,:,:-1]).mean().abs() + \
                #            (Z[:,:,:,1:]-Z[:,:,:,:-1]).mean().abs()
                # loss = loss + 0.0001* total_var
                # loss_string += f"   total var: {round(total_var.item(), 10)}"

                if not b_idx % 10:
                    print((loss_string), end='\n')
                log.append((replaceloss.item(),
                            injectloss.item() if args.inject else 0,
                            normloss.item() if args.L1 or args.L2 else 0,
                            critic_loss.item() if args.live else 0))
                opti.zero_grad()
                loss.backward()
                opti.step()
                # log_file.write(log_msg+"\n")

                # VIZ -----------------------------------
                if (trainf and not b_idx % 100) or testf:  # VISUALIZE
                    vizs = []
                    # A = XWPAD
                    # B = XWPAD[negatives]
                    A = A.cpu().detach().permute(0, 2, 3, 1)
                    B = B.cpu().detach().permute(0, 2, 3, 1)
                    Z = Z.cpu().detach().permute(0, 2, 3, 1)
                    replaced = A * (1 - Z) + Z * B
                    injected = B * (1 - Z) + Z * A
                    # print(Z.shape, A.shape, B.shape, replaced.shape, injected.shape)
                    viz = hsv_to_rgb(A.numpy() / 255) if self.args.color == "HSV" else A.numpy() / 255
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(np.zeros_like(viz))
                    vizs.append(np.zeros_like(viz))
                    vizs.append(viz)
                    viz = hsv_to_rgb(B.numpy() / 255) if self.args.color == "HSV" else B.numpy() / 255
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)
                    viz = hsv_to_rgb(replaced.numpy() / 255) if self.args.color == "HSV" else replaced.numpy() / 255
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)
                    viz = hsv_to_rgb(injected.numpy() / 255) if self.args.color == "HSV" else injected.numpy() / 255
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)
                    viz = T.cat((Z, Z, Z), dim=-1).cpu().numpy()
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)

                    viz = np.concatenate(vizs, axis=0)
                    img = Image.fromarray(np.uint8(255 * viz))
                    if True:
                        adder = 12
                        draw = ImageDraw.Draw(img)
                        for i, value in enumerate(pred.tolist()):
                            x, y = int(i * img.width / len(pred)), 0
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                        for i, value in enumerate(negpred.tolist()):
                            x, y = int(i * img.width / len(Y)), adder
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                        for i, value in enumerate(replacevalue.tolist()):
                            x, y = int(i * img.width / len(Y)), 2 * adder
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                        if args.inject:
                            for i, value in enumerate(injectvalue.tolist()):
                                x, y = int(i * img.width / len(Y)), 3 * adder
                                draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

                        for i, value in enumerate(pos_fails.tolist()):
                            x, y = int(i * img.width / len(Y)), 5 * adder
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                        for i, value in enumerate(neg_fails.tolist()):
                            x, y = int(i * img.width / len(Y)), 6 * adder
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                        """for i, value in enumerate(critic_fail_mask.tolist()):
                            x, y = int(i * img.width / len(Y)), 49
                            draw.text((x, y), str(round(value, 1)), fill=(255, 255, 255), font=self.font)"""

                    # plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path + f"e{epoch}_b{b_idx}.png")

                if False:  # VISUALIZE
                    viz1 = hsv_to_rgb(X.numpy() / 255) if self.args.color == "HSV" else A.numpy() / 255
                    viz1 = np.concatenate(viz1, axis=1)
                    Z = unet(XP)
                    Z = Z.detach().permute(0, 2, 3, 1)
                    seg = X.float() / 255
                    seg[:, :, :, 1] = Z.squeeze()
                    viz2 = hsv_to_rgb(seg.numpy()) if self.args.color == "HSV" else seg.numpy()
                    viz2 = np.concatenate(viz2, axis=1)
                    viz4 = T.cat((Z, Z, Z), dim=-1).cpu().numpy()
                    viz4 = np.concatenate(viz4, axis=1)

                    viz = np.concatenate((viz1, viz2, viz4), axis=0)
                    img = Image.fromarray(np.uint8(255 * viz))
                    draw = ImageDraw.Draw(img)
                    YY = T.sigmoid(critic(XP)).squeeze()
                    for i, value in enumerate(YY.tolist()):
                        x, y = int(i * img.width / len(YY)), 1
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

                    # plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path + f"e{epoch}_b{b_idx}.png")

            # PLOT LOSS
            plt.clf()
            llog = np.array(log)
            loss_names = ["replace", "inject", "norm", "live-critic"]
            for i in range(len(llog[0])):
                plt.plot(get_moving_avg(llog[:, i], 30), label=loss_names[i])
            plt.legend()
            plt.savefig(result_path + "_loss.png")

            # SAVE MODEL
            if not (epoch + 1) % args.saveevery:
                self.save_models(modelnames=[self.unetname])

            if trainf:
                # critic.sched.step()
                pass

        if False:
            log = np.array(log)
            end = len(log) // 10
            log1 = log[:10 * end, 0]
            log1 = log1.reshape((-1, 10))
            log1 = log1.mean(axis=-1)
            plt.plot(log1, label="value loss")
            log2 = log[:10 * end, 1]
            log2 = log2.reshape((-1, 10))
            log2 = log2.mean(axis=-1)
            plt.plot(log2, label="L2 norm")
            plt.legend()
            plt.savefig(result_path + f"loss.png")
        print()
        self.save_models(modelnames=[self.unetname])

    def trans_embeds(self):
        args = self.args
        result_path = self.result_path + "trans/"
        print(result_path)
        os.makedirs(result_path, exist_ok=True)

        # loader = self.dataloader
        # encoder = ConvEncoder64x64(enc_dim=64)
        # decoder = ConvDecoder64x64(enc_dim=64, end=[nn.Sigmoid()])
        # opti = T.optim.Adam(chain(encoder.parameters(recurse=True), decoder.parameters(recurse=True)))

        model = VGGEmbedder()
        batch = self.X[:8, 0]
        X = batch / 255.0
        with T.no_grad():
            out, embeds = model.forward(X, hsv=True)
        full_embeds = embeds.permute(0, 2, 3, 1)
        print(out.shape, embeds.shape)

        all_simmaps = []
        interval = 128
        step = 64
        thresh = 0.5
        for scope in [(i * step, (i * step) + interval) for i in range(0, full_embeds.shape[-1] // step)]:
            scope = slice(*scope)
            embeds = full_embeds[:, :, :, scope]
            print(scope)

            local_maps = []
            for target in [(4, 32, 32), (4, 12, 32), (0, 48, 32)]:
                target = embeds[target[0], target[1], target[2]]
                simmap = T.cosine_similarity(embeds, target, dim=-1)
                threshed = simmap * (simmap > thresh)
                local_maps.extend([simmap, threshed])

            spacer = T.ones(len(batch), 10, 64) * 0.5
            local_maps.append(spacer)
            all_simmaps.extend(local_maps)

        vizs = []
        vizX = hsv_to_rgb(X) if self.args.color == "HSV" else X
        vizs.append(np.concatenate(vizX, axis=1))

        for simmap in all_simmaps:
            # print(simmap.shape)
            if simmap.shape[1] == 10:
                sim = T.stack((simmap, simmap, simmap), dim=-1).numpy()
                vizs.append(np.concatenate(sim, axis=1))
            else:
                vizs.append(np.concatenate(vizX * (simmap.numpy()[:, :, :, None]), axis=1))

        viz = np.concatenate(vizs, axis=0)
        img = Image.fromarray(np.uint8(255 * viz))
        draw = ImageDraw.Draw(img)
        print(img.width, img.height)

        for i, value in enumerate([]):
            x, y = int(i * img.width / len(PY)), 1
            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

        picargs = f"i{interval}-s{step}.png"
        img.save(result_path + picargs)
        print("saved:", result_path + picargs)

        for epoch in range(0):
            for bix, (X, Y, I) in enumerate(loader):
                X = X / 255
                XP = X[:, 0].permute(0, 3, 1, 2).float()

                btlnck = encoder(XP)
                recon = decoder(btlnck)
                # print(XP, recon)

                opti.zero_grad()
                loss = F.mse_loss(recon, XP)
                loss.backward()
                opti.step()
                if not bix % 10:
                    print(epoch, bix, "rec loss:", loss.item())

                if not bix % 100:  # VISUALIZE
                    vizs = []
                    X = X[:, 0]
                    X2 = recon.detach().permute(0, 2, 3, 1)
                    vizX = hsv_to_rgb(X.numpy()) if self.args.color == "HSV" else X.numpy()
                    vizX2 = hsv_to_rgb(X2.numpy()) if self.args.color == "HSV" else X.numpy()

                    vizs.append(np.concatenate(vizX, axis=1))
                    vizs.append(np.concatenate(vizX2, axis=1))
                    viz = np.concatenate(vizs, axis=0)
                    img = Image.fromarray(np.uint8(255 * viz))
                    draw = ImageDraw.Draw(img)

                    for i, value in enumerate([]):
                        x, y = int(i * img.width / len(PY)), 1
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

                    img.save(result_path + f"e{epoch}_b{bix}.png")

    def dream(self):
        args = self.args
        # Setup save path and Logger
        result_path = self.result_path + f"dreamsteps{args.dreamsteps}" + "/"
        os.makedirs(result_path, exist_ok=True)
        loader = self.testdataloader
        critic = self.critic
        dreamsteps = args.dreamsteps

        # Epoch and Batch Loops
        for b_idx, (X, Y, I) in enumerate(loader):
            # FORWARD PASS---------------------------

            XP = (X.permute(0, 3, 1, 2).float().to(self.device)).requires_grad_()

            opti = T.optim.Adam([XP], lr=0.1)
            if args.discounted:
                Y = Y[:, args.rewidx].float().to(self.device)
            else:
                Y = Y.float().to(self.device)

            pred = critic(XP).squeeze()
            original_value = T.sigmoid(pred)
            original_img = XP.data.detach().clone()
            for upidx in range(dreamsteps):
                if args.discounted:
                    loss = F.mse_loss(pred, T.zeros_like(pred))
                else:
                    loss = F.binary_cross_entropy_with_logits(pred, T.zeros_like(pred))
                # print(f"b{b_idx} up-step {upidx}", loss.item(), end="\r")
                critic.zero_grad()
                loss.backward()
                # print("grad", XP.grad[0])
                avg_grad = np.abs(XP.grad.data.cpu().numpy()).mean()
                norm_lr = 0.01 / avg_grad
                XP.data += norm_lr * avg_grad
                XP = XP.clamp(0, 255)
                pred = critic(XP).squeeze()
                final_value = T.sigmoid(pred)
                XP.grad.data.zero_()
                # print(XP.requires_grad)
                # print("grad", XP.grad[0])
            print((XP - original_img).data.max())
            # log_file.write(log_msg+"\n")

            # VIZ -----------------------------------
            XP = XP.detach().permute(0, 2, 3, 1).numpy() / 255
            pre = pre.detach().permute(0, 2, 3, 1).numpy() / 255
            viz3 = np.zeros_like(XP.numpy())
            viz3[:, :, 2] = np.abs(np.mean((XP - original_img).numpy(), axis=-1))
            viz = hsv_to_rgb(XP) if self.args.color == "HSV" else XP
            viz = np.concatenate(viz, axis=1)
            viz2 = hsv_to_rgb(X.numpy() / 255) if self.args.color == "HSV" else X.numpy() / 255
            viz2 = np.concatenate(viz2, axis=1)

            viz = np.concatenate((viz, viz2, viz3), axis=0)
            img = Image.fromarray(np.uint8(255 * viz))
            draw = ImageDraw.Draw(img)
            for i, value in enumerate(final_value.tolist()):
                x, y = int(i * img.width / len(Yfinal)), 1
                draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
            for i, value in enumerate(original_value.tolist()):
                x, y = int(i * img.width / len(Yorig)), int(1 + img.height / 2)
                draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

            # plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
            img.save(result_path + f"b{b_idx}.png")

        print()

    def vis_unet_embeddings(self):
        viz_path = self.result_path + "umap/"
        os.makedirs(viz_path, exist_ok=True)

        unet = self.unet
        n_samples = 10
        np.random.seed(42)

        # GET DATA
        sidxs = np.random.choice(np.arange(len(self.HX)), n_samples)
        raw_batch = self.HX[sidxs]
        print("rawbatch", raw_batch.shape)

        # GET TREE TRUNK DATA
        treepath = "debug/trees/"
        treeY = []
        treeX = []
        immages = [n for n in os.listdir(treepath) if "w" in n or "b" in n]
        for idx, name in enumerate(immages):
            print(name)
            if not ('w' in name or 'b' in name):
                continue
            img = cv2.imread(treepath + name, cv2.IMREAD_UNCHANGED)
            y = img[:, :, -1]
            y[y == 255] = idx + 1
            x = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
            treeX.append(x)
            treeY.append(y)
        treeX = np.stack(treeX, axis=0)
        treeY = np.stack(treeY, axis=0)
        treeY[treeY > 5] = 0
        print(treeY[treeY > 0])
        treebatch = T.from_numpy(treeX).permute(0, 3, 1, 2).float()
        print("trees", treeX.shape, treeY.shape)

        batch = T.from_numpy(raw_batch).permute(0, 3, 1, 2).float()
        print("batch", batch.shape)

        with T.no_grad():
            mask, embeds = unet.forward(batch, embeds=True)
            tmask, tembeds = unet.forward(treebatch, embeds=True)

        # masks
        resized_mask = transforms.Resize(embeds.shape[-2:])(mask)
        flat_mask = resized_mask.permute(0, 2, 3, 1).flatten(0, -2).numpy()
        resized_tmask = transforms.Resize(embeds.shape[-2:])(tmask)
        flat_tmask = resized_tmask.permute(0, 2, 3, 1).flatten(0, -2).numpy()
        resized_treeY = transforms.Resize(embeds.shape[-2:])(T.from_numpy(treeY))
        flat_treeY = resized_treeY.unsqueeze(-1).flatten(0, -2).numpy().squeeze()
        print("flat mask", flat_mask.shape)
        print("flat treey", flat_treeY.shape)

        # embeds
        permuted_embeds = embeds.permute(0, 2, 3, 1)
        flat_embeds = permuted_embeds.flatten(0, -2).numpy()
        permuted_tembeds = tembeds.permute(0, 2, 3, 1)
        flat_tembeds = permuted_tembeds.flatten(0, -2).numpy()

        print("umapping...")
        my_umap = umap.UMAP(random_state=42)
        standard_embedding = my_umap.fit_transform(flat_embeds)
        tree_embedding = my_umap.transform(flat_tembeds)[flat_treeY > 0]
        treeY_color = flat_treeY[flat_treeY > 0]
        x = standard_embedding[:, 0]
        y = standard_embedding[:, 1]
        print("plotting...")
        plt.scatter(x, y, c=flat_mask, s=0.1)
        plt.scatter(tree_embedding[:, 0], tree_embedding[:, 1], marker="+", s=0.1, c=treeY_color, cmap="cool")
        plt.savefig(viz_path + "umap-plot.png", dpi=300)

        # xrange = (6 < x) & (x< 8)
        # yrange = y < -2
        flat_selection = x > 15  # xrange & yrange
        print("flat sel", flat_selection.shape)
        batch_selection = flat_selection.reshape(n_samples, *embeds.shape[-2:])
        print("batch sel", batch_selection.shape)
        resized_batch_selection = transforms.Resize(batch.shape[-2:])(T.from_numpy(batch_selection)).unsqueeze(
            -1).numpy()
        print("resized batch sel", resized_batch_selection.shape)

        ref_mask = mask.permute(0, 2, 3, 1)
        batch_mask = np.ones_like(raw_batch) * resized_batch_selection * (ref_mask > 0.5).numpy()
        batch_mask_2 = np.ones_like(raw_batch) * resized_batch_selection * (ref_mask < 0.5).numpy()
        print("batch mask", batch_mask.shape)

        rgb_batch = hsv_to_rgb(raw_batch / 255)
        print("rgb batch", rgb_batch.shape)
        masked_rgb_batch = batch_mask * rgb_batch
        masked_rgb_batch_2 = batch_mask_2 * rgb_batch
        print("masked rgb batch", masked_rgb_batch.shape)

        for idx, frame in enumerate(masked_rgb_batch):
            img = np.concatenate((rgb_batch[idx], frame, batch_mask[idx], masked_rgb_batch_2[idx], batch_mask_2[idx]),
                                 axis=1)
            plt.imsave(viz_path + f"{idx}.png", img)

    def shift_batch(self, X):
        xshift = int(self.args.shift * T.rand(1))
        if T.rand(1) > 0.5:
            # X = T.cat((X[:, :, yshift:], X[:, :, :yshift]), dim=2)
            X = T.cat((X[:, :, xshift:], X[:, :, :xshift]), dim=2)
        else:
            X = T.cat((X[:, :, -xshift:], X[:, :, :-xshift]), dim=2)
        return X

    def segment(self, mode="train", test=0):
        args = self.args
        testf = mode == "test"
        trainf = mode == "train"
        loader = self.dataloader if trainf else self.testdataloader
        critic = self.critic
        unet = self.unet
        opti = T.optim.Adam(unet.parameters(recurse=True), lr=args.lr)
        if args.live:
            if args.resnet:
                critic_opti = T.optim.Adam(critic.head.parameters(recurse=True), lr=1e-4)
            else:
                critic_opti = T.optim.Adam(critic.parameters(recurse=True), lr=1e-4)
        if args.clustercritic:
            with gzip.open(self.data_path + f"{str(self.args.clustercritic)}-cluster", 'rb') as fp:
                channels = pickle.load(fp)

        # Setup save path and Logger
        result_path = self.result_path + "segment/" + mode + "/"
        os.makedirs(result_path, exist_ok=True)
        log_file = open(result_path + "log.txt", "w")
        log_file.write(f"{self.args}\n\n")
        if trainf:
            log = []

        # Epoch and Batch Loops
        for epoch in range(int(testf) or self.args.epochs):
            for b_idx, (X, Y, I) in enumerate(loader):
                loss_string = ""
                # FORWARD PASS---------------------------
                XP = X.permute(0, 3, 1, 2).float().to(self.device)
                if args.discounted:
                    Y = Y[:, args.rewidx].float().to(self.device)
                else:
                    Y = Y.float().to(self.device)

                # FILTER in A and B
                if trainf:
                    if not args.clustercritic:
                        pred = critic(XP).squeeze()
                    else:
                        CHS = T.from_numpy(channels[I]).float().to(self.device)
                        XPE = T.cat((XP, CHS), dim=1)
                        pred = critic(XPE).squeeze()
                    pred = T.sigmoid(pred)
                    if args.live:
                        critic_loss = F.binary_cross_entropy_with_logits(pred, Y)
                        critic_opti.zero_grad()
                        critic_loss.backward()
                        critic_opti.step()
                        loss_string = f"live-critic {critic_loss.item()}   " + loss_string

                    # mask = pred>args.threshold
                    negmask = pred < (1 - args.threshold)
                    if not negmask.sum():
                        print(negmask.sum())
                        continue
                    # print(negmask.shape)
                    # print(np.nonzero(negmask))
                    # print(np.nonzero(negmask.numpy())[0])
                    negatives = np.random.choice((np.nonzero(negmask.cpu().numpy()))[0], len(X))
                    A = XP
                    B = XP[negatives]
                    Z = unet(A)
                    merged = A * (1 - Z) + Z * B
                    if not args.clustercritic:
                        mergevalue = critic(merged).squeeze()
                    else:
                        mergechs = CHS * (1 - Z) + Z * CHS[negatives]
                        mergecombined = T.cat((merged, mergechs), dim=1)
                        mergevalue = critic(mergecombined).squeeze()
                    valueloss = F.binary_cross_entropy_with_logits(mergevalue, T.zeros_like(mergevalue))
                    loss = valueloss
                    if args.L1:
                        if args.staticnorm:
                            valuefak = 1
                        else:
                            valuefak = 1 - pred.detach().view(-1, 1, 1, 1)
                        normloss = args.L1 * F.l1_loss(valuefak * Z, T.zeros_like(Z))
                        # normloss = args.L1 * (valuefak*Z).mean()
                        loss = loss + normloss
                        loss_string = f"L1: {normloss.item()}   " + loss_string
                    elif args.L2:
                        if args.staticnorm:
                            valuefak = 1
                        else:
                            valuefak = 1 - pred.detach().view(-1, 1, 1, 1)
                        normloss = args.L2 * F.mse_loss(valuefak * Z, T.zeros_like(Z))
                        loss = loss + normloss
                        loss_string = f"L2: {normloss.item()}   " + loss_string
                    if args.distnorm:
                        mask = Z.cpu().detach()
                        w = X.shape[1]
                        b = X.shape[0]
                        xs = T.arange(w).repeat((b, 1, w, 1)).float() / w
                        ys = T.arange(w).repeat((b, 1, w, 1)).transpose(2, 3).float() / w
                        # print(xs[0], xs[1], xs.shape, ys.shape, mask.shape)
                        xvote = (xs * mask).flatten(start_dim=-2).mean(dim=-1).squeeze().view(b, 1, 1, 1)
                        yvote = (ys * mask).flatten(start_dim=-2).mean(dim=-1).squeeze().view(b, 1, 1, 1)
                        # print(xs.shape, xvote.shape)
                        xs -= xvote  # X Distance
                        ys -= yvote  # Y Distance
                        dist = (xs.pow(2) + xs.pow(2)).pow(0.5)
                        target = mask - dist
                        target[target < 0] = 0
                        distloss = 5 * F.mse_loss(Z, target.to(self.device))
                        loss = loss + distloss
                        loss_string = f"dist-norm: {distloss.item()}   " + loss_string

                    mergevalue = T.sigmoid(mergevalue)
                    loss_string = f"e{epoch} b{b_idx}  value-loss: {loss.item()}   " + loss_string
                    print((loss_string))
                    log.append((valueloss.item(), normloss.item() if args.L1 or args.L2 else 0))
                    opti.zero_grad()
                    loss.backward()
                    opti.step()
                # log_file.write(log_msg+"\n")

                # VIZ -----------------------------------
                if (trainf and not b_idx % 100):  # VISUALIZE
                    vizs = []
                    A = A.cpu().detach().permute(0, 2, 3, 1)
                    B = B.cpu().detach().permute(0, 2, 3, 1)
                    Z = Z.cpu().detach().permute(0, 2, 3, 1)
                    # print("SHAPE", Z.shape)
                    merged = merged.cpu().detach().permute(0, 2, 3, 1)
                    viz = hsv_to_rgb(A.numpy() / 255) if self.args.color == "HSV" else A.numpy() / 255
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)
                    viz = hsv_to_rgb(B.numpy() / 255) if self.args.color == "HSV" else B.numpy() / 255
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)
                    viz = hsv_to_rgb(merged.numpy() / 255) if self.args.color == "HSV" else merged.numpy() / 255
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)
                    viz = T.cat((Z, Z, Z), dim=-1).cpu().numpy()
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)

                    viz = np.concatenate(vizs, axis=0)
                    img = Image.fromarray(np.uint8(255 * viz))
                    draw = ImageDraw.Draw(img)
                    for i, value in enumerate(pred.tolist()):
                        x, y = int(i * img.width / len(pred)), 1
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                    for i, value in enumerate(pred[negatives].tolist()):
                        x, y = int(i * img.width / len(pred[negatives])), int(1 + img.height / 3)
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                    for i, value in enumerate(mergevalue.tolist()):
                        x, y = int(i * img.width / len(mergevalue)), int(1 + 2 * img.height / 3)
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

                    # plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path + f"e{epoch}_b{b_idx}.png")

                if testf:  # VISUALIZE
                    viz1 = hsv_to_rgb(X.numpy() / 255) if self.args.color == "HSV" else A.numpy() / 255
                    viz1 = np.concatenate(viz1, axis=1)
                    Z = unet(XP)
                    Z = Z.detach().permute(0, 2, 3, 1)
                    seg = X.float() / 255
                    seg[:, :, :, 1] = Z.squeeze()
                    viz2 = hsv_to_rgb(seg.numpy()) if self.args.color == "HSV" else seg.numpy()
                    viz2 = np.concatenate(viz2, axis=1)
                    viz4 = T.cat((Z, Z, Z), dim=-1).cpu().numpy()
                    viz4 = np.concatenate(viz4, axis=1)

                    viz = np.concatenate((viz1, viz2, viz4), axis=0)
                    img = Image.fromarray(np.uint8(255 * viz))
                    draw = ImageDraw.Draw(img)
                    YY = T.sigmoid(critic(XP)).squeeze()
                    for i, value in enumerate(YY.tolist()):
                        x, y = int(i * img.width / len(YY)), 1
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

                    # plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path + f"e{epoch}_b{b_idx}.png")

            if epoch and not epoch % args.saveevery:
                self.save_models(modelnames=[self.unetname])

            if trainf:
                # critic.sched.step()
                pass
        if trainf:
            log = np.array(log)
            end = len(log) // 10
            log1 = log[:10 * end, 0]
            log1 = log1.reshape((-1, 10))
            log1 = log1.mean(axis=-1)
            plt.plot(log1, label="value loss")
            log2 = log[:10 * end, 1]
            log2 = log2.reshape((-1, 10))
            log2 = log2.mean(axis=-1)
            plt.plot(log2, label="L2 norm")
            plt.legend()
            plt.savefig(result_path + f"loss.png")
        print()

    def sum_segm(self, mode="train", test=0):
        args = self.args
        testf = mode == "test"
        trainf = mode == "train"
        loader = self.dataloader if trainf else self.testdataloader
        critic = self.critic
        unet = self.unet
        opti = T.optim.Adam(unet.parameters(recurse=True))
        if args.live:
            if args.resnet:
                critic_opti = T.optim.Adam(critic.head.parameters(recurse=True), lr=1e-4)
            else:
                critic_opti = T.optim.Adam(critic.parameters(recurse=True), lr=1e-4)
        if args.clustercritic:
            with gzip.open(self.data_path + f"{str(self.args.clustercritic)}-cluster", 'rb') as fp:
                channels = pickle.load(fp)

        # Setup save path and Logger
        result_path = self.result_path + "segment/" + mode + "/"
        os.makedirs(result_path, exist_ok=True)
        log_file = open(result_path + "log.txt", "w")
        log_file.write(f"{self.args}\n\n")
        if trainf:
            log = []

        # Epoch and Batch Loops
        for epoch in range(int(testf) or self.args.epochs):
            for b_idx, (X, Y, I) in enumerate(loader):
                loss_string = ""
                # Formatting
                XP = X.permute(0, 3, 1, 2).float().to(self.device)
                if args.discounted:
                    Y = Y[:, args.rewidx].float().to(self.device)
                else:
                    Y = Y.float().to(self.device)

                # Unet
                mask = unet(XP)

                # Loss
                sum_flat_mask = mask.view(mask.shape[0], -1).sum(dim=1)
                target = (mask.view(mask.shape[0], -1) > 0.5).sum(dim=1)
                target[target < 10] = 10.0
                loss = F.mse_loss(sum_flat_mask, target)
                opti.zero_grad()
                loss.backward()
                opti.step()

                # VIZUALIZE
                if (trainf and not b_idx % 100) or testf:
                    vizs = []
                    vizX = hsv_to_rgb(X.numpy() / 255) if self.args.color == "HSV" else X.numpy() / 255
                    mask = mask.detach().permute(0, 2, 3, 1)
                    vizS = T.cat((mask, mask, mask), dim=-1).cpu().numpy()

                    if testf:
                        self.batch_to_vid([vizX, vizS], Y=Y.toList())
                    else:
                        vizXconc = np.concatenate(vizX, axis=1)
                        vizs.append(vizXconcconc)
                        vizSconc = np.concatenate(vizS, axis=1)
                        vizs.append(vizSconc)

                        viz = np.concatenate(vizs, axis=0)
                        img = Image.fromarray(np.uint8(255 * viz))
                        draw = ImageDraw.Draw(img)
                        for i, value in enumerate(Y.tolist()):
                            x, y = int(i * img.width / len(Y)), 1
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

                        # plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                        img.save(result_path + f"e{epoch}_b{b_idx}.png")

                if False:  # VISUALIZE
                    viz1 = hsv_to_rgb(X.numpy() / 255) if self.args.color == "HSV" else A.numpy() / 255
                    viz1 = np.concatenate(viz1, axis=1)
                    Z = unet(XP)
                    Z = Z.detach().permute(0, 2, 3, 1)
                    seg = X.float() / 255
                    seg[:, :, :, 1] = Z.squeeze()
                    viz2 = hsv_to_rgb(seg.numpy()) if self.args.color == "HSV" else seg.numpy()
                    viz2 = np.concatenate(viz2, axis=1)
                    viz4 = T.cat((Z, Z, Z), dim=-1).cpu().numpy()
                    viz4 = np.concatenate(viz4, axis=1)

                    viz = np.concatenate((viz1, viz2, viz4), axis=0)
                    img = Image.fromarray(np.uint8(255 * viz))
                    draw = ImageDraw.Draw(img)
                    YY = T.sigmoid(critic(XP)).squeeze()
                    for i, value in enumerate(YY.tolist()):
                        x, y = int(i * img.width / len(YY)), 1
                        draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

                    # plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path + f"e{epoch}_b{b_idx}.png")

            if epoch and not epoch % args.saveevery:
                self.save_models(modelnames=[self.unetname])

    def cluster(self, mode="train", test=0):
        args = self.args
        testf = mode == "test"
        trainf = mode == "train"
        loader = self.dataloader if trainf else self.testdataloader
        batchsize = loader.batch_size
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 14)
        size = self.trainsize if trainf else self.testsize
        assert not (args.clustersave and args.savekmeans)
        if args.clustersave:
            cdict = dict([(k, np.zeros((size, int(k), 64, 64))) for k in args.cluster.split(',')])

        treemask = np.zeros((64, 64), dtype=np.uint8)
        treemask[21:42, 25:39] = 1

        # Setup save path and Logger
        result_path = self.result_path + "cluster/" + mode + "/"
        os.makedirs(result_path, exist_ok=True)
        log_file = open(result_path + "log.txt", "w")
        log_file.write(f"{self.args}\n\n")

        if args.savekmeans:
            sel = np.random.choice(np.arange(len(self.X)), 256)
            loader = [(T.from_numpy(self.X[sel]), self.Y[sel], 0)]
            batchsize = 256

        chs = [0, 1]
        # Epoch and Batch Loops
        for epoch in range(1):
            for b_idx, (X, Y, I) in enumerate(loader):
                if args.clustersave:
                    print(f"clustering dataset: {b_idx}/{math.ceil(args.datasize / batchsize)}")
                else:
                    print("generating kmeans...")
                if args.color == "RGB":
                    X = rgb_to_hsv(X)

                pixels = X.view(-1, 3)
                test = pixels.view(X.shape)
                assert (X == test).all()
                pixels = pixels[:, chs].float() / 255
                pixels[:, 1] *= 0.1

                plt.ylim(0, 1)
                plt.hist2d(pixels[:, 0].numpy(), pixels[:, 1].numpy(), 100)
                plt.savefig(result_path + f"e{epoch}_b{b_idx}_scatter.png")

                vizs = []
                viz = hsv_to_rgb(X.numpy() / 255) if self.args.color == "HSV" else X.numpy() / 255
                treemaskviz = viz.copy()
                treemaskviz[:, treemask == 0] *= 0.5
                viz = np.concatenate(viz, axis=1)
                treemaskviz = np.concatenate(treemaskviz, axis=1)
                vizs.append(viz)
                vizs.append(treemaskviz)
                text = []
                for nc in [int(number) for number in args.cluster.split(',')]:

                    clusters = KMeans(n_clusters=nc).fit(pixels)
                    labels = clusters.labels_.reshape(X.shape[:-1])

                    clusterlayers = []
                    clustervalues = []
                    for clidx in range(nc):
                        row = X.numpy()
                        labelselect = labels == clidx
                        clusterlayers.append(labelselect)

                        # determine values
                        if not args.savekmeans:
                            Ylabel = Y.numpy()
                        else:
                            Ylabel = Y
                        tmpsel = Ylabel == 1
                        tmptm = np.tile(treemask, (int(np.sum(tmpsel)), 1, 1))
                        # print(tmptm.shape, labelselect[tmpsel].shape)
                        clustervalues.append(np.sum(labelselect[tmpsel] * tmptm) / np.sum(labelselect[tmpsel]))
                        # print(clidx, "clustervalue:", clustervalues[-1])

                        if False:  # Spatial Clustering
                            for pi in range(len(row)):
                                w = X.shape[1]
                                xs = T.arange(w).repeat((w, 1)).float() / w
                                ys = T.arange(w).repeat((w, 1)).transpose(0, 1).float() / w
                                positions = T.stack((xs, ys), dim=-1)
                                flat_pos = positions.view(-1, 2)
                                sublabels = np.zeros((w, w))

                                for li in range(nc):
                                    square_selection = labels[pi] == li
                                    flat_selection = square_selection.reshape(-1)
                                    # print(positions, positions.shape)
                                    sub = KMeans(n_clusters=2).fit(flat_pos[flat_selection])
                                    flat_sub_labels = sub.labels_
                                    sublabels[square_selection] = flat_sub_labels

                            row[pi, :, :, 0] = 255 - 255 * ((labels[pi] / (nc * 2)) * (1 + sublabels))

                        row[:, :, :, 1] = 255 * ((labelselect))

                        if b_idx < 10 and args.viz:  # "VISUALIZE"
                            viz = hsv_to_rgb(row / 255) if self.args.color == "HSV" else row / 255
                            viz[:, treemask == 0] *= 0.5
                            viz = np.concatenate(viz, axis=1)
                            vizs.append(viz)
                            segmap = np.stack((labelselect, labelselect, labelselect), axis=-1)
                            viz = np.concatenate(segmap, axis=1)
                            vizs.append(viz)
                            text.append(f"{nc} {clidx}\n{clustervalues[clidx]}")

                    # SAVE KMEANS
                    targetcluster = np.argmax(clustervalues)
                    print("Target Cluster:", targetcluster)
                    if args.savekmeans:
                        with open(self.save_path + f'kmeans.p', 'wb') as fp:
                            pickle.dump((clusters, targetcluster), fp)

                    clusterlayers = np.transpose(np.array(clusterlayers, dtype=np.uint8), axes=(1, 0, 2, 3))
                    # print(clusterlayers.shape)
                    if args.clustersave:
                        cdict[str(nc)][I] = clusterlayers

                if b_idx < 10 and args.viz:  # "VISUALIZE"
                    viz = np.concatenate(vizs, axis=0)[:, :64 * 128]
                    img = Image.fromarray(np.uint8(255 * viz))
                    draw = ImageDraw.Draw(img)
                    for i, word in enumerate(text):
                        begin = 2
                        x, y = 0, (begin + i * 2) * img.height / (2 * len(text) + begin)
                        draw.text((x, y), word, fill=(255, 255, 255), font=font)
                    # for i, value in enumerate(pred[negatives].tolist()):
                    #     x, y = int(i*img.width/len(pred[negatives])), int(1+img.height/3)
                    #     draw.text((x, y), str(round(value, 3)), fill=(255,255,255), font=self.font)
                    # for i, value in enumerate(mergevalue.tolist()):
                    #     x, y = int(i*img.width/len(mergevalue)), int(1+2*img.height/3)
                    #     draw.text((x, y), str(round(value, 3)), fill=(255,255,255), font=self.font)

                    # plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path + f"e{epoch}_b{b_idx}.png")

                if args.savekmeans:
                    break
        if args.clustersave:
            for key in cdict:
                print(key, cdict[key].shape)
                with gzip.GzipFile(self.data_path + f"{mode}-{key}-cluster", 'wb') as fp:
                    pickle.dump(cdict[key], fp)

    def batch_to_vid(self, batches, Y=None):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        ywid = batches[0].shape[1]
        xwid = batches[0].shape[2]
        out = cv2.VideoWriter(resultdir + result_args + '.avi', fourcc, 20.0, (xwid * len(batches), ywid))

        frames = np.concatenate(batches, axis=2)
        frames = (255 * frames).astype(np.uint8)
        for idx, frame in enumerate(frames):
            if Y is not None:
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                x, y = 1, 1
                draw.text((x, y), str(round(Y[idx], 3)), fill=(255, 255, 255), font=self.font)
                frame = np.array(img)
            out.write(frame)
        out.release()

    def create_patch_embedding_clusters(self):
        print("Starting to create patch embedding clusters with tree prob")
        args = self.args
        # HYPERARAMS
        patchwid = self.args.embed_patch_width
        stride = 2
        embed_dim = self.args.embed_dim
        n_clusters = self.args.embed_cluster
        reward_idx = 4
        n_samples = self.args.embed_train_samples
        channels = [0] if args.hue else ([0, 1, 2] if args.hsv else [0, 1])

        self.embedder = PatchEmbedder(embed_dim=embed_dim, n_cluster=n_clusters,
                                      channels=channels, pw=patchwid, stride=stride,
                                      norm=self.args.embed_norm)

        # REAL DATASET
        if not args.dummy:
            # LOAD NAV DATA
            # navdatadir = f"./data/navigate/train/{self.args.embed_train_samples*3}"
            navdatadir = f"./train/navigate/train/{self.args.navdatasize}"
            if not os.path.exists(navdatadir + "data.pickle"):
                self.collect_navigation_dataset(datadir=navdatadir)

            with gzip.open(navdatadir + "data.pickle", 'rb') as fp:
                NX, NY = pickle.load(fp)
                NY = NY[:, 0]
                NX = NX / 255
            print("loaded navigation data:", NX.shape, NY.shape)

            # FUSE WITH TREE DATA
            high_reward = self.Y[:, reward_idx] >= self.args.embed_pos_threshold
            print("high reward frames in treechop dataset:", sum(high_reward))
            TX = self.X[high_reward] / 255
            TY = np.ones(len(TX))
            navselection = np.random.randint(len(NX), size=n_samples)
            treeselection = np.random.randint(len(TY), size=n_samples)
            X = np.concatenate((TX[treeselection], NX[navselection]), axis=0)
            Y = np.concatenate((TY[treeselection], NY[navselection]), axis=0)
            print("fused dataset:", X.shape, Y.shape)

            # VIS FUSED DATASET
            fused_dir = f"train/patchembed/{self.embed_data_args}-train-frames/"
            os.makedirs(fused_dir, exist_ok=True)
            RGB_X = hsv_to_rgb(X)
            shape = RGB_X.shape[:3]
            xmid = shape[2] / 2
            ymid = shape[1] / 2
            xslice = slice(int(xmid - shape[1] / 12), math.ceil(xmid + shape[1] / 12))
            yslice = slice(int(ymid - shape[2] / 4), math.ceil(ymid + shape[2] / 4))
            RGB_X[Y == 1] *= 0.5
            RGB_X[Y == 1, yslice, xslice] *= 2
            for idx in range(0, len(X), 10):
                plt.imsave(fused_dir + f"{'negative' if not Y[idx] else 'positive'}-{idx}.png", RGB_X[idx])

        # DUMMY DATASET
        if self.args.dummy:
            tree = cv2.cvtColor(cv2.imread("train/navigate/tree.png"), cv2.COLOR_BGR2RGB)
            nav = cv2.cvtColor(cv2.imread("train/navigate/nav.png"), cv2.COLOR_BGR2RGB)
            X = np.stack((tree, nav), axis=0)
            X = rgb_to_hsv(X / 255)
            Y = np.array([1, 0])
            print("using dummy dataset:", X.shape)

        # PIXEL CLUSTERS
        pixels = X.reshape(-1, 3)[::10, channels]
        print("fitting pixel clusters (gmm) to pixels with shape:", pixels.shape)
        pixel_clusters = GMM(n_components=embed_dim).fit(pixels)
        self.embedder.pixel_clusters = pixel_clusters

        print("embedding the dataset...")
        flat_embeds, pshape = self.embedder.embed_batch(X)

        if False:
            # CREATE PATCHES
            print("creating patches...")
            patches = self.embedder.make_patches(X, patchwid, stride)
            print("patches shape and max:", patches.shape, np.max(patches))

            # CREATE EMBEDDINGS
            print("creating embeddings...")
            embeds = self.embedder.embed_patches(patches, verbose=True)

            # CLUSTER EMBEDDING SPACE
            print("clustering embedding space...")
            flat_embeds = embeds.reshape(-1, embed_dim)

        # CLUSTER PATCH EMBEDS
        skipped_embeds = flat_embeds[::5]
        # print("fitting the embedding clusters (gmm) on embeds with shape:", skipped_embeds.shape)
        # embed_clusters = GMM(n_components=n_clusters)
        print("fitting the embedding clusters (kmeans) on embeds with shape:", skipped_embeds.shape)
        embed_clusters = KMeans(n_clusters=n_clusters)
        embed_clusters.fit(skipped_embeds)
        flat_labels = embed_clusters.predict(flat_embeds)
        labels = flat_labels.reshape(pshape[0:3])

        # CALC CLUSTER TREE PROBABILITIES
        print("calculating cluster tree probs...")
        # gt = np.ones(embeds.shape[:3])*Y[:,None,None]
        shape = pshape[:3]
        gt = np.zeros(shape)
        xmid = shape[2] / 2
        ymid = shape[1] / 2
        xslice = slice(int(xmid - shape[1] / 10), math.ceil(xmid + shape[1] / 10))
        yslice = slice(int(ymid - shape[2] / 3), math.ceil(ymid + shape[2] / 3))
        gt[Y == 1, yslice, xslice] = 1
        flat_gt = gt.reshape(-1)
        tree_probs = np.zeros((n_clusters, 4))
        num_all_pos = np.sum(flat_gt)
        for idx in range(n_clusters):
            flat_patch_selection = flat_labels == idx
            num_pos = np.sum(flat_gt[flat_patch_selection])
            num_label = np.sum(flat_patch_selection)
            tree_probs[idx, 0] = num_pos / len(labels)
            tree_probs[idx, 1] = num_label / len(labels)
            tree_probs[idx, 2] = num_pos / num_label
            tree_probs[idx, 3] = num_pos / num_all_pos
        tree_probs[:, 2] /= np.max(tree_probs[:, 2])

        # SAVE CLUSTERS AND PROBS
        print("cluster probs:", tree_probs)
        self.embedder.patch_embed_clusters = embed_clusters
        self.embedder.patch_embed_cluster_tree_probs = tree_probs

        os.makedirs(self.embed_data_path, exist_ok=True)
        with open(self.embed_data_path + self.embed_data_args + ".pickle", "wb") as fp:
            pickle.dump((embed_clusters, tree_probs, self.args.embed_dim, pixel_clusters, self.embedder.w,
                         self.embedder.channels, self.embedder.norm), fp)

        print("Finished creating patch embedding clusters with tree probs")

    def vis_embed(self):
        args = self.args
        resultdir = f"./train/patch-embed/result-videos-3/"

        if args.integrated or args.final:
            resultdir = f"./train/patch-embed/integrated-result-videos/"
        result_args = f"{self.embed_data_args_specific}"
        os.makedirs(resultdir, exist_ok=True)

        # LOAD CLUSTERS AND PROBS
        embed_tuple_path = self.embed_data_path + self.embed_data_args + ".pickle"
        print(embed_tuple_path)
        if not os.path.exists(embed_tuple_path):
            print("no clusters and probs found...")
            self.create_patch_embedding_clusters()
        else:
            print("found clusters and probs...")
            self.embedder = PatchEmbedder(self.args.embed_dim, self.args.embed_cluster,
                                          pw=self.args.embed_patch_width,
                                          channels=[0] if self.args.hue else ([0, 1, 2] if self.args.hsv else [0, 1]),
                                          norm=self.args.embed_norm)
            self.embedder.load_embed_tuple(embed_tuple_path)

        threshold = self.embedder.convert_treshold(0.4)
        print("TRHESHOLD:", threshold)

        # GET DATA
        if self.args.dummy:
            tree = cv2.cvtColor(cv2.imread("train/navigate/tree.png"), cv2.COLOR_BGR2RGB)
            nav = cv2.cvtColor(cv2.imread("train/navigate/nav.png"), cv2.COLOR_BGR2RGB)
            X = np.stack((tree, nav), axis=0)
            X = rgb_to_hsv(X / 255)
        else:
            X = self.XX[:1000] / 255

            navdatadir = f"./train/navigate/test/{1000}"
            if not os.path.exists(navdatadir + "data.pickle"):
                self.collect_navigation_dataset(datadir=navdatadir)

            with gzip.open(navdatadir + "data.pickle", 'rb') as fp:
                NX, NY = pickle.load(fp)
                NY = NY[:, 0]
                NX = NX / 255
            print("loaded navigation test data:", NX.shape, NY.shape)

            # print(type(X), NX.shape)
            X = np.concatenate((X, NX), axis=0)

        if False:
            # MAKE PATCHES
            patches = self.embedder.make_patches(X, 8, 2)
            print("patches shape:", patches.shape)

            # CALC PROBS
            probs = self.embedder.calc_tree_probs_for_patches(patches, verbose=True)
            print("probs shape:", probs.shape)

        print("embedding test frames...", X.shape)
        batchsize = 512
        problist, labellist = [], []
        for bidx in range(0, 1024, batchsize):
            print("progress between", bidx / len(X))
            probs, labels = self.embedder.predict_batch(X[bidx:bidx + batchsize], verbose=True)
            labels = labels.astype(probs.dtype)
            problist.append(probs)
            labellist.append(labels)
        probs = np.concatenate(problist, axis=0)
        labels = np.concatenate(labellist, axis=0)
        print("labels shape", labels.shape, "probs shape:", probs.shape)

        rgb = hsv_to_rgb(X)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = resultdir + result_args + '.avi'
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (64 * 4, 64))
        print("generating result video:", video_path)
        for idx, frame in enumerate(probs):
            print("at frame:", idx, "/", len(probs), end='\r')
            resized_frame = np.ones((64, 64, 3)) * cv2.resize(frame, (64, 64))[:, :, None]
            clean_mask = resized_frame > threshold
            masked_rgb = rgb[idx] * clean_mask

            # labeled = np.ones((64, 64, 3))
            # labeled[:,:,0] = cv2.resize(labels[idx], (64,64))/self.embedder.n_cluster
            # labeled = hsv_to_rgb(labeled)

            pic = np.concatenate((rgb[idx], masked_rgb, clean_mask, resized_frame), axis=1)
            # plt.imsave(resultdir+f"{idx}.png", pic)
            uint8_bgr = cv2.cvtColor((255 * pic).astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(uint8_bgr)
        out.release()

    def visualize(self, online=False):
        args = self.args
        resultdir = self.path
        scale = 4
        ph = 32
        pad = 0
        os.makedirs(resultdir, exist_ok=True)

        # LOAD
        """
        if not args.purevis:
            if args.external_critic:
                if args.vgg:
                    critic = VGGCritic(resize=(args.resize,args.resize))
                else:
                    critic = Critic()
                critic.load_state_dict(T.load(args.external_critic))

                resultdir = "/".join(args.external_critic.split("/")[:-1]) +"/"
            #self.load_models([self.unetname])
            #critic = lambda x: self.unet.forward(x, critic=True)
        """

        # GET DATA
        """if online:
            X = []
            vid = cv2.VideoCapture("debug/dummy/live-clip-01.avi")
            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break
                X.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
            X = np.stack(X, axis=0)
        else:"""
        if args.trainasvis:
            # X = self.X[:, 0]
            # Y = self.Y[:, 0, args.rewidx]
            X, Y, I = self.clean_data() if args.cleaned else self.collect_data()

            X = X[:args.trainasvis]
            Y = Y[:, :args.trainasvis]
        else:
            print(self.XX.shape, self.YY.shape)
            X = self.XX
            Y = self.YY[args.rewidx]

        """
        if args.withnav:
            navdatadir = f"./train/navigate/test/{1000}"
            if not os.path.exists(navdatadir + "data.pickle"):
                self.collect_navigation_dataset(datadir=navdatadir)

            with gzip.open(navdatadir + "data.pickle", 'rb') as fp:
                NX, NY = pickle.load(fp)
                NY = NY[:, 0]
                NX = NX
            print("loaded navigation test data:", NX.shape, NY.shape)

            #print(type(X), NX.shape)
            X = np.concatenate((X,NX), axis=0)
            Y = np.concatenate((Y,NY), axis=0)
        """
        if args.purevis:
            idxs = [int(e) for e in args.purevis.split(",")]
            Y = Y[idxs]
            X = X[None]
        else:
            critic = self.critic.eval()
            masker = self.masker.eval()
            batchsize = 128
            preds = []
            masks = []
            sals = []
            with T.no_grad():
                for bidx in range(0, len(X), batchsize):
                    print("progress at", bidx / len(X), end="\r")
                    # print(Y.shape)
                    batch = T.from_numpy(X[bidx:bidx + batchsize]).permute(0, 3, 1, 2).float().to(self.device) / 255.0
                    print("TYPE", batch.device)
                    # print(batch)
                    pred, embeds = critic(batch, collect=True)
                    preds.append(pred.squeeze().cpu().numpy())
                    if args.vismasker:
                        mask = masker(batch, embeds)
                        masks.append(mask.cpu().numpy())
                    """
                    batch.requires_grad = True
                    pred.sum().backward()
                    preds.append(pred.detach().numpy())
                    sal = batch.grad.sum(dim=1)
                    sal = sal/(sal.flatten(1).max(dim=-1).values[:,None,None])
                    sal = sal.numpy()
                    salpos = (sal * (sal > 0))
                    salneg = np.abs(sal * (sal <= 0))
                    salpos = np.stack((salpos, salpos, salpos), -1)
                    salneg = np.stack((salneg, salneg, salneg), -1)
                    #print(salpos.shape)
                    sals.append(np.stack((salpos, salneg), axis=0))
                    """
            preds = np.concatenate(preds, axis=0)
            # sals = np.concatenate(sals, axis=1)
            # print("sals", sals.shape)
            Y = np.stack((Y, preds), axis=0)
            if masks:
                masks = np.concatenate(masks, axis=0)
                masks = np.concatenate((masks, masks, masks), axis=1)
                print("masks", masks.shape)
                masks = masks.transpose(0, 2, 3, 1)
                # print(X.max(), masks.max())
                X = np.stack((X, X * masks), axis=0)
            else:
                X = X[None]

            print("yshape", Y.shape)
            print("xshape", X.shape)

        def make_video(name, frames, values, sorting=None):
            framelist = []
            if sorting is not None:
                frames = frames[:, sorting]
                values = values[:, sorting]

            ph = 32
            plotbars = [make_plotbar(ph, 64, values[i]) for i in range(len(values))]
            ph = plotbars[0].shape[0] * len(plotbars)
            length = len(frames[0])
            video_path = resultdir + name
            width = scale * (frames.shape[3])
            height = scale * (frames.shape[2] * frames.shape[0] + ph)
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # out = cv2.VideoWriter(video_path, fourcc, 4.0, (width, height))
            print("video dimensions:", width, height)
            # font = ImageFont.truetype("./isy_minerl/segm/etc/Ubuntu-R.ttf", 11)
            for idx in range(length):
                if not idx % 100:
                    print("at frame:", idx, "/", length, end='\r')
                # RGB PICS
                pic = np.concatenate(frames[:, idx], axis=0)

                # PLOTS
                plots = []
                for plot_idx in range(len(plotbars)):
                    plot = plotbars[plot_idx][:, idx:idx + 64].copy()
                    plot[:, (64 // 2)] *= np.array((1, 0, 0))
                    plots.append(plot)
                plots = np.concatenate(plots, axis=0)
                # pic = np.concatenate((pic, pic*sals[0,cidx][:,:,None], pic*sals[1,cidx][:,:,None]))
                pic = np.concatenate((pic, plots), axis=0)

                # RESIZE
                pic = cv2.resize(pic, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                img = Image.fromarray(np.uint8(pic))

                # DRAWING
                draw = ImageDraw.Draw(img)
                h = pic.shape[0] - 12 - ph * scale
                draw.text((64 * scale - 26, h - 1), str(sorting[idx]) if sorting is not None else str(idx),
                          fill=(255, 255, 255))
                for val_idx in range(len(Y)):
                    draw.text((1, 1 + (15 * val_idx)), str(round(values[val_idx, idx].item(), 3)),
                              fill=(255, 255, 255))
                # plt.imsave(resultdir+f"{idx}.png", pic)

                # WRITING TO VIDEO
                uint8_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # plt.imshow(cv2.cvtColor(uint8_bgr, cv2.COLOR_BGR2RGB))
                # plt.show()
                # out.write(uint8_bgr)
                framelist.append(np.array(img))
            # out.release()

            # vidwrite(resultdir+"test.mp4", np.stack(framelist), framerate=4)
            vidwrite(video_path, np.stack(framelist), framerate=4)

        # padder = lambda x, v: np.pad(x, ((pad,pad),(pad,pad),(0,0)), mode='constant', constant_values=v)

        visname = args.visname
        make_video(f"{visname}.mp4", X, Y)
        sorting = np.argsort(Y[args.sortidx])[::-1]
        make_video(f"{visname}-pred-sorted.mp4", X, Y, sorting=sorting)
        if args.sortidx:
            sorting = sorting = np.argsort(Y[0])[::-1]
            make_video(f"{visname}-GT-sorted.mp4", X, Y, sorting=sorting)
        # ffmpeg.input(resultdir+f"{visname}.avi").output(resultdir+f"{visname}.mp4").run()

    def eval(self):
        args = self.args
        resultdir = self.path
        scale = 4
        ph = 32
        pad = 0
        os.makedirs(resultdir, exist_ok=True)

        critic = self.critic.to(self.device).eval()
        masker = self.masker.to(self.device).eval()
        if args.separate:
            sepcrit = self.sepcrit.to(self.device).eval()
        batchsize = 128
        preds = []
        M = []
        evaldatapath = "train/red-trees/"
        X = np.load(evaldatapath + "X.npy")/255.0
        print("X", X.shape, np.min(X), np.max(X))
        Y = np.expand_dims(np.all(np.load(evaldatapath + "Y.npy"), axis=-1), axis=-1)
        # print(Y)
        # print("Y min max", Y.shape, np.min(Y), np.max(Y))

        with T.no_grad():
            for bidx in range(0, len(X), batchsize):
                print("eval at", bidx / len(X), end="\r")
                # print(Y.shape)
                batch = T.from_numpy(X[bidx:bidx + batchsize]).permute(0, 3, 1, 2).float().to(
                    self.device)
                # print(batch)
                pred, embeds = critic(batch, collect=True)
                preds.append(pred.squeeze().cpu().numpy())

                if args.separate:
                    _, embeds = sepcrit(batch, collect=True)
                mask = masker(batch, embeds)
                M.append(mask.cpu().numpy())

        M = np.concatenate(M, axis=0)
        M = M.transpose(0, 2, 3, 1)
        M = (M > args.eval)
        preds = np.concatenate(preds, axis=0)
        highs = preds > args.high_rew_thresh
        lows = preds < args.low_rew_thresh
        # print("M", M.shape, "Y", Y.shape)

        # Y = Y(Y > 0.1)
        # intersection = np.sum((M & Y).reshape(M.shape[0],-1), axis=1)
        # union = np.sum((M | Y).reshape(M.shape[0],-1), axis=1)
        ious = []
        for select in []:  # [highs, lows]:
            intersection = np.sum(M[select] & Y[select])
            union = np.sum(M[select] | Y[select])
            iou = intersection / union
            print("RESULTS", intersection, union, iou)
            ious.append(iou)
        # iou = sum(ious)/len(ious)
        # print("AVERAGE", iou)

        intersection = np.sum(M & Y)
        union = np.sum(M | Y)
        iou = intersection / union
        iou = round(iou, 3)
        intersection = np.sum(M[highs] & Y[highs])
        union = np.sum(M[highs] | Y[highs])
        highiou = intersection / union
        hiou = round(highiou, 3)


        print(f"\nRESULTS EVAL", "high percentage", round(sum(highs) / len(M), 2), "ious", iou, hiou)
        # self.ious = (iou, highiou)
        critic = self.critic.to(self.device).train()
        masker = self.masker.to(self.device).train()

        if iou>self.ious[0] and args.visbesteval:
            M = np.concatenate((M, M, M), axis=-1)
            Y = np.concatenate((Y, Y, Y), axis=-1)
            frames = np.concatenate((X, Y, M), axis=2)
            # frames = np.concatenate((X, np.stack((M,M,M), axis=-1), np.stack((Y,Y,Y), axis=-1)), axis=1)
            frames = (frames * 255).astype(np.uint8)
            vidwrite(resultdir + f"iou={round(iou, 3)}.mp4", frames, framerate=4)

        return iou, hiou

    def vis_unet(self, online=False):
        args = self.args
        resultdir = self.result_path

        os.makedirs(resultdir, exist_ok=True)

        # LOAD UNET
        self.load_models([self.unetname])
        critic = self.unet.critic
        unet = self.unet

        # GET DATA
        if online:
            X = []
            vid = cv2.VideoCapture("debug/dummy/live-clip-01.avi")
            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break
                X.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
            X = np.stack(X, axis=0)
        else:
            # print(len(self.XX))
            X = self.XX[:1000]

            navdatadir = f"./train/navigate/test/{1000}"
            if not os.path.exists(navdatadir + "data.pickle"):
                self.collect_navigation_dataset(datadir=navdatadir)

            with gzip.open(navdatadir + "data.pickle", 'rb') as fp:
                NX, NY = pickle.load(fp)
                NY = NY[:, 0]
                NX = NX
            print("loaded navigation test data:", NX.shape, NY.shape)

            # print(type(X), NX.shape)
            X = np.concatenate((X, NX), axis=0)

        batchsize = 512
        masks = []
        for bidx in range(0, len(X), batchsize):
            print("progress at", bidx / len(X))
            with T.no_grad():
                batch = T.from_numpy(X[bidx:bidx + batchsize]).permute(0, 3, 1, 2).float()
                # print(batch)
                mask = unet(batch).numpy()
            masks.append(mask)
        masks = np.concatenate(masks, axis=0).transpose(0, 2, 3, 1)

        rgb = hsv_to_rgb(X / 255) * 255
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = resultdir + ('offline-eval.avi' if not online else 'online-eval.avi')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (6 * 66 * 3, 6 * 66))
        padder = lambda x, v: np.pad(x, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=v)
        print("generating result video:", video_path)
        for idx, mask in enumerate(masks):
            print("at frame:", idx, "/", len(masks), end='\r')
            mask = np.ones(rgb[idx].shape) * mask
            masked_rgb = rgb[idx] * mask
            # print(mask.shape)
            mask = padder(mask, 0.5)
            masked_rgb = padder(masked_rgb, 125)
            pov = padder(rgb[idx], 125)
            pic = np.concatenate((pov, masked_rgb, 255 * mask), axis=1)
            pic = cv2.resize(pic, (0, 0), fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
            # plt.imsave(resultdir+f"{idx}.png", pic)
            uint8_bgr = cv2.cvtColor((pic).astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(uint8_bgr)
        out.release()

    def vis_white_tree(self):
        resultdir = f"./train/patch-embed/result-videos-2/"
        result_args = f"{self.embed_data_args}"
        os.makedirs(resultdir, exist_ok=True)

        # LOAD CLUSTERS AND PROBS
        embed_tuple_path = self.embed_data_path + self.embed_data_args + ".pickle"
        print(embed_tuple_path)
        if not os.path.exists(embed_tuple_path):
            print("no clusters and probs found...")
            self.create_patch_embedding_clusters()
        else:
            print("found clusters and probs...")
            self.embedder = PatchEmbedder(self.args.embed_dim, self.args.embed_cluster,
                                          pw=self.args.embed_patch_width,
                                          channels=[0] if args.hue else ([0, 1, 2] if args.hsv else [0, 1]))
            self.embedder.load_embed_tuple(embed_tuple_path)

        # GET DATA
        if self.args.dummy:
            tree = cv2.cvtColor(cv2.imread("train/navigate/tree.png"), cv2.COLOR_BGR2RGB)
            nav = cv2.cvtColor(cv2.imread("train/navigate/nav.png"), cv2.COLOR_BGR2RGB)
            X = np.stack((tree, nav), axis=0)
            X = rgb_to_hsv(X / 255)
        else:
            X = self.XX[:330] / 255

        if False:
            # MAKE PATCHES
            patches = self.embedder.make_patches(X, 8, 2)
            print("patches shape:", patches.shape)

            # CALC PROBS
            probs = self.embedder.calc_tree_probs_for_patches(patches, verbose=True)
            print("probs shape:", probs.shape)

        print("embedding batch...", X.shape)
        probs, labels = self.embedder.predict_batch(X, verbose=True)
        labels = labels.astype(probs.dtype)
        print("labels shape", labels.shape, "probs shape:", probs.shape)

        y1, y2, x1, x2 = 0.3, 0.7, 0.75, 1
        # white_tree = cv2.cvtColor(cv2.imread("white_tree.png"), cv2.COLOR_BGR2HSV)
        white_tree = (X[324] * 255).astype(np.uint8)
        cv2.imwrite("white_tree.png", cv2.cvtColor(white_tree, cv2.COLOR_HSV2BGR))
        crop = white_tree[int(64 * y1):int(64 * y2), int(64 * x1):int(64 * x2)]
        cv2.imwrite("white_tree_crop.png", cv2.cvtColor(crop, cv2.COLOR_HSV2BGR))
        _, wtlabels = self.embedder.predict_batch(white_tree[None] / 255.0)
        wtlabels = wtlabels[0, int(28 * y1):int(28 * y2), int(28 * x1):int(28 * x2)].reshape(-1)
        wtlabelset = np.argsort([np.sum(wtlabels == i) for i in range(self.embedder.embed_dim)])[::-1][:3]
        print(wtlabels, wtlabelset)

        for idx, label in enumerate(wtlabelset):
            top_pix_colors = self.embedder.patch_label_to_color(label)
            cv2.imwrite(
                f"eval/white_tree_{'hue' if self.args.hue else 'hs'}_2_patch_cluster_top_pixel_colors_{idx}.png",
                top_pix_colors)

        def clean(labels):
            for y in range(labels.shape[0]):
                for x in range(labels.shape[1]):
                    if not labels[y, x] in wtlabelset:
                        labels[y, x] = 0
            return labels

        rgb = hsv_to_rgb(X)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(resultdir + result_args + '.avi', fourcc, 20.0, (64 * 5, 64))
        os.makedirs(resultdir + result_args, exist_ok=True)
        for idx, frame in enumerate(probs):
            print("visualizing results, at frame:", idx, "/", len(probs), end='\r')
            resized_frame = np.ones((64, 64, 3)) * cv2.resize(frame, (64, 64))[:, :, None]
            clean_mask = resized_frame > 0.7
            masked_rgb = rgb[idx] * resized_frame

            labeled = np.ones((64, 64, 3))
            frame_labels = cv2.resize(labels[idx], (64, 64))
            cleaned_frame_labels = clean(frame_labels)
            labeled[:, :, 2] = cleaned_frame_labels / self.embedder.n_cluster
            labeled[:, :, 0] = frame_labels / self.embedder.n_cluster
            labeled = hsv_to_rgb(labeled)

            pic = np.concatenate((rgb[idx], masked_rgb, resized_frame, clean_mask, labeled), axis=1)

            plt.imsave(resultdir + result_args + f"/{idx}.png", pic)
            uint8_bgr = cv2.cvtColor((255 * pic).astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(uint8_bgr)
        out.release()

    def vis_pixels(self):
        # GET PIXELS
        # data = self.X[self.Y[:,3]>0.9]
        # data = data[:,10:54,26:39]
        data = self.X
        # navdatadir = "./data/navigate/"
        # with gzip.open(navdatadir+"data.pickle", 'rb') as fp:
        #    data, NY = pickle.load(fp)
        # print(data.shape)
        pixels = data.reshape(-1, 3)

        # PLOTS SETUP
        my_cmap = copy.copy(cm.get_cmap('plasma'))
        my_cmap.set_bad(my_cmap.colors[0])
        # print(cm.cmaps_listed)
        hs_pic = np.array([[[h, s, 1] for s in range(255)] for h in range(255)])
        hs_pic = 255 * hsv_to_rgb(hs_pic / 255)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        ax1.set_aspect(1)
        ax2.set_aspect(1)
        ax3.set_aspect(1)

        ax2.imshow(hs_pic)
        ax2.invert_yaxis()
        ax1.hist2d(pixels[:, 0], pixels[:, 1], bins=100, norm=colors.LogNorm(), cmap=my_cmap)
        # plt.gca().set_aspect('equal', adjustable='box')

        # plt.show()
        X = pixels[::200, :2]
        print(X.shape)
        comps = 100
        gmm = GMM(n_components=comps)
        labels = gmm.fit_predict(X)
        pixperlabel = [np.sum(labels == i) / len(labels) for i in range(comps)]
        normed_ppl_perpix = (pixperlabel / max(pixperlabel))[labels]
        print(sorted(pixperlabel))
        ax3.scatter(X[:, 0], X[:, 1], c=labels, s=0.5, cmap='jet')
        ax3.set_xlim(0, 255)
        ax3.set_ylim(0, 255)
        plt.tight_layout()
        plt.show()

    def collect_split_dataset(self, path, size=2000, wait=10, test=0, datadir="./train/stuff/"):
        args = self.args
        os.makedirs(datadir + "samples/", exist_ok=True)
        # os.environ["MINERL_DATA_ROOT"] = "./data"
        if not os.path.exists(f"{os.getenv('MINERL_DATA_ROOT', 'data/')}/MineRLTreechopVectorObf-v0"):
            minerl.data.download(os.getenv('MINERL_DATA_ROOT', 'data/'), experiment='MineRLTreechopVectorObf-v0')
        data = minerl.data.make('MineRLTreechopVectorObf-v0', data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
                                num_workers=args.workers[0], worker_batch_size=args.workers[1])
        X = []
        Y = []
        cons = self.args.cons
        wait = self.args.wait
        delay = self.args.delay
        warmup = self.args.warmup
        chunksize = self.args.chunksize

        print("collecting data set with", size, "frames")
        for b_idx, (state, act, rew, next_state, done) in enumerate(data.batch_iter(test or 10,
                                                                                    2 * wait if not test else size,
                                                                                    preload_buffer_size=args.workers[
                                                                                        2])):
            print("at batch", b_idx, end='\r')
            # vector = state['vector']

            # CONVERt COLOR
            pov = state['pov']
            if self.args.color == "HSV":
                pov = (255 * rgb_to_hsv(pov / 255)).astype(np.uint8)

            rewards = []
            approaches = []
            if test:
                for rowidx in range(len(rew)):
                    rewards.extend(rew[rowidx])
                    approaches.extend(pov[rowidx])
            else:
                chops = [(i, pos) for (i, pos) in enumerate(np.argmax(rew == 1, axis=1)) if pos > wait]
                # print(np.max(rew, axis=1))
                # print(chops)
                for chopidx, (rowidx, tidx) in enumerate(chops):
                    rewards.extend([0] * chunksize)
                    approaches.extend(pov[rowidx, warmup:warmup + chunksize])
                    rewards.extend([1] * chunksize)
                    approaches.extend(pov[rowidx, tidx - chunksize + 1 - delay:tidx + 1 - delay])

                    if len(X) < 500:  # SAVE IMG
                        effchsize = chunksize * 2
                        for chunkidx in range(effchsize):
                            img = Image.fromarray(
                                np.uint8(255 * hsv_to_rgb(approaches[chopidx * effchsize + chunkidx] / 255)))
                            # draw = ImageDraw.Draw(img)
                            # x, y = 0, 0
                            # draw.text((x, y), "\n".join([str(round(entry,3))
                            # for entry in rewtuple]), fill= (255,255,255), font=self.font)
                            img.save(datadir + "samples/" + f"{b_idx}-{chopidx}-{chunkidx}-" +
                                     f"{'A' if rewards[chopidx * effchsize + chunkidx] else 'B'}.png")

            if approaches:
                X.extend(approaches)
                Y.extend(rewards)

            if test:
                break

            print(len(X))
            if len(X) >= size:
                X = X[:size]
                Y = Y[:size]
                break

        X = np.array(X, dtype=np.uint8)
        Y = np.array(Y)
        with gzip.GzipFile(path, 'wb') as fp:
            pickle.dump((X, Y), fp)

    def collect_navigation_dataset(self, datadir="./train/navigate/train/2000", nperchunk=50):
        args = self.args
        print("Collecting nav dataset...")
        os.makedirs(datadir + "samples/", exist_ok=True)
        if not os.path.exists(f"{os.getenv('MINERL_DATA_ROOT', 'data/')}/MineRLNavigateVectorObf-v0"):
            minerl.data.download(os.getenv('MINERL_DATA_ROOT', 'data/'), experiment='MineRLNavigateVectorObf-v0')
        data = minerl.data.make('MineRLNavigateVectorObf-v0', data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
                                num_workers=args.workers[0], worker_batch_size=args.workers[1])
        names = data.get_trajectory_names()

        # INIT STRUCTURES
        X = []
        Y = []
        n_per_chunk = nperchunk
        skip = 20 * nperchunk
        datasetsize = int(datadir.split('/')[-1])

        # ITER OVER EPISODES
        for fridx, name in enumerate(names):
            # EXTRACT EPISODE
            state, action, reward, state_next, done = zip(*data.load_data(name))

            # CONVERT COLOR
            pov = np.array([s['pov'] for s in state])
            if self.args.color == "HSV":
                pov = (255 * rgb_to_hsv(pov / 255)).astype(np.uint8)

            selections = [pov[skip * i:skip * i + n_per_chunk] for i in range(len(pov) // skip)]
            # rewards = np.zeros((len(selections), 1))

            if args.viz:
                if len(X) < 300:
                    for chidx, chunk in enumerate(selections):
                        for fi, frame in enumerate(chunk):
                            img = Image.fromarray(np.uint8(255 * hsv_to_rgb(frame / 255)))
                            draw = ImageDraw.Draw(img)
                            # print(rewards)
                            rewtuple = (0,)
                            x, y = 0, 0
                            draw.text((x, y), "\n".join([str(round(entry, 3)) for entry in rewtuple]),
                                      fill=(255, 255, 255), font=self.font)
                            img.save(datadir + "samples/" + f"{name}-{chidx}-{fi}.png")

            episode_chunks = []
            for chunk in selections:
                episode_chunks.extend(chunk)
            X.extend(episode_chunks)
            if len(X) > datasetsize:
                break

        # CONVERT TO FINAL ARRAYS
        X = np.array(X, dtype=np.uint8)
        Y = np.zeros((X.shape[0], 1))

        # SAVE AS ZIPPED FILE
        with gzip.GzipFile(datadir + "data.pickle", 'wb') as fp:
            pickle.dump((X, Y), fp)

    def collect_discounted_dataset(self, path, size=2000, datadir="./train/stuff/", test=0):
        args = self.args
        os.makedirs(datadir + "samples/", exist_ok=True)
        # os.environ["MINERL_DATA_ROOT"] = "./data"
        if not os.path.exists(f"{os.getenv('MINERL_DATA_ROOT', 'data/')}/MineRLTreechopVectorObf-v0"):
            minerl.data.download(os.getenv('MINERL_DATA_ROOT', 'data/'), experiment='MineRLTreechopVectorObf-v0')
        data = minerl.data.make('MineRLTreechopVectorObf-v0', data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
                                num_workers=args.workers[0], worker_batch_size=args.workers[1])
        names = data.get_trajectory_names()
        X = []
        Y = []
        cons = self.args.cons
        delay = self.args.delay
        delta = self.args.delta
        gamma = self.args.gamma
        revgamma = self.args.revgamma
        trajsize = self.args.trajsize
        if test:
            testsize = size
            size = size * test

        print("collecting data set with", size, "frames")
        # for b_idx, (state, act, reward, next_state, done) in
        # enumerate(data.batch_iter(test or 10, cons if not test else testsize, preload_buffer_size=args.workers[2])):
        for fridx, name in enumerate(names):
            # EXTRACT EPISODE
            state, action, reward, state_next, done = zip(*data.load_data(name))

            # CONVERT COLOR
            pov = np.array([s['pov'] for s in state])
            if self.args.color == "HSV":
                pov = (255 * rgb_to_hsv(pov / 255)).astype(np.uint8)

            # DETECT AND FILTER CHOPS
            chops = np.nonzero(reward)[0]
            deltas = chops[1:] - chops[:-1]
            big_enough_delta = deltas > 50
            chops = np.concatenate((chops[None, 0], chops[1:][big_enough_delta]))
            # print(chops)

            # INIT EPISODE SET
            approaches = []
            rewards = []

            # VERIFY CHOPS AND SEQUENCES
            if chops.size == 0:
                continue
            end = np.max(chops)
            sequ = pov[:end + 1]
            reward = reward[:end + 1]
            assert reward[-1] > 0, "ERROR wrong chop detection"

            # INIT DISCOUNT
            delaycount = delay
            rowrew = []
            selection = []
            addfak = 0
            revaddfak = 0
            relchopidx = 0
            chopidx = -1

            # DISCOUNT LOOP
            for i in range(1, len(reward) + 1):
                delaycount -= 1
                relchopidx -= 1

                # RESET
                if reward[-i] > 0:
                    if len(reward) + i == chops[chopidx]:
                        relchopidx = 0
                        chopidx -= 1
                    fak = 1  # exponential
                    sub = 0  # subtraction
                    addfak += 1  # exponanential with add-reset
                    revfak = 1
                    revaddfak += 1
                    revhelper = 0.01
                    # fak = 0
                    delaycount = delay

                # DELAY AND TRAJECTORY SKIP
                if delaycount > 0 or relchopidx <= -trajsize - delay:
                    continue

                # STORE REWARDS AND INDEXES
                selection.append(-i)
                rewtuple = (relchopidx, fak, addfak, revfak, revaddfak, sub)
                rowrew.append(rewtuple)

                # DISCOUNT FAKTORS
                fak *= gamma
                sub -= 1
                addfak *= gamma
                revfak = max(revfak - revhelper, 0)
                revaddfak = max(revaddfak - revhelper, 0)
                revhelper *= revgamma

            # EXTEND EPISODE SET
            # print(row)
            rewards.extend(rowrew[::-1])
            approaches.extend(sequ[selection[::-1]])

            # SAVE SAMPLE IMGS
            if args.viz:
                if len(X) < 300:
                    for fi, frame in enumerate(approaches):
                        img = Image.fromarray(np.uint8(255 * hsv_to_rgb(frame / 255)))
                        draw = ImageDraw.Draw(img)
                        # print(rewards)
                        rewtuple = rewards[fi]
                        x, y = 0, 0
                        draw.text((x, y), "\n".join([str(round(entry, 3)) for entry in rewtuple]),
                                  fill=(255, 255, 255), font=self.font)
                        img.save(datadir + "samples/" + f"{name}-{fi}.png")

            # EXTEND FULL DATA SET
            if approaches:
                X.extend(approaches)
                Y.extend(rewards)
                print("dataset size", len(X))

            # QUIT IF SIZE REACHED
            if len(X) >= size:
                X = X[:size]
                Y = Y[:size]
                break

        # CONVERT TO FINAL ARRAYS
        X = np.array(X, dtype=np.uint8)
        Y = np.array(Y)

        # SAVE AS ZIPPED FILE
        with gzip.GzipFile(path, 'wb') as fp:
            pickle.dump((X, Y), fp)

    def collect_sarsa_dataset(self, path, size=2000, datadir="./train/stuff/", test=0):
        args = self.args
        # path = path +"sarsa/"
        # os.makedirs(path, exist_ok=True)
        os.makedirs(datadir + "samples/", exist_ok=True)
        # os.environ["MINERL_DATA_ROOT"] = "./data"
        if not os.path.exists(f"{os.getenv('MINERL_DATA_ROOT', 'data/')}/MineRLTreechopVectorObf-v0"):
            minerl.data.download(os.getenv('MINERL_DATA_ROOT', 'data/'), experiment='MineRLTreechopVectorObf-v0')
        data = minerl.data.make('MineRLTreechopVectorObf-v0', data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
                                num_workers=args.workers[0], worker_batch_size=args.workers[1])
        names = data.get_trajectory_names()
        np.random.default_rng().shuffle(names)
        X = []
        Y = []
        cons = self.args.cons
        delay = self.args.delay
        delta = self.args.delta
        # gamma = self.args.gamma
        gamma = 0.9
        revgamma = self.args.revgamma
        trajsize = self.args.trajsize
        fskip = args.fskip
        skip = args.chunksize
        if test:
            skip = 1

        print("collecting data set with", size, "frames")
        # for b_idx, (state, act, reward, next_state, done) in
        # enumerate(data.batch_iter(test or 10, cons if not test else testsize, preload_buffer_size=args.workers[2])):
        for fridx, name in enumerate(names):
            print(name)
            # EXTRACT EPISODE
            state, action, reward, state_next, done = zip(*data.load_data(name))

            # CONVERT COLOR
            pov = np.array([s['pov'] for s in state])
            if self.args.color == "HSV":
                pov = (255 * rgb_to_hsv(pov / 255)).astype(np.uint8)

            # INIT EPISODE SET
            approaches = []
            rewards = []
            selection = []

            # DISCOUNT LOOP
            """
            delay_count = 0
            carry = 0
            for i in range(1, len(reward)+1):
                if delay_count>0:
                    delay_count -= 1
                    continue
                carry = reward[-i] + gamma*carry
                if reward[-i]:
                    delay_count = delay
                    continue
                raw_rew = reward[-i]
                rewards.append((raw_rew, carry))
                selection.append(-i)

            # EXTEND EPISODE SET
            rewards = rewards[::-1]
            approaches.extend(pov[selection[::-1]])
            """
            # INIT DISCOUNT
            delaycount = delay
            rowrew = []
            selection = []
            fak = 0
            revfak = 0
            sub = 0
            addfak = 0
            revaddfak = 0
            revhelper = 0.01
            relchopidx = 0
            chopidx = -1

            # DISCOUNT LOOP
            i = 0
            while i < (len(reward) + 1):
                # delaycount -= 1
                # RESET
                latest_rew = reward[-i]
                if reward[-i] > 0:
                    lowest = max(-i - delay, -len(reward))
                    comeup = reward[lowest: -i][::-1]
                    # print(comeup, -i, -i-delay, -len(reward))
                    if sum(comeup) > 0:
                        # print(comeup)
                        move = np.min(np.nonzero(comeup))
                        # print(move)
                        i += move + 1
                        # print(reward[-i])
                        continue
                    fak = 1  # exponential
                    sub = 0  # subtraction
                    addfak += 1  # exponanential with add-reset
                    revfak = 1
                    revaddfak += 1
                    revhelper = 0.01
                    # fak = 0
                    # delaycount = delay

                # DELAY AND TRAJECTORY SKIP
                # if delaycount > 0:
                #    continue

                # STORE REWARDS AND INDEXES
                selection.append(-i)
                rewtuple = (latest_rew, fak, addfak, revfak, revaddfak, sub)
                rowrew.append(rewtuple)
                latest_rew = 0

                # DISCOUNT FAKTORS
                fak *= gamma
                sub -= 1
                addfak *= gamma
                revfak = max(revfak - revhelper, 0)
                revaddfak = max(revaddfak - revhelper, 0)
                revhelper *= revgamma

                i += 1

            rewards.extend(rowrew[::-1])
            approaches.extend(pov[selection[::-1]])

            # SAVE SAMPLE IMGS
            if args.viz:
                if len(X) < 300:
                    for fi, frame in enumerate(approaches):
                        img = Image.fromarray(np.uint8(255 * hsv_to_rgb(frame / 255)))
                        draw = ImageDraw.Draw(img)
                        # print(rewards)
                        rewtuple = rewards[fi]
                        x, y = 0, 0
                        draw.text((x, y), "\n".join([str(round(entry, 3)) for entry in rewtuple]),
                                  fill=(255, 255, 255), font=self.font)
                        img.save(datadir + "samples/" + f"{name}-{fi}.png")

            # EXTEND FULL DATA SET
            if len(approaches) > fskip + 1:
                ep_len = (len(approaches) // fskip) * fskip
                # approaches = [approaches[i:i+fskip] for i in range(0, ep_len, skip)]
                # rewards = [rewards[i:i+fskip] for i in range(0, ep_len, skip)]
                approaches = np.array(approaches)
                rewards = np.array(rewards)
                approaches = [approaches[[i, i + fskip]] for i in range(0, ep_len - fskip - 1, skip)]
                rewards = [rewards[[i, i + fskip]] for i in range(0, ep_len - fskip - 1, skip)]

                # add to full dataset
                X.extend(approaches)
                Y.extend(rewards)
                print("dataset size", len(X))

            # QUIT IF SIZE REACHED
            if len(X) >= size:
                X = X[:size]
                Y = Y[:size]
                break

        # CONVERT TO FINAL ARRAYS
        X = np.array(X, dtype=np.uint8)
        Y = np.array(Y)

        # SAVE AS ZIPPED FILE
        with gzip.GzipFile(path, 'wb') as fp:
            pickle.dump((X, Y), fp)

    def collect_sarsa_dataset_backup(self, path, size=2000, datadir="./train/stuff/", test=0):
        args = self.args
        # path = path +"sarsa/"
        # os.makedirs(path, exist_ok=True)
        os.makedirs(datadir + "samples/", exist_ok=True)
        # os.environ["MINERL_DATA_ROOT"] = "./data"
        if not os.path.exists(f"{os.getenv('MINERL_DATA_ROOT', 'data/')}/MineRLTreechopVectorObf-v0"):
            minerl.data.download(os.getenv('MINERL_DATA_ROOT', 'data/'), experiment='MineRLTreechopVectorObf-v0')
        data = minerl.data.make('MineRLTreechopVectorObf-v0', data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
                                num_workers=args.workers[0], worker_batch_size=args.workers[1])
        names = data.get_trajectory_names()
        X = []
        Y = []
        cons = self.args.cons
        delay = self.args.delay
        delta = self.args.delta
        gamma = self.args.gamma
        revgamma = self.args.revgamma
        trajsize = self.args.trajsize
        if test:
            testsize = size
            size = size * test

        print("collecting data set with", size, "frames")
        # for b_idx, (state, act, reward, next_state, done) in
        # enumerate(data.batch_iter(test or 10, cons if not test else testsize, preload_buffer_size=args.workers[2])):
        for fridx, name in enumerate(names):
            # EXTRACT EPISODE
            state, action, reward, state_next, done = zip(*data.load_data(name))

            # CONVERT COLOR
            pov = np.array([s['pov'] for s in state])
            if self.args.color == "HSV":
                pov = (255 * rgb_to_hsv(pov / 255)).astype(np.uint8)

            # DETECT AND FILTER CHOPS
            chops = np.nonzero(reward)[0]
            deltas = chops[1:] - chops[:-1]
            big_enough_delta = deltas > 50
            chops = np.concatenate((chops[None, 0], chops[1:][big_enough_delta]))
            # print(chops)

            # INIT EPISODE SET
            approaches = []
            rewards = []

            # VERIFY CHOPS AND SEQUENCES
            if chops.size == 0:
                continue
            end = np.max(chops)
            sequ = pov[:end + 1]
            reward = reward[:end + 1]
            assert reward[-1] > 0, "ERROR wrong chop detection"

            # INIT DISCOUNT
            delaycount = delay
            rowrew = []
            selection = []
            addfak = 0
            revaddfak = 0
            relchopidx = 0
            chopidx = -1

            # DISCOUNT LOOP
            for i in range(1, len(reward) + 1):
                delaycount -= 1
                relchopidx -= 1

                # RESET
                if reward[-i] > 0:
                    if len(reward) + i == chops[chopidx]:
                        relchopidx = 0
                        chopidx -= 1
                    fak = 1  # exponential
                    sub = 0  # subtraction
                    addfak += 1  # exponanential with add-reset
                    revfak = 1
                    revaddfak += 1
                    revhelper = 0.01
                    # fak = 0
                    delaycount = delay

                # DELAY AND TRAJECTORY SKIP
                if delaycount > 0 or relchopidx <= -trajsize - delay:
                    continue

                # STORE REWARDS AND INDEXES
                selection.append(-i)
                rewtuple = (relchopidx, fak, addfak, revfak, revaddfak, sub)
                rowrew.append(rewtuple)

                # DISCOUNT FAKTORS
                fak *= gamma
                sub -= 1
                addfak *= gamma
                revfak = max(revfak - revhelper, 0)
                revaddfak = max(revaddfak - revhelper, 0)
                revhelper *= revgamma

            # EXTEND EPISODE SET
            # print(row)
            rewards.extend(rowrew[::-1])
            approaches.extend(sequ[selection[::-1]])

            # SAVE SAMPLE IMGS
            if args.viz:
                if len(X) < 300:
                    for fi, frame in enumerate(approaches):
                        img = Image.fromarray(np.uint8(255 * hsv_to_rgb(frame / 255)))
                        draw = ImageDraw.Draw(img)
                        # print(rewards)
                        rewtuple = rewards[fi]
                        x, y = 0, 0
                        draw.text((x, y), "\n".join([str(round(entry, 3)) for entry in rewtuple]),
                                  fill=(255, 255, 255), font=self.font)
                        img.save(datadir + "samples/" + f"{name}-{fi}.png")

            # EXTEND FULL DATA SET
            if len(approaches) > 1:
                obs_with_next = zip(approaches[:-1], approaches[1:])
                r_with_next = zip(rewards[:-1], rewards[1:])
                X.extend(obs_with_next)
                Y.extend(r_with_next)
                print("dataset size", len(X))

            # QUIT IF SIZE REACHED
            if len(X) >= size:
                X = X[:size]
                Y = Y[:size]
                break

        # CONVERT TO FINAL ARRAYS
        X = np.array(X, dtype=np.uint8)
        Y = np.array(Y)

        # SAVE AS ZIPPED FILE
        with gzip.GzipFile(path, 'wb') as fp:
            pickle.dump((X, Y), fp)

    def collect_discounted_dataset_old(self, path, size=2000, datadir="./train/stuff/", test=0):
        os.makedirs(datadir + "samples/", exist_ok=True)
        os.environ["MINERL_DATA_ROOT"] = "./data"
        # minerl.data.download("./data", experiment='MineRLTreechopVectorObf-v0')
        # data = minerl.data.make('MineRLTreechopVectorObf-v0')
        data = minerl.data.make('MineRLTreechopVectorObf-v0',
                                data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
                                num_workers=args.workers[0],
                                worker_batch_size=args.workers[1])
        X = []
        Y = []
        cons = self.args.cons
        delay = self.args.delay
        delta = self.args.delta
        if test:
            testsize = size
            size = size * test

        print("collecting data set with", size, "frames")
        for b_idx, (state, act, rew, next_state, done) in enumerate(
                data.batch_iter(test or 10, cons if not test else testsize, preload_buffer_size=args.workers[2])):
            print("at batch", b_idx, end='\n')
            # vector = state['vector']

            # CONVERT COLOR
            pov = state['pov']
            if self.args.color == "HSV":
                pov = (255 * rgb_to_hsv(pov / 255)).astype(np.uint8)

            gamma = self.args.gamma
            revgamma = self.args.revgamma
            stepsize = 2
            # chops = [(i,pos) for (i,pos) in enumerate(np.argmax(rew, axis=1)) if pos>wait+stepsize*cons]
            # chops = [(i,pos) for (i,pos) in enumerate(np.argmax(rew, axis=1)) if pos>wait and pos <cons]
            approaches = []
            rewards = []
            # rewimg = pov[rew==1][0]
            # print(rewimg.shape)
            # plt.imsave(f"./results/Critic/stuff/rewimg-{b_idx}.png", hsv_to_rgb(revimg/255))
            for ri, orow in enumerate(rew):
                chops = np.nonzero(orow)[0]
                deltas = chops[1:] - chops[:-1]
                big_enough_delta = deltas > 50
                chops = np.concatenate((chops[None, 0], chops[1:][big_enough_delta]))
                # print(chops, row)
                if chops.size == 0:
                    continue
                end = np.max(chops) if not test else testsize - 1
                sequ = pov[ri, :end + 1]
                orow = orow[:end + 1]
                assert test or orow[-1] > 0, "ERROR wrong chop detection"

                delaycount = delay
                rowrew = []
                selection = []
                addfak = 0
                revaddfak = 0
                if test:
                    fak = 0
                    sub = 0
                    revfak = 0
                    revhelper = 0.01

                for i in range(1, len(orow) + 1):
                    delaycount -= 1

                    if orow[-i] > 0:  # RESET
                        fak = 1  # exponential
                        sub = 0  # subtraction
                        addfak += 1  # exponanential with add-reset
                        revfak = 1
                        revaddfak += 1
                        revhelper = 0.01
                        # fak = 0
                        delaycount = delay
                    if delaycount > 0:
                        continue
                    selection.append(-i)
                    rewtuple = (fak, addfak, revfak, revaddfak, sub)
                    rowrew.append(rewtuple)

                    # DISCOUNT
                    fak *= gamma
                    sub -= 1
                    addfak *= gamma
                    revfak = max(revfak - revhelper, 0)
                    revaddfak = max(revaddfak - revhelper, 0)
                    revhelper *= revgamma

                # print(row)
                rewards.extend(rowrew[::-1])
                approaches.extend(sequ[selection[::-1]])

            if len(X) < 300:  # SAVE IMG
                for fi, frame in enumerate(approaches):
                    img = Image.fromarray(np.uint8(255 * hsv_to_rgb(frame / 255)))
                    draw = ImageDraw.Draw(img)
                    # print(rewards)
                    rewtuple = rewards[fi]
                    x, y = 0, 0
                    draw.text((x, y), "\n".join([str(round(entry, 3)) for entry in rewtuple]),
                              fill=(255, 255, 255), font=self.font)
                    img.save(datadir + "samples/" + f"{b_idx}-{fi}.png")

            # print(approaches)
            if approaches:
                X.extend(approaches)
                Y.extend(rewards)

            # print(len(X))

            if len(X) >= size:
                X = X[:size]
                Y = Y[:size]
                break

        # CONVERT TO FINAL ARRAYS
        X = np.array(X, dtype=np.uint8)
        Y = np.array(Y)

        with gzip.GzipFile(path, 'wb') as fp:
            pickle.dump((X, Y), fp)

    def collect_water(self):
        # print(os.getenv('MINERL_DATA_ROOT', 'data/'))
        args = self.args
        size = args.waterdatasize
        waterpath = "train/water/"
        watername = f"{size}.pickle"
        print(f"Collecting {size} Water Frames into {waterpath}...")
        if os.path.exists(waterpath + watername):
            with gzip.open(waterpath + watername, 'rb') as fp:
                return pickle.load(fp)

        if not os.path.exists(f"{os.getenv('MINERL_DATA_ROOT', 'data/')}/MineRLNavigateDenseVectorObf-v0"):
            minerl.data.download(os.getenv('MINERL_DATA_ROOT', 'data/'), experiment='MineRLNavigateDenseVectorObf-v0')
        data = minerl.data.make('MineRLNavigateDenseVectorObf-v0', data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
                                num_workers=1, worker_batch_size=1)
        water = []
        count = 0
        for state, action, reward, next_state, done in data.batch_iter(
                batch_size=32, num_epochs=1, seq_len=100, preload_buffer_size=1):

            state = state['pov']
            if self.args.color == "HSV":
                state = (255 * rgb_to_hsv(state / 255)).astype(np.uint8)

            for si, pov in enumerate(state):
                for fi, frame in enumerate(pov):
                    count += 1
                    # img = Image.fromarray(frame)
                    # draw = ImageDraw.Draw(img)
                    rew = reward[si, fi]
                    # x, y = 2, 2
                    # draw.text((x, y), str(rew), fill=(255, 255, 255), font=font)
                    # img = np.array(img)
                    if 0.095 < rew < 0.11:
                        water.append(frame)
                    if len(water) > size:
                        break
                print("water frames collected:", len(water))
                if len(water) > size:
                    break
            if len(water) > size:
                break

        # CONVERT TO FINAL ARRAYS
        water = np.array(water, dtype=np.uint8)

        # SAVE AS ZIPPED FILE
        os.makedirs(waterpath, exist_ok=True)
        with gzip.GzipFile(waterpath + watername, 'wb') as fp:
            pickle.dump(water, fp)

        return water

    def collect_data(self):
        args = self.args
        datadir = self.data_path
        envname = args.envname
        mode = args.datamode
        filepath = datadir + f"{envname}-{mode}-{args.datasize}-[{args.gammas}].pickle"
        print("collecting dataset at", filepath)
        if os.path.exists(filepath):
            print("loading dataset...")
            with gzip.open(datadir + f"{envname}-{mode}-{args.datasize}-[{args.gammas}].pickle", 'rb') as fp:
                X, Y, I = pickle.load(fp)
            print("finished loading dataset")
            return X, Y, I

        os.makedirs(datadir, exist_ok=True)
        if not os.path.exists(f"{os.getenv('MINERL_DATA_ROOT', 'data/')}/MineRL{envname}VectorObf-v0"):
            minerl.data.download(os.getenv('MINERL_DATA_ROOT', 'data/'), experiment=f'MineRL{envname}VectorObf-v0')
        data = minerl.data.make(f'MineRL{envname}VectorObf-v0', data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
                                num_workers=args.workers[0], worker_batch_size=args.workers[1])
        names = data.get_trajectory_names()

        size = args.datasize + args.testsize
        # np.random.default_rng().shuffle(names)
        X = np.zeros((size, 64, 64, 3), dtype=np.uint8)
        Y = np.zeros((7, size), dtype=np.float)
        I = np.zeros(size, dtype=np.uint16)
        print("collecting straight data set with", args.datasize, "+", args.testsize, "frames")

        # DEV
        full_ep_lens = 0

        runidx = 0
        for name_idx, name in enumerate(names):
            # print(name)
            print("percentage of episodes used so far:", round(name_idx / len(names) * 100),
                  "dataset size:", runidx,
                  "full ep lens:", full_ep_lens)
            # EXTRACT EPISODE
            state, action, reward, _, done = zip(*data.load_data(name))
            reward = np.array(reward)
            pov = np.stack([s['pov'] for s in state])

            # get full ep len of all
            full_ep_lens += len(pov)

            if mode == "begin":
                # Only take frames until first reward:
                add = np.argmax(reward > 0) + 1 if reward.any() else add
                if add > 1000:
                    continue
                # print("first reward frame idx", add)
                reward = reward[:add]
            elif mode == "trunk":
                mask = [True] + [np.sum(reward[max(0, i - 35):i]) == 0 for i in range(1, len(reward))]
                pov = pov[mask]
                reward = reward[mask]

            add = min(size - runidx, len(pov))
            reward = reward[:add]
            reward = (reward > 0).astype(np.float)
            X[runidx:runidx + add] = pov[:add]
            Y[0, runidx:runidx + add] = reward
            I[runidx:runidx + add] = range(len(pov))[:add]

            for rewidx, gamma in \
                    enumerate(args.gammas.split('-')):
                # FORMATING RAW REWARD
                gamma = float(gamma)
                local_reward = reward.copy()
                for i in range(2, add + 1):
                    last = gamma * local_reward[-i + 1]
                    current = local_reward[-i]
                    local_reward[-i] = min(current + last, 1)

                Y[rewidx + 1, runidx:runidx + add] = local_reward

            runidx += add
            if runidx >= size:
                break

        # SAVE AS ZIPPED FILE
        with gzip.GzipFile(filepath, 'wb') as fp:
            pickle.dump((X[:runidx], Y[:, :runidx], I[:runidx]), fp)

        # DEV
        print("full ep length:", full_ep_lens, "beginning percentage", size / full_ep_lens)

        return X, Y, I

    def dev(self):
        args = self.args
        envname = args.envname
        datadir = "train/data/straight/"
        filepath = datadir + f"{envname}-{args.datasize}.pickle"
        if os.path.exists(filepath):
            print("loading dataset...")
            with gzip.open(datadir + f"{args.datasize}.pickle", 'rb') as fp:
                X, Y, I = pickle.load(fp)
            print("finished loading dataset")
            return X, Y, I

        os.makedirs(datadir, exist_ok=True)
        if not os.path.exists(f"{os.getenv('MINERL_DATA_ROOT', 'data/')}/MineRL{envname}VectorObf-v0"):
            minerl.data.download(os.getenv('MINERL_DATA_ROOT', 'data/'), experiment=f'MineRL{envname}VectorObf-v0')
        data = minerl.data.make(f'MineRL{envname}VectorObf-v0', data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
                                num_workers=args.workers[0], worker_batch_size=args.workers[1])
        names = data.get_trajectory_names()

        size = args.datasize
        # np.random.default_rng().shuffle(names)
        # X = np.zeros((size, 64, 64, 3), dtype=np.uint8)
        Y = np.zeros((7, size), dtype=np.float)
        # I = np.zeros(size, dtype=np.uint16)
        # Y = []
        print("collecting dev data set with")

        # DEV
        full_ep_lens = 0

        runidx = 0
        for name_idx, name in enumerate(names):
            # print(name)
            print("percentage of episodes used so far:", round(name_idx / len(names) * 100), "dataset size", runidx)
            # EXTRACT EPISODE
            state, action, reward, _, done = zip(*data.load_data(name))

            # CONVERT COLOR
            # pov = np.stack([s['pov'] for s in state])
            add = min(size - runidx, len(reward))
            reward = np.array(reward[:add])

            # DEV
            # Only take frames until first reward:
            add = np.argmax(reward > 0) + 1 if reward.any() else add
            # print("first reward frame idx", add)
            reward = reward[:add]
            reward = (reward > 0).astype(np.float)
            # get full ep len of all
            # full_ep_lens += len(pov)

            # X[runidx:runidx + add] = pov[:add]
            # Y[0, runidx:runidx + add] = reward
            # I[runidx:runidx + add] = range(len(pov))[:add]

            for rewidx, (gamma, nonrew) in \
                    enumerate([(0.99, 0), (0.98, 0), (0.97, 0), (0.99, -1), (0.95, -1), (0.90, -1)]):
                # FORMATING RAW REWARD
                if nonrew:
                    local_reward = ((reward <= 0) * nonrew).astype(np.float)
                    for i in range(2, add + 1):
                        last = gamma * local_reward[-i + 1]
                        current = local_reward[-i]
                        local_reward[-i] = 0 if current == 0 else current + last
                else:
                    local_reward = reward.copy()
                    for i in range(2, add + 1):
                        last = gamma * local_reward[-i + 1]
                        current = local_reward[-i]
                        local_reward[-i] = current + last

                Y[rewidx + 1, runidx:runidx + add] = local_reward

            runidx += add
            if runidx >= size:
                break

        # SAVE AS ZIPPED FILE
        with gzip.GzipFile(filepath, 'wb') as fp:
            pickle.dump(Y, fp)

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        axs[0, 0].hist(Y[0])
        axs[0, 0].set_title("raw reward")
        axs[0, 1].hist(Y[1])
        axs[0, 1].set_title("discount 0.99")
        axs[1, 0].hist(Y[2])
        axs[1, 0].set_title("discount 0.98")
        axs[1, 1].hist(Y[3])
        axs[1, 1].set_title("discount 0.97")
        plt.show()

        return Y

    def clean_data(self, vis=0):
        args = self.args
        visdir = f"train/data/straight/{args.datasize}-vis/"
        os.makedirs(visdir, exist_ok=True)
        datadir = f"train/data/straight/"
        os.makedirs(datadir, exist_ok=True)
        filepath = datadir + f"{args.datasize}-clean.pickle"
        chunklen = 100
        distance_between_consecutive_rewards = chunklen
        distance_to_reward = 80
        shift = 20

        if os.path.exists(filepath):
            print("loading clean dataset...")
            with gzip.open(datadir + f"{args.datasize}-clean.pickle", 'rb') as fp:
                X, Y, I = pickle.load(fp)
            print("finished loading clean dataset")
            return X, Y, I

        # VIZ CHOPS METHOD
        def save_frame(name, frame):
            path = f"{visdir}" + name + f".png"
            dirpath = os.path.sep.join(path.split(os.path.sep)[:-1])
            # print(path, dirpath)
            os.makedirs(dirpath, exist_ok=True)
            plt.imsave(path, frame)

        X, Y, I = self.collect_data()
        Y = Y[0]

        # EXTRACT CHOPS
        chops = np.nonzero(Y)[0]
        choptimes = I[chops]
        # print("raw chops:", chops)
        deltas = choptimes[1:] - choptimes[:-1]
        big_enough_deltas = deltas > distance_between_consecutive_rewards
        negative_deltas = deltas < 0
        accepted_chop_times = big_enough_deltas | negative_deltas
        clean_chops = np.concatenate((chops[None, 0], chops[1:][accepted_chop_times]))
        # print("clean chops:", clean_chops)

        # EXCTRACT FAR FROM CHOPS
        faridxs = [i for i in range(len(X)) if not
        ((Y[max(i - distance_to_reward, 0):i + distance_to_reward] &
          ((I[max(i - distance_to_reward, 0):i + distance_to_reward] - I[i]) > 0)).any() or Y[i])]
        # print(faridxs)
        # print("faridxs:", set(faridxs).intersection(set(chops)))

        shift_chops = clean_chops[I[clean_chops] >= shift] - shift
        chunk_chops = shift_chops[I[shift_chops] >= chunklen]
        clean_idxs = np.concatenate([1 + np.arange(i - 100, i) for i in chunk_chops])
        for i in range(5):
            Y[chunk_chops - i] = 1
        # print(clean_idxs)
        # print(Y[clean_idxs+20])

        print("ratio of raw chops to all frames:", len(chops) / len(X))
        print("ratio of cleaned chops to raw chops:", len(clean_chops) / len(chops))
        print("ratio of cleaned chops to all frames:", len(clean_chops) / len(X))
        print("final size of clean chunked dataset", len(clean_idxs), "out of", len(X))

        # SAVE CLEAN DATA
        X, Y, I = X[clean_idxs], Y[clean_idxs], I[clean_idxs]
        with gzip.open(filepath, 'wb') as fp:
            pickle.dump((X, Y, I), fp)

        if vis:
            n_samples = 1000

            # clean data chunks
            print(f"saving first {n_samples // chunklen} cleaned chunks")
            for fix, chix in enumerate(clean_idxs[:n_samples]):
                save_frame(f"chunks/{fix // chunklen}/{fix % chunklen}", X[chix])

            if vis > 1:
                # first chops
                print("saving first chops")
                for fix, chix in enumerate(clean_chops[:n_samples]):
                    save_frame(f"first/first-{fix}", X[chix])

                # consec chops
                print("saving consecutive chops")
                for fix, chix in enumerate(list(set(chops).difference(set(clean_chops)))[:n_samples]):
                    save_frame(f"consec/consec-{fix}", X[chix])

                print("saving shifted first chops")
                # shifted chops
                for shift in [5, 10, 15, 20]:
                    shift_chops = clean_chops[I[clean_chops] >= shift] - shift
                    for fix, chix in enumerate(shift_chops[:n_samples]):
                        save_frame(f"shift/shift-{shift}-{fix}", X[chix])

                print("saving far from reward chops")
                # far from chops
                for fix, chix in enumerate(faridxs[:n_samples]):
                    save_frame(f"far/far-{fix}", X[chix])

        return X, Y, I


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-viz", action="store_true")
    # parser.add_argument("-final", action="store_true")
    # parser.add_argument("-test", action="store_true")
    # parser.add_argument("-unet", action="store_true")
    # parser.add_argument("-resnet", action="store_true")
    # parser.add_argument("-distnorm", action="store_true")
    # parser.add_argument("-sumsegm", action="store_true")
    # parser.add_argument("-dream", action="store_true")
    # parser.add_argument("-discounted", type=bool, default=True)
    # parser.add_argument("-sigmoid", action="store_true")
    # parser.add_argument("-clustersave", action="store_true")
    # parser.add_argument("-savekmeans", action="store_true")
    # parser.add_argument("-integrated", action="store_true")
    # parser.add_argument("-grounded", action="store_true")
    # parser.add_argument("-dummy", action="store_true")
    # parser.add_argument("-grid", action="store_true")
    # parser.add_argument("-hue", action="store_true")
    # parser.add_argument("-hsv", action="store_true")
    # parser.add_argument("-contrastive", action="store_true")
    # parser.add_argument("-patchembed", action="store_true")
    # parser.add_argument("-visembed", action="store_true")
    # parser.add_argument("-small", action="store_true")
    # parser.add_argument("-water", action="store_true")
    # parser.add_argument("-pool", type=bool, default=True)
    # parser.add_argument("-navneg", action="store_true")
    # parser.add_argument("-umap", action="store_true")
    # parser.add_argument("-visunet", action="store_true")
    # parser.add_argument("-ucritic", action="store_true")
    # parser.add_argument("-transpose", action="store_true")
    # parser.add_argument("-pure", type=bool, default=True)
    # parser.add_argument("-copy", action="store_true")
    # parser.add_argument("-vgg", action="store_true")
    # parser.add_argument("-trans", action="store_true")
    # parser.add_argument("-withnav", action="store_true")
    # parser.add_argument("-proof", action="store_true")
    parser.add_argument("-train", action="store_true")
    # parser.add_argument("-live", action="store_true")
    # parser.add_argument("-live", type=bool, default=True)
    parser.add_argument("-cleaned", action="store_true")
    parser.add_argument("-frozen", action="store_true")
    # parser.add_argument("-critic", action="store_true")
    parser.add_argument("-masker", action="store_true")
    parser.add_argument("-critic", type=bool, default=True)
    # parser.add_argument("-masker", type=bool, default=True)
    parser.add_argument("-mload", action="store_true")
    parser.add_argument("-cload", action="store_true")
    parser.add_argument("-staticnorm", type=bool, default=True)
    parser.add_argument("-clippify", action="store_true")
    parser.add_argument("-debug", action="store_true")
    # parser.add_argument("-inject", action="store_true")
    # parser.add_argument("-inject", type=bool, default=True)
    parser.add_argument("-noinject", action="store_true")
    parser.add_argument("-freeze", action="store_true")
    parser.add_argument("-viscritic", action="store_true")
    parser.add_argument("-vismasker", action="store_true")
    # parser.add_argument("-viscritic", type=bool, default=True)
    # parser.add_argument("-vismasker", type=bool, default=True)
    # parser.add_argument("-eval", type=bool, default=True)
    parser.add_argument("-visdataset", action="store_true")
    parser.add_argument("-visbesteval", action="store_true")
    parser.add_argument("-trunk", action="store_true")
    parser.add_argument("-higheval", action="store_true")
    parser.add_argument("-separate", action="store_true")

    parser.add_argument("--eval", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--threshrew", type=float, default=0)
    parser.add_argument("--trainasvis", type=int, default=0)
    parser.add_argument("--false", type=bool, default=False)
    parser.add_argument("--envname", type=str, default="Treechop")
    parser.add_argument("--visname", type=str, default="curves")
    parser.add_argument("--datamode", type=str, default="trunk")
    parser.add_argument("--purevis", type=str, default="")
    parser.add_argument("--sortidx", type=int, default=1)
    parser.add_argument("--chfak", type=int, default=1)
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--lfak", type=int, default=5)
    parser.add_argument("--neck", type=int, default=32)
    parser.add_argument("--clossfak", type=int, default=5)
    parser.add_argument("--cepochs", type=int, default=15)
    parser.add_argument("--mepochs", type=int, default=2)
    parser.add_argument("--high-rew-thresh", type=float, default=0.7)
    parser.add_argument("--low-rew-thresh", type=float, default=0.3)
    parser.add_argument("--L2", type=float, default=0.0)
    parser.add_argument("--L1", type=float, default=1)
    parser.add_argument("--saveevery", type=int, default=5)
    parser.add_argument("--visevery", type=int, default=100)
    parser.add_argument("--rewidx", type=int, default=1)
    parser.add_argument("--gammas", type=str, default="0.98-0.97-0.96-0.95")
    parser.add_argument("--testsize", type=int, default=5000)
    parser.add_argument("--datasize", type=int, default=100000)
    # parser.add_argument("--navdatasize", type=int, default=10000)
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--runs", type=int, default=1)
    # parser.add_argument('--resize', type=int, default=64)
    # parser.add_argument("--fskip", type=int, default=5)
    # parser.add_argument("--etha", type=float, default=0.95)
    # parser.add_argument("--window", type=str, default="64-64")
    # parser.add_argument("--blur", type=int, default=0)
    # parser.add_argument("--cluster", type=str, default="")
    # parser.add_argument("--clustercritic", type=int, default=0)
    # parser.add_argument("--trajsize", type=int, default=50)
    # parser.add_argument("--gray", type=bool, default=True)
    # parser.add_argument("--dreamsteps", type=int, default=0)
    # parser.add_argument("--threshold", type=float, default=0.9)
    # parser.add_argument("--rewidx", type=int, default=3)
    # parser.add_argument("--wait", type=int, default=120)
    # parser.add_argument("--delay", type=int, default=0)
    # parser.add_argument("--warmup", type=int, default=20)
    # parser.add_argument("--gamma", type=float, default=0.95)
    # parser.add_argument("--revgamma", type=float, default=1.1)
    # parser.add_argument("--delta", type=int, default=50)
    # parser.add_argument("--chunksize", type=int, default=10)
    # parser.add_argument("--cons", type=int, default=250)
    # parser.add_argument("--embed-dim", type=int, default=100)
    # parser.add_argument("--embed-cluster", type=int, default=100)
    # parser.add_argument("--embed-train-samples", type=int, default=700)
    # parser.add_argument("--embed-patch-width", type=int, default=10)
    # parser.add_argument("--embed-pos-threshold", type=float, default=0.9)
    # parser.add_argument("--embed-norm", type=str, default="raw")
    # parser.add_argument("--color", type=str, default="HSV")
    # parser.add_argument("--external-critic", type=str, default="")

    args = parser.parse_args()
    args.workers = (1, 1, 1)
    args.live = not args.frozen
    args.inject = not args.noinject
    print(args)

    # H.collect_data()
    resfile = "results.txt"
    resfile = open(resfile, 'a')

    res = []
    for run_idx in range(args.runs):
        print("RUN", run_idx)
        H = Handler(args)
        H.load_data()
        if args.trainasvis:
            H.visualize()
            exit()
        if args.cload:
            H.load_models(modelnames=[H.criticname])
        if args.mload:
            H.load_models(modelnames=[H.maskername])
        if args.critic:
            H.critic_pipe_old(mode="train")
            H.save_models(modelnames=[H.criticname])
        if args.masker:
            # H.load_models(modelnames=[H.criticname])
            H.segmentation_training()
            H.save_models(modelnames=[H.maskername])
        if args.viscritic or args.vismasker:
            H.visualize()

        res.append((H.ious, H.bestepoch))
        # H.reset_models()

    ious = np.array([e[0][0] for e in res]).round(3)
    hious = np.array([e[0][1] for e in res]).round(3)
    avg_iou = np.mean(ious)
    avg_hiou = np.mean(hious)
    std_ious = np.std(ious)
    std_hious = np.std(hious)
    best_iou = max(ious)
    best_hiou = max(hious)
    resfile.write(f"{args.name} avg={avg_iou} std={std_ious} best={best_iou} havg={avg_hiou} hstd={std_hious} hbest={best_hiou} {res}\n")
    """
    if args.water:
        H.train_water_discriminator()
    if args.cluster:
        if args.train or args.savekmeans:
            H.cluster(mode="train")
        if args.test:
            H.cluster(mode="test")

    if args.trans:
        H.trans_embeds()
    if args.train:
        if args.critic:
            H.critic_pipe_old(mode="train")
            H.save_models(modelnames=[H.criticname])
        if args.unet:
            H.load_models(modelnames=[H.criticname])
            H.segment(mode="train")
            H.save_models(modelnames=[H.unetname])
    if args.test:
        if not args.train:
            H.load_models()
        if args.critic:
            H.critic_pipe(mode="test")
        if args.unet:
            H.segment(mode="test")
    if args.dreamsteps:
        if not args.train:
            H.load_models(modelnames=[H.criticname])
        H.dream()
    if args.sumsegm:
        if args.train:
            H.sum_segm(mode="train")
        if args.test:
            H.sum_segm(mode="test")
    if args.contrastive:
        if args.critic:
            H.contrastive_critic_pipe()
        if args.viscritic:
            H.visualize()
        if args.segm:
            H.contrastive_merge_segmentation()
        if args.visunet:
            H.vis_unet(online=True)
            H.vis_unet()
        if args.umap:
            H.vis_unet_embeddings()
    if args.patchembed:
        # H.vis_pixels()
        H.create_patch_embedding_clusters()
    if args.visembed and args.viz:
        H.vis_embed()
        # H.vis_white_tree()
    """