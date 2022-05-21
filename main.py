import argparse
import PIL
from matplotlib import pyplot as plt
import os
import sys
from numpy.core.defchararray import join
import ctypes

import pydensecrf
libgcc_s = ctypes.CDLL('libgcc_s.so.1') # libgcc_s.so.1 error workaround
from nets import *
import numpy as np
import torch as T
import torch.nn.functional as F
import pickle
import minerl
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2
import logging as L
import gzip
import math
from PIL import Image, ImageDraw, ImageFont
import ffmpeg
from pydensecrf import densecrf as denseCRF


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
        self.path = f"{args.name}/"
        self.train_path = self.path + "train/"
        self.result_path = self.path + "results/"
        self.save_path = self.path + "saves/"
        self.data_path = "runs/data/straight/"
        self.save_paths = {
            self.criticname: f"{self.save_path}critic-{self.critic_args}.pt",
            self.maskername: f"{self.save_path}masker-{self.masker_args}.pt"
        }

        # L.basicConfig(filename=f'./logs/{args.name}.log', format='%(asctime)s %(levelname)s %(name)s %(message)s', level=L.INFO)

    def reset_models(self):
        args = self.args
        self.critic = NewCritic(bottleneck=args.neck, chfak=args.chfak, dropout=args.dropout)
        self.masker = UnetDecoder(bottleneck=args.neck, chfak=args.chfak)
        if self.args.separate:
            self.sepcrit = NewCritic(bottleneck=args.neck, chfak=args.chfak, dropout=args.dropout)

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
                if not self.args.train:
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

    def critic_pipe(self, mode="train", test=0):
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
        if args.directeval:
            ious = self.eval()
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

        self.log(f"\nallframes {len(preds)}  frames>{args.high_rew_thresh}", positives.sum(),
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

        self.log("n positives", self.Xpos.shape[0])
        self.log("positives:", self.Ypos)
        self.log("HIGH REW THRESH", args.high_rew_thresh)
        assert (preds[positives].mean()) > args.high_rew_thresh
        # assert np.mean(self.Ypos[args.rewidx]) > args.high_rew_thresh
        self.log("n negatives", self.Xneg.shape[0])

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


        if args.directeval:
            self.eval()
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

            if False:
                ious = self.eval()
                if ious[0] > self.ious[0]:
                    self.ious = ious
                    self.bestepoch = epoch

        print()
        self.save_models(modelnames=[self.maskername])

    def saliency(self, X):
        critic = self.critic
        X = T.tensor(X, requires_grad=True).to(self.device)
        logits = critic(X)
        logits.backward()
        print(X.grad)

    def shift_batch(self, X):
        xshift = int(self.args.shift * T.rand(1))
        if T.rand(1) > 0.5:
            # X = T.cat((X[:, :, yshift:], X[:, :, :yshift]), dim=2)
            X = T.cat((X[:, :, xshift:], X[:, :, :xshift]), dim=2)
        else:
            X = T.cat((X[:, :, -xshift:], X[:, :, :-xshift]), dim=2)
        return X

    def batch_to_vid(self, batches, Y=None):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        ywid = batches[0].shape[1]
        xwid = batches[0].shape[2]
        out = cv2.VideoWriter(resultdir + result_args + '.mp4', fourcc, 20.0, (xwid * len(batches), ywid))

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
        video_path = resultdir + result_args + '.mp4'
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

    def log(self, *args, key="", **kargs):
        if key in ["info"]:
            print(*args, **kargs)

    def eval(self, folder="", vis=False):
        self.log("STARTING EVAL")
        args = self.args
        resultdir = "eval-results/"+args.name+"/"
        scale = 4
        ph = 32
        pad = 0
        os.makedirs(resultdir, exist_ok=True)

        if not args.noevalmode:
            critic = self.critic.to(self.device).eval()
            masker = self.masker.to(self.device).eval()
            if args.separate:
                sepcrit = self.sepcrit.to(self.device).eval()
        else:
            critic = self.critic.to(self.device).train()
            masker = self.masker.to(self.device).train()
            if args.separate:
                sepcrit = self.sepcrit.to(self.device).train()
        batchsize = 128
        imgs = []
        preds = []
        M = []
        salM = []
        allM = []

        if folder:
            X = np.concatenate([np.array(Image.open(img_file_name)) for img_file_name in os.listdir(folder)])/255.0
        else:
            evaldatapath = "red-trees/"
            X = np.load(evaldatapath + "X.npy")/255.0
            if args.resimages:
                X = np.load("resimgs.npy")
        self.log("X", X.shape, np.min(X), np.max(X))
        Y = np.expand_dims(np.all(np.load(evaldatapath + "Y.npy"), axis=-1), axis=-1) if not args.resimages else np.zeros((len(X),1,64,64))

        X = X[100:5000:2]
        Y = Y[100:5000:2]

        allM.append(Y.transpose(0,3,1,2))
        self.log("Yshape", Y.shape)
        # print(Y)
        # print("Y min max", Y.shape, np.min(Y), np.max(Y))

        for bidx in range(0, len(X), batchsize):
            print("eval at", bidx / len(X), end="\r")
            # print(Y.shape)
            imgs.append(X[bidx:bidx + batchsize])
            batch = T.from_numpy(X[bidx:bidx + batchsize]).permute(0, 3, 1, 2).float().to(
                self.device)
            if args.salience:
                batch = T.tensor(batch, requires_grad=True)
            # print(batch)
            pred, embeds = critic(batch, collect=True)
            preds.append(pred.detach().squeeze().cpu().numpy())
            if args.separate:
                _, embeds = sepcrit(batch, collect=True)
            
            if args.salience:
                pred.mean().backward()
                m = batch.grad.abs().sum(dim=1)[:,None]
                #mask = mask * pred.detach()[:,:,None,None]
                salM.append(m.detach().cpu().numpy())
            
            mask = masker(batch, embeds)
            M.append(mask.detach().cpu().numpy())

        M = np.concatenate(M, axis=0)
        if args.salience:
            salM = np.concatenate(salM, axis=0)
        preds = np.concatenate(preds, axis=0)
        imgs = np.concatenate(imgs, axis=0)

        hardM = M>args.eval_thresh
        #hardM = hardM.transpose(0, 2, 3, 1)
        self.log("Mshape", M.shape, "HardM", hardM.shape)
        allM.extend([M, hardM])

        if args.crf:               
            crfM = self.crf(imgs, M, Y)
            self.log("crfMshape", crfM.shape)
            allM.append(crfM)

        #grid = [float(e) for e in args.grid.split("-")] or args.eval
        #print("GRID", grid)
        if args.salience:
            thresh = args.salience_thresh
            #norm = (M.flatten(2).max(dim=2).values)[:,:,None,None]
            if args.salglobal:
                norm = (salM*(salM>=0)).mean()*thresh
                norm = norm
                print("THRESH", thresh, "mean", (salM*(salM>=0)).mean(), "NORMshape", norm.shape)
            else:
                k = int(salM.shape[-1]*salM.shape[-2]*thresh)
                flat_m = salM.reshape(salM.shape[0],1,-1)
                norm = np.sort(flat_m, axis=-1, order=None)
                print("NORM SHAPE", norm.shape, "K", k)
                norm = norm[:,:,k,None,None]
                
            #print("thresh", thresh, "norm value", norm.shape, norm)
            #print(M.shape, norm.shape)
            salM = salM/(norm+sys.float_info.min)
            salM = salM*preds[:,None,None,None]
            salM[(salM >= 1)] = 1
            salhardM = (salM>thresh).astype(np.uint8)
            
            self.log("salMshape", salM.shape, salhardM.shape)
            allM.extend([salM, salhardM])

            if args.crf:
                salcrfM = self.crf(imgs, salM, Y)
                self.log("salcrfMshape", salcrfM.shape)
                allM.append(salcrfM)                
        
        iou = self.get_iou(hardM.squeeze(), Y.squeeze())
        ious = [iou]
        if args.crf:
            crfiou = self.get_iou(crfM.squeeze(), Y.squeeze())
            ious.append(crfiou)
        if args.salience:
            saliou = self.get_iou(salhardM.squeeze(), Y.squeeze())
            ious.append(saliou)
            if args.crf:
                salcrfiou = self.get_iou(salcrfM.squeeze(), Y.squeeze())
                ious.append(salcrfiou)

        self.log(f"\nRESULTS", ious)
        # self.ious = (iou, highiou)
        #critic = self.critic.to(self.device).train()
        #masker = self.masker.to(self.device).train()

        if args.resimages:
            for img_idx, img in enumerate(hardM.squeeze()):
                os.makedirs(resultdir+"resimages/")
                plt.imsave(resultdir+"resimages/"+f"{img_idx}.png")
            
        if iou>self.ious[0] and args.visbesteval:
            reordering = [0,1,4,3,2,7,6,5]
            if not args.crf:
                reordering = [0,1,3,2,5]
            elif not args.salience:
                reordering.remove(6)
                reordering.remove(5)
            print(reordering)
            short = len(reordering) != 8
            fosi = 30
            scalef = 3
            font = ImageFont.truetype("./isy_minerl/segm/etc/Ubuntu-R.ttf", fosi)
            masks = [np.concatenate((m,m,m), axis=1).transpose(0,2,3,1) for m in allM]
            frames = [X]+masks
            frames = [(frames[i], print(i))[0] for i in reordering[:len(masks)+1]]
            self.log("REORDERING", reordering, "FRAMES", len(frames))
            frames = np.concatenate(frames, axis=2)
            #flat = lambda x: x.reshape(x.shape[:-2],-1)
            #resquare = lambda x: x.reshape(x.shape[:-1], 64, 64)
            #both = np.concate
            #minim = lambda a,b: np.min(flat(a))
            y = allM[0]
            #colors = [np.concatenate(((y&(1-m))|((1-y)&m), (y&m)+0.5*((1-y)&m), np.zeros_like(m)), axis=1).transpose(0,2,3,1) if i in [0,2,3,5,6] else 0.1*np.ones_like(masks[0], dtype=np.uint8)
            #    for (i,m) in enumerate(allM)]
            colors = [np.concatenate(((y&(1-m))+0.5*((1-y)&m), (y&m)+0.5*((1-y)&m), 0.5*((1-y)&m)), axis=1).transpose(0,2,3,1) if i in [0,2,3,5,6] else 0.1*np.ones_like(masks[0], dtype=np.uint8)
                for (i,m) in enumerate([m.astype(int) for m in allM])]
            #self.log("MASKS", [m.shape for m in masks])
            colorframes = [X]+colors
            colorframes = [colorframes[i] for i in reordering]
            colorframes = np.concatenate(colorframes, axis=2)
            #frames = np.concatenate((X, np.stack((M,M,M), axis=-1), np.stack((Y,Y,Y), axis=-1)), axis=1)
            frames = np.concatenate([frames, colorframes], axis=1)
            frames = (frames * 255).astype(np.uint8)
            frames = F.interpolate(T.from_numpy(frames).float().permute(0,3,1,2), scale_factor=scalef).permute(0,2,3,1).numpy()

            #frames = np.pad(frames, ((0,0), (32,32), (0,0), (0,0)))
            titles = ["RGB\nimage", "ground\ntruth", "mask", "thresholded\nmask\nIoU=0.41", "mask\nCRF\nIoU=0.45", "saliency\nmap", "thresholded\nsaliency\nIoU=0.22", "salience\nCRF\nIoU=0.11"]
            titles = [titles[i] for i in reordering]
            titlesarray = Image.fromarray(np.zeros((fosi*4, frames.shape[2], 3), dtype=np.uint8))
            draw = ImageDraw.Draw(titlesarray)
            for i in range(len(reordering)):
                text = titles[i]
                x, y = fosi//5 + 64*scalef*i, fosi//5
                draw.text((x, y), text, font=font)
            titlesarray = np.tile(titlesarray, (frames.shape[0], 1, 1, 1))

            legend = ["GREEN = True Positive", "RED = False Negative", "GRAY = False Positive", "BLACK = True Negative"]
            legendarray = Image.fromarray(np.zeros((fosi*(4 if short else 2), frames.shape[2], 3), dtype=np.uint8))
            legendcolors = [(0,255,0), (255,0,0), (125,125,125), (255,255,255)]
            legendspacing = int((frames.shape[2]-2)/len(legend))
            draw = ImageDraw.Draw(legendarray)
            for i,l in enumerate(legend):
                conditionallinebreak = "\n" if short and i>1 else ""
                draw.text((fosi//5 + i*legendspacing, fosi//5), l+conditionallinebreak, font=font, fill=legendcolors[i])
            legendarray = np.tile(legendarray, (frames.shape[0], 1, 1, 1))

            frames = np.concatenate((titlesarray, frames, legendarray), axis=1)
            
            if args.output_video:
                args.output_video += "/"
            vidwrite(f"{args.output_video}iou={round(iou, 3)}.mp4", frames, framerate=10)

        """
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
        """
        return ious

    def segment(self, folder):
        print("STARTING SEGMENTATION...")
        args = self.args
        resultdir = self.path
        os.makedirs(resultdir, exist_ok=True)

        if not args.noevalmode:
            critic = self.critic.to(self.device).eval()
            masker = self.masker.to(self.device).eval()
            if args.separate:
                sepcrit = self.sepcrit.to(self.device).eval()
        else:
            critic = self.critic.to(self.device).train()
            masker = self.masker.to(self.device).train()
            if args.separate:
                sepcrit = self.sepcrit.to(self.device).train()
        batchsize = 128
        imgs = []
        preds = []
        M = []
        salM = []
        allM = []

        img_names = os.listdir(folder)
        X = np.stack([np.array(Image.open(f"{folder}/{img_file_name}")) for img_file_name in img_names])/255.0
        img_names = [a[:-1-a[::-1].index(".")] for a in img_names if "." in a]

        for bidx in range(0, len(X), batchsize):
            print("segmentation in progress", round(bidx / len(X), 2), end="%\r")
            # print(Y.shape)
            imgs.append(X[bidx:bidx + batchsize])
            batch = T.from_numpy(X[bidx:bidx + batchsize]).permute(0, 3, 1, 2).float().to(
                self.device)
            if args.salience:
                batch = T.tensor(batch, requires_grad=True)
            # print(batch)
            pred, embeds = critic(batch, collect=True)
            preds.append(pred.detach().squeeze().cpu().numpy())
            if args.separate:
                _, embeds = sepcrit(batch, collect=True)
            
            if args.salience:
                pred.mean().backward()
                m = batch.grad.abs().sum(dim=1)[:,None]
                #mask = mask * pred.detach()[:,:,None,None]
                salM.append(m.detach().cpu().numpy())
            
            mask = masker(batch, embeds)
            M.append(mask.detach().cpu().numpy())
        print()

        print("postprocessing...")
        M = np.concatenate(M, axis=0)
        if args.process_salience:
            salM = np.concatenate(salM, axis=0)
        preds = np.concatenate(preds, axis=0)
        imgs = np.concatenate(imgs, axis=0)
        allM.append(M)
        self.log("Mshape", M.shape)

        if args.binarymaskthreshold:
            hardM = M>=args.binarymaskthreshold
            #hardM = hardM.transpose(0, 2, 3, 1)
            self.log("HardM", hardM.shape)
            allM.append(hardM)

        if args.crf:               
            crfM = self.crf(imgs, M, np.zeros_like(M).astype(np.bool).transpose(0,2,3,1))
            self.log("crfMshape", crfM.shape)
            allM.append(crfM)

        #grid = [float(e) for e in args.grid.split("-")] or args.eval
        #print("GRID", grid)
        if args.process_salience:
            thresh = args.salience_thresh
            #norm = (M.flatten(2).max(dim=2).values)[:,:,None,None]
            if args.salglobal:
                norm = (salM*(salM>=0)).mean()*thresh
                norm = norm
                self.log("THRESH", thresh, "mean", (salM*(salM>=0)).mean(), "NORMshape", norm.shape)
            else:
                k = int(salM.shape[-1]*salM.shape[-2]*thresh)
                flat_m = salM.reshape(salM.shape[0],1,-1)
                norm = np.sort(flat_m, axis=-1, order=None)
                print("NORM SHAPE", norm.shape, "K", k)
                norm = norm[:,:,k,None,None]
                
            #print("thresh", thresh, "norm value", norm.shape, norm)
            #print(M.shape, norm.shape)
            salM = salM/(norm+sys.float_info.min)
            salM = salM*preds[:,None,None,None]
            salM[(salM >= 1)] = 1
            salhardM = (salM>thresh).astype(np.uint8)
            
            self.log("salMshape", salM.shape, salhardM.shape)
            allM.extend([salM, salhardM])

            if args.crf:
                salcrfM = self.crf(imgs, salM, np.zeros_like(salM).astype(np.bool).transpose(0,2,3,1))
                self.log("salcrfMshape", salcrfM.shape)
                allM.append(salcrfM)                

        if args.resimages:
            for img_idx, img in enumerate(hardM.squeeze()):
                os.makedirs(resultdir+"resimages/")
                plt.imsave(resultdir+"resimages/"+f"{img_idx}.png")
            
        outpath = args.mask_output_imgs
        os.makedirs(outpath, exist_ok=True)
        masks = np.stack([X]+[np.concatenate((m,m,m), axis=1).transpose(0,2,3,1) for m in allM], axis=1)
        columns = ["raw-mask", "thresholded-mask", "crf-mask","saliency-map", "thresholded-saliency", "crf-saliency"]
        self.log("MASK SHAPE", masks.shape)
        for fidx in range(masks.shape[0]):
            if args.concatenated:
                array = np.concatenate((masks[fidx]*255).astype(np.uint8), axis=-2)
                img = Image.fromarray(array)
                img.save(f"{outpath}/{img_names[fidx]}_with_mask.png")
            else:
                for midx in range(1, masks.shape[1]):
                    img = Image.fromarray((masks[fidx,midx]*255).astype(np.uint8))
                    img.save(f"{outpath}/{img_names[fidx]}-{columns[midx-1]}.png")


    def crf(self, imgs, mask, Y, skip=1):
        mask = mask.copy()
        resultdir = self.path
        os.makedirs(resultdir+"crf/", exist_ok=True)
        w1    = [22]   # weight of bilateral term
        alpha = [12]    # spatial std
        beta  = [3.1]    # rgb  std
        w2    = [8]    # weight of spatial term
        gamma = [1.8]     # spatial std
        it    = [10]   # iteration
        res = []
        params = []
        for param in [(a,b,c,d,e,i) for a in w1 for b in alpha for c in beta for d in w2 for e in gamma for i in it]:
            M = mask[::skip]
            #param = (w1, alpha, beta, w2, gamma, it)
            for i, img in enumerate(imgs[::skip]):
                maskframe = M[i,0]
                prob = np.stack((1-maskframe, maskframe), axis=-1)
                seg = denseCRF.densecrf((255*img).astype(np.uint8), prob, param)
                if not i%50:
                    plt.imsave(resultdir+f"crf/{i}_mask.png", maskframe)
                    plt.imsave(resultdir+f"crf/{i}_img.png", img)
                    plt.imsave(resultdir+f"crf/{i}_crf.png", seg)
                #print("seg values", np.max(seg))
                M[i,0] = seg
            M = M.transpose(0, 2, 3, 1).astype(np.bool)
            # print("types", M.dtype, Y.dtype, "shapes", M.shape, Y.shape, Y[::skip].shape, Y[::skip].dtype)
            r = np.sum(Y[::skip] & M)/np.sum(Y[::skip] | M)
            res.append(r)
            params.append(param)

        res = np.array(res)
        order = np.argsort(res)
        res = res[order]
        params = np.array(params)[order]
        # print("results:", list(zip(params[-10:], res[-10:])))
        mask[::skip] = M.transpose(0,3,1,2)
        return (mask >= 1)

    def get_iou(self, A, B):
        self.log("IOU shapes", A.shape, B.shape)
        intersection = np.sum(A & B)
        union = np.sum(A | B)
        iou = intersection / union
        return round(iou, 3)

    def collect_data(self):
        args = self.args
        datadir = self.data_path
        envname = args.envname
        mode = args.datamode
        filepath = datadir + f"{envname}-{mode}-{args.datasize}-[{args.gammas}].pickle"
        print("collecting dataset at", filepath)
        if os.path.exists(filepath):
            print("loading existing dataset...")
            with gzip.open(datadir + f"{envname}-{mode}-{args.datasize}-[{args.gammas}].pickle", 'rb') as fp:
                X, Y, I = pickle.load(fp)
            print("finished loading exisiting dataset")
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
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-cleaned", action="store_true")
    parser.add_argument("-frozen", action="store_true")
    parser.add_argument("-masker", type=bool, default=True)
    parser.add_argument("-critic", type=bool, default=True)
    parser.add_argument("-cload", type=bool, default=True)
    parser.add_argument("-mload", type=bool, default=True)
    parser.add_argument("-staticnorm", type=bool, default=True)
    parser.add_argument("-clippify", action="store_true")
    parser.add_argument("-debug", action="store_true")
    parser.add_argument("-noinject", action="store_true")
    parser.add_argument("-freeze", action="store_true")
    parser.add_argument("-viscritic", action="store_true")
    parser.add_argument("-vismasker", action="store_true")
    parser.add_argument("-visdataset", action="store_true")
    parser.add_argument("-visbesteval", type=bool, default=True)
    parser.add_argument("-trunk", action="store_true")
    parser.add_argument("-higheval", action="store_true")
    parser.add_argument("-separate", action="store_true")
    parser.add_argument("-salience", action="store_true")
    parser.add_argument("-process_salience", action="store_true")
    parser.add_argument("-salglobal", type=bool, default=True)
    parser.add_argument("-grabcut", action="store_true")
    parser.add_argument("-crf", action="store_true")
    parser.add_argument("-directeval", action="store_true")
    parser.add_argument("-soft", action="store_true")
    parser.add_argument("-resimages", action="store_true")
    parser.add_argument("-noevalmode", action="store_true")
    parser.add_argument("-eval", action="store_true")
    parser.add_argument("-process", action="store_true")
    parser.add_argument("-test", action="store_true")
    parser.add_argument("-concatenated", action="store_true")
    parser.add_argument("-softmask", action="store_true")


    parser.add_argument("--salience-thresh", type=float, default="1.5")
    parser.add_argument("--eval-thresh", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--threshrew", type=float, default=0)
    parser.add_argument("--trainasvis", type=int, default=0)
    parser.add_argument("--false", type=bool, default=False)
    parser.add_argument("--envname", type=str, default="Treechop")
    parser.add_argument("--visname", type=str, default="curves")
    parser.add_argument("--datamode", type=str, default="trunk")
    parser.add_argument("--purevis", type=str, default="")
    parser.add_argument("--sortidx", type=int, default=1)
    parser.add_argument("--chfak", type=int, default=1)
    parser.add_argument("--shift", type=int, default=12)
    parser.add_argument("--lfak", type=int, default=5)
    parser.add_argument("--neck", type=int, default=32)
    parser.add_argument("--clossfak", type=int, default=5)
    parser.add_argument("--cepochs", type=int, default=15)
    parser.add_argument("--mepochs", type=int, default=1)
    parser.add_argument("--high-rew-thresh", type=float, default=0.7)
    parser.add_argument("--low-rew-thresh", type=float, default=0.3)
    parser.add_argument("--L2", type=float, default=0.0)
    parser.add_argument("--L1", type=float, default=0.5)
    parser.add_argument("--saveevery", type=int, default=5)
    parser.add_argument("--visevery", type=int, default=100)
    parser.add_argument("--rewidx", type=int, default=1)
    parser.add_argument("--gammas", type=str, default="0.98-0.97-0.96-0.95")
    parser.add_argument("--testsize", type=int, default=5000)
    parser.add_argument("--datasize", type=int, default=100000)
    parser.add_argument("--name", type=str, default="default-model")
    parser.add_argument("--model", type=str, default="default-model")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--source-imgs", type=str, default="")
    parser.add_argument("--mask-output-imgs", type=str, default="results")
    parser.add_argument("--output-video", type=str, default="")
    parser.add_argument("--binarymaskthreshold", type=float, default=0.5)

    args = parser.parse_args()
    args.workers = (1, 1, 1)
    args.live = not args.frozen
    args.inject = not args.noinject
    args.name = args.model
    if args.test:
        args.eval = True
        args.train = True if not args.cload else False
        args.visbesteval = True
        args.crf = False
        args.salience = True
    
    #print(args)
    H = Handler(args)
    if args.train:
        H.load_data()
    if args.trainasvis:
        H.visualize()
        exit()
    if args.cload:
        H.load_models(modelnames=[H.criticname])
    if args.mload:
        H.load_models(modelnames=[H.maskername])
    if args.train:
        if args.critic:
            H.critic_pipe(mode="train")
            H.save_models(modelnames=[H.criticname])
        if args.masker:
            H.segmentation_training()
            H.save_models(modelnames=[H.maskername])
    if args.eval:
        H.eval()
    if args.viscritic or args.vismasker:
        H.visualize()
    if args.process:
        H.segment(folder=args.source_imgs)

if __name__=="__main__":
    main()