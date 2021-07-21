import torch as T
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import torchvision.models as visionmodels
from torchvision import transforms
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import numpy as np


class Printer(nn.Module):
    def __init__(self):
        super(Printer, self).__init__()

    def forward(self, X):
        print(X.shape)
        return X


class AutoEncoder(nn.Module):
    def __init__(self, width, enc_dim, colorchs, activation=nn.Tanh):
        super().__init__()
        self.width = width
        self.enc = nn.Sequential(
            nn.Linear(colorchs * width * width, 32),
            activation(),
            nn.Linear(32, 16),
            activation(),
            nn.Linear(16, enc_dim),
            activation()
        )
        self.dec = nn.Sequential(
            nn.Linear(enc_dim, 16),
            activation(),
            nn.Linear(16, 32),
            activation(),
            nn.Linear(32, colorchs * width * width)
        )
        self.opti = T.optim.Adam(chain(self.enc.parameters(), self.dec.parameters()), 1e-3)
        self.sched = T.optim.lr_scheduler.StepLR(self.opti, 1, 0.5)

    def forward(self, x: T.Tensor):
        shape = x.shape
        x = x.flatten(start_dim=1)
        enc = self.enc(x)
        x = self.dec(enc)
        x = x.view(shape)
        return x, enc

    def convert_enc(self, enc):
        return (enc + 1) / 2

    def train_batch(self, batch):
        res, enc = self.forward(batch)  # Forward pass

        loss = F.binary_cross_entropy_with_logits(res, batch)  # Loss
        self.opti.zero_grad()
        loss.backward()
        self.opti.step()

        return loss.item(), T.sigmoid(res.detach()), enc.detach()

    def test_batch(self, batch):
        with T.no_grad():
            res, enc = self.forward(batch)  # Forward pass()

        return T.sigmoid(res.detach()), enc.detach()


class VAE(nn.Module):
    def __init__(self, width, enc_dim, colorchs, activation=nn.ReLU):
        super().__init__()
        self.width = width
        self.enc = nn.Sequential(
            nn.Linear(colorchs * width * width, 32),
            activation(),
            nn.Linear(32, 16),
            activation(),
            nn.Lineanetsr(16, enc_dim * 2)
        )
        self.dec = nn.Sequential(
            nn.Linear(enc_dim, 16),
            activation(),
            nn.Linear(16, 32),
            activation(),
            nn.Linear(32, colorchs * width * width)
        )
        self.opti = T.optim.Adam(chain(self.enc.parameters(), self.dec.parameters()), lr=1e-3)
        self.sched = T.optim.lr_scheduler.StepLR(self.opti, 1, 0.5)

    def forward(self, x: T.Tensor):
        shape = x.shape
        x = x.flatten(start_dim=1)
        enc = self.enc(x)

        mean = enc[:, :enc.shape[-1] // 2]
        log_std = enc[:, enc.shape[-1] // 2:]
        # std = self.convert_enc(std)
        dist = T.distributions.Normal(mean, log_std.exp())
        sample = dist.rsample()

        x = self.dec(sample)
        x = x.view(shape)
        return x, mean, log_std, dist

    def convert_enc(self, enc):
        return (enc + 1) / 2

    def train_batch(self, batch):
        res, mean, log_std, dist = self.forward(batch)  # Forward pass

        recon_loss = F.binary_cross_entropy_with_logits(res, batch, reduction="sum")  # Loss
        # regul_loss = T.distributions.kl_divergence(dist, T.distributions.Normal(T.zeros_like(mean), T.ones_like(mean)))
        kl_loss = -0.5 * T.sum(1 + log_std - mean.pow(2) - log_std.exp())

        # print( recon_loss, regul_loss)
        # print(kl_loss)
        loss = recon_loss + 1 * kl_loss
        # print(mean.mean(dim=0), log_std.mean(dim=0))

        self.opti.zero_grad()
        loss.backward()
        self.opti.step()

        return loss.item(), T.sigmoid(res.detach()), mean.detach()

    def test_batch(self, batch):
        with T.no_grad():
            res, mean, _, _ = self.forward(batch)  # Forward pass()

        return T.sigmoid(res.detach()), mean.detach()


class Critic(nn.Module):
    def __init__(self, width=64, enc_dim=1, colorchs=3, chfak=1, activation=nn.ReLU, end=[], pool="max"):
        super().__init__()
        self.width = width
        stride = 1 if pool == "max" else 2
        pool = nn.MaxPool2d(2) if pool == "max" else nn.Identity()
        modules = [
            nn.Conv2d(colorchs, 8 * chfak, 3, stride, 1),
            activation(),
            pool,
            nn.Conv2d(8 * chfak, 8 * chfak, 3, stride, 1),
            activation(),
            pool,
            nn.Conv2d(8 * chfak, 8 * chfak, 3, stride, 1),
            activation(),
            pool,
            nn.Conv2d(8 * chfak, 16 * chfak, 3, stride, 1),
            activation(),
            pool,
            nn.Conv2d(16 * chfak, 1, 4)]
        modules.extend(end)
        self.enc = nn.Sequential(*modules)

    def forward(self, X):
        return self.enc(X)


class NewCritic(nn.Module):
    def __init__(self, width=64, dims=[8, 8, 8, 16], bottleneck=32, colorchs=3, chfak=1, activation=nn.ReLU, pool="max",
                 dropout=0.5):
        super().__init__()
        self.width = width
        stride = 1 if pool == "max" else 2
        dims = np.array(dims) * chfak
        pool = nn.MaxPool2d(2) if pool == "max" else nn.Identity()
        self.pool = pool
        features = [
            nn.Conv2d(colorchs, dims[0], 3, stride, 1),
            activation(),
            pool,
            nn.Conv2d(dims[0], dims[1], 3, stride, 1),
            activation(),
            pool,
            nn.Conv2d(dims[1], dims[2], 3, stride, 1),
            activation(),
            pool,
            nn.Dropout(dropout),
            nn.Conv2d(dims[2], dims[3], 3, stride, 1),
            activation(),
            pool,
            nn.Dropout(dropout),
            nn.Conv2d(dims[3], bottleneck * chfak, 4),
            activation()]
        self.features = nn.Sequential(*features)

        self.crit = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chfak * bottleneck, chfak * bottleneck),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(chfak * bottleneck, 1),
            nn.Sigmoid()
        )

    def forward(self, X, collect=False):
            embeds = []
            # print(list(self.features))
            for layer in list(self.features):
                X = layer(X)
                if collect and isinstance(layer, type(self.pool)):
                    embeds.append(X)
            if collect:
                embeds.append(X)
        # print("last embed", X.shape)
        pred = self.crit(X)

        if collect:
            return pred, embeds
        else:
            return pred


class Pass(nn.Module):
    def __init__(self):
        super(Pass, self).__init__()

    def forward(self, X):
        return X


class ConvEncoder64x64(nn.Module):
    def __init__(self, pooling="max", enc_dim=1, in_chs=3, chfak=1, activation=nn.ReLU, end=[]):
        super().__init__()
        stride = 1 if pooling in ["max", "avg"] else 2
        pool = nn.MaxPool2d(2) if pooling == "max" else nn.AvgPool2d(2) if pooling == "avg" else Pass()
        self.acti = activation
        modules = [
            nn.Conv2d(in_chs, 8 * chfak, 3, stride, 1),
            activation(),
            pool,
            nn.Conv2d(8 * chfak, 8 * chfak, 3, stride, 1),
            activation(),
            pool,
            nn.Conv2d(8 * chfak, 16 * chfak, 3, stride, 1),
            activation(),
            pool,
            nn.Conv2d(16 * chfak, 16 * chfak, 3, stride, 1),
            activation(),
            pool,
            nn.Conv2d(16 * chfak, enc_dim, 4),
            nn.Tanh()
        ]
        modules.extend(end)
        self.layers = modules
        self.enc = nn.Sequential(*modules)

    def forward(self, X, embed=False):
        embeds = []
        embed_dims = 0
        for layer in self.layers:
            X = layer(X)
            if (isinstance(layer, self.acti) or isinstance(layer, nn.Tanh)) and embed:
                embeds.append(F.upsample(X, size=(64, 64)))

        return (X, T.cat(embeds, dim=1)) if embed else X


class ConvDecoder64x64(nn.Module):
    def __init__(self, enc_dim=1, out_chs=3, chfak=1, activation=nn.ReLU, end=[]):
        super().__init__()
        self.ups = nn.Upsample(scale_factor=(2, 2))
        ups = self.ups
        stride = 1
        modules = [
            nn.Upsample(scale_factor=(4, 4)),
            nn.Conv2d(enc_dim, 8 * chfak, 3, stride, 1),
            activation(),
            ups,
            nn.Conv2d(8 * chfak, 8 * chfak, 3, stride, 1),
            activation(),
            ups,
            nn.Conv2d(8 * chfak, 8 * chfak, 3, stride, 1),
            activation(),
            ups,
            nn.Conv2d(8 * chfak, 8 * chfak, 3, stride, 1),
            activation(),
            ups,
            nn.Conv2d(8 * chfak, out_chs, 3, stride, 1)]
        modules.extend(end)
        self.enc = nn.Sequential(*modules)

    def forward(self, X):
        return self.enc(X)


class PolicyNet(nn.Module):
    def __init__(self, actions=None, enc_dim=64, in_chs=6, out_chs=64, chfak=1, end=[nn.Tanh()]):
        super().__init__()
        self.in_chs = in_chs
        self.actions = actions
        encoder = ConvEncoder64x64(
            in_chs=in_chs,
            chfak=chfak,
            enc_dim=enc_dim,
            end=[nn.ReLU(), nn.Flatten()]
        )
        self.policy_net = nn.Sequential(
            encoder,
            nn.Linear(enc_dim, out_chs),
            *end
        )

    def forward(self, obs):
        return self.policy_net(obs)

    def get_action(self, obs):
        with T.no_grad():
            logits = self.forward(obs)
            idxs = T.argmax(logits, dim=-1).squeeze()
            return self.actions[idxs]


class SmallCritic(nn.Module):
    def __init__(self, width=16, enc_dim=1, colorchs=3, chfak=1, activation=nn.ReLU, end=[]):
        super().__init__()
        self.width = width
        modules = [
            nn.Conv2d(colorchs, 8 * chfak, 3, 2, 1),
            activation(),
            # Printer(),
            # nn.MaxPool2d(2),
            nn.Conv2d(8 * chfak, 16 * chfak, 3, 2, 1),
            activation(),
            # Printer(),
            # nn.MaxPool2d(2),
            nn.Conv2d(16 * chfak, 1, 4),
            # Printer()
        ]
        modules.extend(end)
        self.enc = nn.Sequential(*modules)

    def forward(self, X):
        return self.enc(X)


class NoScaleCritic(nn.Module):
    def __init__(self, width=16, enc_dim=1, colorchs=3, chfak=1, activation=nn.ReLU, end=[]):
        super().__init__()
        self.width = width
        modules = [nn.Conv2d(colorchs, 64 * chfak, 3, 1, 1),
                   activation(),
                   # nn.MaxPool2d(2),
                   nn.Conv2d(64 * chfak, 16 * chfak, 3, 1, 1),
                   activation(),
                   # nn.MaxPool2d(2),
                   nn.Conv2d(16 * chfak, 1, 4)]
        modules.extend(end)
        self.enc = nn.Sequential(*modules)

    def forward(self, X):
        return self.enc(X)


class Unet(nn.Module):
    def __init__(self, width=64, edims=[8, 8, 8, 16], ddims=[8, 8, 8, 16], bottleneck=32,
                 colorchs=3, chfak=1, activation=nn.ReLU, pool="max", upsample=True, pure=False):
        super().__init__()
        edims = np.array(edims, dtype=np.int) * chfak
        ddims = np.array(ddims, dtype=np.int) * chfak
        self.width = width
        stride = 1 if pool == "max" else 2
        self.pool = nn.MaxPool2d(2) if pool == "max" else nn.Identity()
        self.acti = nn.LeakyReLU(0.2)
        self.ups = nn.Upsample(scale_factor=(2, 2))
        self.upsample = upsample
        self.pure = pure
        self.enc = [
            nn.Conv2d(colorchs, edims[0], 3, stride, 1),
            nn.Conv2d(edims[0], edims[1], 3, stride, 1),
            nn.Conv2d(edims[1], edims[2], 3, stride, 1),
            nn.Conv2d(edims[2], edims[3], 3, stride, 1),
            nn.Conv2d(edims[3], bottleneck, 4)
        ]
        self.dec = [
            nn.Conv2d(edims[0] + ddims[0], 1, 3, 1, 1) if upsample else
            nn.ConvTranspose2d(edims[0] + ddims[0], 1, 4, 2, 1),
            nn.Conv2d(edims[1] + ddims[1], ddims[0], 3, 1, 1) if upsample else
            nn.ConvTranspose2d(edims[1] + ddims[1], ddims[0], 4, 2, 1),
            nn.Conv2d(edims[2] + ddims[2], ddims[1], 3, 1, 1) if upsample else
            nn.ConvTranspose2d(edims[2] + ddims[2], ddims[1], 4, 2, 1),
            nn.Conv2d(edims[3] + ddims[3], ddims[2], 3, 1, 1) if upsample else
            nn.ConvTranspose2d(edims[3] + ddims[3], ddims[2], 4, 2, 1),
            nn.ConvTranspose2d(bottleneck, ddims[3], 4, 1, 0) if not pure else
            nn.Conv2d(bottleneck, ddims[3], 3, 1, 1)
        ]
        self.dec_model = nn.Sequential(*self.dec)
        self.enc_model = nn.Sequential(*self.enc)
        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Sigmoid()
            # nn.ReLU()
        )

    def forward(self, X, critic=False, embeds=False):
        pool = self.pool
        ups = self.ups
        acti = self.acti
        enc = self.enc
        dec = self.dec
        x0 = acti(enc[0](X))
        p0 = pool(x0)
        # print(x0.shape)
        x1 = acti(enc[1](p0))
        p1 = pool(x1)
        # print(x1.shape)
        # print(p1.shape)
        x2 = acti(enc[2](p1))
        p2 = pool(x2)
        # print(x2.shape)
        # print(p2.shape)
        x3 = acti(enc[3](p2))
        p3 = pool(x3)
        # print(x3.shape)
        # print(p3.shape)
        x4 = acti(enc[4](p3))
        # print(x4.shape)
        if critic:
            return self.critic(x4)

        if self.pure:
            u3 = acti(dec[4](ups(ups(x4))))
        else:
            u3 = acti(dec[4](x4))

        if self.upsample:
            u2 = acti(dec[3](T.cat((ups(u3), x3), dim=1)))
            # print(u2.shape)
            u1 = acti(dec[2](T.cat((ups(u2), x2), dim=1)))
            # print(u1.shape)
            u0 = acti(dec[1](T.cat((ups(u1), x1), dim=1)))
            # print(u0.shape)
            y = T.sigmoid(dec[0](T.cat((ups(u0), ups(u0)), dim=1)))
        else:
            # print(u3.shape)
            u2 = acti(dec[3](T.cat((u3, p3), dim=1)))
            # print(u2.shape)
            u1 = acti(dec[2](T.cat((u2, p2), dim=1)))
            # print(u1.shape)
            u0 = acti(dec[1](T.cat((u1, p1), dim=1)))
            # print(u0.shape)
            y = T.sigmoid(dec[0](T.cat((u0, p0), dim=1)))
        # print(y.shape)
        res = y if not embeds else (y, u0)
        return res


class UnetDecoder(nn.Module):
    def __init__(self, width=64, edims=[8, 8, 8, 16], ddims=[8, 8, 8, 16], bottleneck=32, masker_channels=16,
                 colorchs=3, chfak=1, activation=nn.ReLU, pool="max", upsample=True, pure=False):
        super().__init__()
        edims = np.array(edims, dtype=np.int) * chfak
        ddims = np.array(ddims, dtype=np.int) * chfak
        bottleneck *= chfak
        self.width = width
        stride = 1 if pool == "max" else 2
        self.pool = nn.MaxPool2d(2) if pool == "max" else nn.Identity()
        self.acti = nn.LeakyReLU(0.01)
        self.ups = nn.Upsample(scale_factor=(2, 2))
        self.upsample = upsample
        self.pure = pure
        self.masker_channels = masker_channels
        """self.old_dec = [
            nn.Conv2d(edims[0] + ddims[0], 1, 3, 1, 1) if upsample else
                nn.ConvTranspose2d(edims[0] + ddims[0], 1, 4, 2, 1),
            nn.Conv2d(edims[1] + ddims[1], ddims[0], 3, 1, 1) if upsample else
                nn.ConvTranspose2d(edims[1] + ddims[1], ddims[0], 4, 2, 1),
            nn.Conv2d(edims[2] + ddims[2], ddims[1], 3, 1, 1) if upsample else
                nn.ConvTranspose2d(edims[2] + ddims[2], ddims[1], 4, 2, 1),
            nn.Conv2d(edims[3] + ddims[3], ddims[2], 3, 1, 1) if upsample else
                nn.ConvTranspose2d(edims[3] + ddims[3], ddims[2], 4, 2, 1),
            nn.ConvTranspose2d(bottleneck, ddims[3], 4, 1, 0) if not pure else
                nn.Conv2d(bottleneck, ddims[3], 3, 1, 1)
        ]"""
        self.dec = [
            nn.Conv2d(edims[0] + ddims[1], ddims[0], 3, 1, 1),
            nn.Conv2d(edims[1] + ddims[2], ddims[1], 3, 1, 1),
            nn.Conv2d(edims[2] + ddims[3], ddims[2], 3, 1, 1),
            nn.Conv2d(edims[3] + bottleneck, ddims[3], 3, 1, 1),
            nn.Conv2d(bottleneck, bottleneck, 1, 1, 0)
        ]
        self.dec_model = nn.Sequential(*self.dec)
        self.masker = nn.Sequential(
            nn.Conv2d(colorchs + ddims[0], self.masker_channels, 3, 1, 1),
            self.acti,
            nn.Conv2d(self.masker_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, X, embeds):
        pool = self.pool
        ups = self.ups
        acti = self.acti
        dec = self.dec

        in4 = embeds[4]
        out4 = dec[4](in4)

        upped4 = ups(ups(out4))
        in3 = T.cat((embeds[3], upped4), dim=1)
        out3 = dec[3](in3)

        upped3 = ups(out3)
        in2 = T.cat((embeds[2], upped3), dim=1)
        out2 = dec[2](in2)

        upped2 = ups(out2)
        in1 = T.cat((embeds[1], upped2), dim=1)
        out1 = dec[1](in1)

        upped1 = ups(out1)
        in0 = T.cat((embeds[0], upped1), dim=1)
        out0 = dec[0](in0)

        upped0 = ups(out0)
        inout = T.cat((X, upped0), dim=1)
        mask = self.masker(inout)

        return mask


class SmallUnet(nn.Module):
    def __init__(self, width=64, edims=[8, 8, 16], ddims=[8, 8, 16], colorchs=3, chfak=1, activation=nn.ReLU):
        super().__init__()
        edims = np.array(edims, dtype=np.int) * chfak
        ddims = np.array(ddims, dtype=np.int) * chfak
        self.width = width
        self.pool = nn.MaxPool2d(2)
        self.acti = activation()
        self.ups = nn.Upsample(scale_factor=(2, 2))
        self.enc = [
            nn.Conv2d(colorchs, edims[0], 3, 1, 1),
            nn.Conv2d(edims[0], edims[1], 3, 1, 1),
            nn.Conv2d(edims[1], edims[2], 4, 1, 0)
        ]
        self.dec = [
            nn.Conv2d(edims[0] + ddims[0], 1, 3, 1, 1),
            nn.Conv2d(edims[1] + ddims[1], ddims[0], 3, 1, 1),
            nn.ConvTranspose2d(ddims[2], ddims[1], 4, 1, 0)
        ]
        self.dec_model = nn.Sequential(*self.dec)
        self.enc_model = nn.Sequential(*self.enc)

    def forward(self, X):
        x0 = self.acti(self.enc[0](X))
        # print(x0.shape)
        x1 = self.acti(self.enc[1](self.pool(x0)))
        # print(x1.shape)
        x2 = self.acti(self.enc[2](self.pool(x1)))
        # print(x2.shape)
        u1 = self.acti(self.dec[2](x2))
        # print(u1.shape, x1.shape)
        u0 = self.acti(self.dec[1](T.cat((self.ups(u1), x1), dim=1)))
        # print(u0.shape)
        y = T.sigmoid(self.dec[0](T.cat((self.ups(u0), x0), dim=1)))
        # print(y.shape)
        return y


class FlexibleUnet(nn.Module):
    def __init__(self, in_dim, chs, wid, hidfac, dropout=False, ks=4, stride=2, pad=1, neck=1, altconv=False):
        super().__init__()
        """
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, 8, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, chs, 4, 2, 1),
            nn.Sigmoid()
        )
        """
        if altconv:
            ks = 3
            stride = 1
        folds = range(1, int(np.math.log2(wid)))
        bottleneck = max(folds)
        acti = nn.ReLU
        drop = lambda: nn.Dropout(0.3, inplace=True)
        convs = [
            nn.Conv2d(int(2 ** (2 + i) * hidfac), int(2 ** (3 + i) * hidfac * (neck if i == bottleneck else 1)), ks,
                      stride, pad) for i in folds]
        encoder = [nn.Conv2d(in_dim, int(8 * hidfac), ks, stride, pad), acti()] + [acti() if i % 2 else convs[i // 2]
                                                                                   for i in range(2 * len(folds))]
        trans_convs = [nn.ConvTranspose2d(int(2 ** (3 + i) * hidfac * (neck if i == bottleneck else 1)),
                                          int(2 ** (2 + i) * hidfac), 4, 2, 1) for i in reversed(folds)]
        decoder = [acti() if i % 2 else trans_convs[i // 2] for i in range(2 * len(folds))] + [
            nn.ConvTranspose2d(int(8 * hidfac), chs, ks, stride, pad), nn.Sigmoid()]
        modules = encoder + decoder
        # add dropout:
        if dropout:
            tmp_mods = []  # add dropouts before each acti
            for mod in modules:
                tmp_mods.append(mod)
                if type(mod) == type(acti()):
                    tmp_mods.append(drop())
            print(tmp_mods)
            modules = tmp_mods

        if altconv:
            tmp_mods = []  # add dropouts before each acti
            for mod in modules:
                tmp_mods.append(mod)
                if type(mod) == type(acti()):
                    tmp_mods.append(nn.MaxPool2d(2, 2))
            print(tmp_mods)
            modules = tmp_mods

        self.model = nn.Sequential(*modules)
        # print(self.model.state_dict().keys())

        """
        convs = [(2**(2+i), 2**(3+i)) for i in folds]
        trans_convs = [(2**(3+i), 2**(2+i)) for i in reversed(folds)]
        print(convs)
        print(trans_convs)
        encoder = [(in_dim,8), 'acti'] + [f"acti" if i%2 else convs[i//2] for i in range(2*len(folds))]
        print(encoder)
        decoder = [f"acti" if i%2 else trans_convs[i//2] for i in range(2*len(folds))] + [(8,chs), 'Sigmoid']
        print(decoder)
        print(*(encoder+decoder), sep='\n')
        """

    def forward(self, X):
        return self.model(X)


class GroundedUnet(nn.Module):
    def __init__(self, width=64, edims=[8, 8, 8, 16, 32], ddims=[8, 8, 8, 16, 32], colorchs=3, activation=nn.ReLU):
        super().__init__()
        self.width = width
        self.pool = nn.MaxPool2d(2)
        self.acti = activation()
        self.ups = nn.Upsample(scale_factor=(2, 2))
        self.down = lambda x: F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        self.enc = [
            nn.Conv2d(colorchs, edims[0], 3, 1, 1),
            nn.Conv2d(3 + edims[0], edims[1], 3, 1, 1),
            nn.Conv2d(3 + edims[1], edims[2], 3, 1, 1),
            nn.Conv2d(3 + edims[2], edims[3], 3, 1, 1),
            nn.Conv2d(edims[3], edims[4], 4)
        ]
        self.dec = [
            nn.Conv2d(edims[0] + ddims[0], 1, 3, 1, 1),
            nn.Conv2d(edims[1] + ddims[1], ddims[0], 3, 1, 1),
            nn.Conv2d(edims[2] + ddims[2], ddims[1], 3, 1, 1),
            nn.Conv2d(edims[3] + ddims[3], ddims[2], 3, 1, 1),
            nn.ConvTranspose2d(ddims[4], ddims[3], 4, 1, 0)
        ]
        self.dec_model = nn.Sequential(*self.dec)
        self.enc_model = nn.Sequential(*self.enc)

    def forward(self, X):
        pool = self.pool
        ups = self.ups
        acti = self.acti
        enc = self.enc
        dec = self.dec
        x0 = acti(enc[0](X))
        # print(x0.shape)
        d1 = self.down(X)
        x1 = acti(enc[1](T.cat((pool(x0), d1), dim=1)))
        # print(x1.shape)
        d2 = self.down(d1)
        x2 = acti(enc[2](T.cat((pool(x1), d2), dim=1)))
        # print(x2.shape)
        d3 = self.down(d2)
        x3 = acti(enc[3](T.cat((pool(x2), d3), dim=1)))
        # print(x3.shape)
        x4 = acti(enc[4](pool(x3)))
        # print(x4.shape)
        u3 = acti(dec[4](x4))
        # print(u3.shape)
        u2 = acti(dec[3](T.cat((ups(u3), x3), dim=1)))
        # print(u2.shape)
        u1 = acti(dec[2](T.cat((ups(u2), x2), dim=1)))
        # print(u1.shape)
        u0 = acti(dec[1](T.cat((ups(u1), x1), dim=1)))
        # print(u0.shape)
        y = T.sigmoid(dec[0](T.cat((ups(u0), x0), dim=1)))
        # print(y.shape)

        return y


class ResNetCritic(nn.Module):
    def __init__(self, HSV=True):
        super().__init__()
        self.HSV = HSV
        self.norm = get_normalizer()
        self.mean = T.tensor([0.485, 0.456, 0.406])
        self.std = T.tensor([0.229, 0.224, 0.225])
        self.resnet = get_resnet18_features()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, X):
        device = X.device
        if X.max() > 1:
            X = X / 255.0
        X = X.permute(0, 2, 3, 1)
        if device != self.mean.device:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        X = (X - self.mean) / self.std
        X = X.permute(0, 3, 1, 2)
        features = self.resnet(X)
        return self.head(features)


class VGGEmbedder(nn.Module):
    def __init__(self):
        super(VGGEmbedder, self).__init__()
        self.model = get_vgg_features()
        self.normer = get_normalizer()

    def forward(self, X, hsv=False):
        if hsv:
            X = T.from_numpy(hsv_to_rgb(X))
        X = self.normer(X).permute(0, 3, 1, 2).float()

        embeds = []
        for layer in self.model.children():
            X = layer(X)
            if (isinstance(layer, nn.ReLU) or isinstance(layer, nn.Tanh)):
                embeds.append(F.interpolate(X, size=(64, 64), mode="bilinear"))

        return (X, T.cat(embeds, dim=1))


def get_vgg_features(frozen=True):
    vgg = visionmodels.vgg11(pretrained=True).features
    if frozen:
        for feature in vgg:
            feature.requires_grad = False
    return vgg


class VGGCritic(nn.Module):
    def __init__(self, resize=(64, 64)):
        super().__init__()
        self.features = get_vgg_features()
        fak = resize[0] // 64
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * fak * 2 * fak, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.normer = get_normalizer()
        self.resize = resize

    def forward(self, X, hsv=False, format=True):
        if hsv:
            X = T.from_numpy(hsv_to_rgb(X.permute(0, 2, 3, 1).numpy())).permute(0, 3, 1, 2)
        if format:
            X = self.normer(X.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.resize != (64, 64):
            X = F.interpolate(X, self.resize)
        # print("raw",X.shape)
        X = self.features(X)
        # print("features", X.shape)
        return self.head(X)


def get_resnet18_features():
    resnet18 = visionmodels.resnet18(pretrained=True)
    features = nn.Sequential(*(list(resnet18.children())[0:8]))
    return features


def get_inceptionv3_features():
    net = visionmodels.inception_v3(pretrained=True)
    features = nn.Sequential(*(list(net.children())[0:8]))
    return features


def get_normalizer():
    # return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return lambda x: (x - T.tensor([0.485, 0.456, 0.406])) / T.tensor([0.229, 0.224, 0.225])


if __name__ == "__main__":
    print((list(get_vgg_features().children())))
    exit()
    bignet = get_resnet18_features()
    X = T.randn(12, 3, 64, 64)
    unet = Unet()
    Z = unet(X)
    ZZ = bignet(X)
    print(ZZ.shape)
    # print(bignet)

