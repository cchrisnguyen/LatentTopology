import os
import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils import data
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh, Linear
from torchvision import datasets, transforms, utils
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter

# --- hyper-parameters --- #
BATCH_SIZE = 128
ITER = 200000
IMAGE_SIZE = 28
NUM_CHANNELS = 1
DIM = 128           
NLAT = 100            # latent dimension
LEAK = 0.2

C_ITERS = 5         # critic iterations
EG_ITERS = 1        # encoder / generator iterations
LAMBDA = 10         # strength of gradient penalty
LEARNING_RATE = 1e-4
BETA1 = 0.5
BETA2 = 0.9

# --- joint critic --- #
class JointCritic(nn.Module):
    def __init__(self, NLAT):
        """ A joint Wasserstein critic function.

        Args:
            x_mapping: An nn.Sequential module that processes x.
            z_mapping: An nn.Sequential module that processes z.
            joint_mapping: An nn.Sequential module that process the output of x_mapping and z_mapping.
        """
        super().__init__()

        self.x_net = nn.Sequential(
            Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
            Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
            Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
            Conv2d(DIM * 4, DIM * 4, 3, 1, 0), LeakyReLU(LEAK))
        
        self.z_net = nn.Sequential(
            Conv2d(NLAT, 512, 1, 1, 0), LeakyReLU(LEAK),
            Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK))
        
        self.joint_net = nn.Sequential(
            Conv2d(DIM * 4 + 512, 1024, 1, 1, 0), LeakyReLU(LEAK),
            Conv2d(1024, 1024, 1, 1, 0), LeakyReLU(LEAK),
            Conv2d(1024, 1, 1, 1, 0))

    def forward(self, x, z):
        assert x.size(0) == z.size(0)
        x_out = self.x_net(x)
        z_out = self.z_net(z)
        joint_input = torch.cat((x_out, z_out), dim=1)
        output = self.joint_net(joint_input)
        return output

# --- BiGAN --- #
class WALI(nn.Module):
    def __init__(self, NLAT):
        """ Adversarially learned inference (a.k.a. bi-directional GAN) with Wasserstein critic.

        Args:
            E: Encoder p(z|x).
            G: Generator p(x|z).
            C: Wasserstein critic function f(x, z).
        """
        super().__init__()

        self.E = nn.Sequential(
            Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(),
            Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(),
            Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(),
            Conv2d(DIM * 4, DIM * 4, 3, 1, 0, bias=False), BatchNorm2d(DIM * 4), ReLU(),
            Conv2d(DIM * 4, NLAT, 1, 1, 0)
            )
        
        self.G = nn.Sequential(
            ConvTranspose2d(NLAT, DIM * 4, 4, 1, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(),
            ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(),
            ConvTranspose2d(DIM * 2, DIM, 3, 2, 0, bias=False), BatchNorm2d(DIM), ReLU(),
            ConvTranspose2d(DIM, NUM_CHANNELS, 4, 3, 0, bias=False), Tanh())
        
        self.C = JointCritic(NLAT)

    def get_encoder_parameters(self):
        return self.E.parameters()

    def get_generator_parameters(self):
        return self.G.parameters()

    def get_critic_parameters(self):
        return self.C.parameters()

    def encode(self, x):
        return self.E(x)

    def generate(self, z):
        return self.G(z)

    def reconstruct(self, x):
        return self.generate(self.encode(x))

    def criticize(self, x, z_hat, x_tilde, z):
        input_x = torch.cat((x, x_tilde), dim=0)
        input_z = torch.cat((z_hat, z), dim=0)
        output = self.C(input_x, input_z)
        data_preds, sample_preds = output[:x.size(0)], output[x.size(0):]
        return data_preds, sample_preds

    def calculate_grad_penalty(self, x, z_hat, x_tilde, z):
        bsize = x.size(0)
        eps = torch.rand(bsize, 1, 1, 1).to(x.device) # eps ~ Unif[0, 1]
        intp_x = eps * x + (1 - eps) * x_tilde
        intp_z = eps * z_hat + (1 - eps) * z
        intp_x.requires_grad = True
        intp_z.requires_grad = True
        C_intp_loss = self.C(intp_x, intp_z).sum()
        grads = autograd.grad(C_intp_loss, (intp_x, intp_z), retain_graph=True, create_graph=True)
        grads_x, grads_z = grads[0].view(bsize, -1), grads[1].view(bsize, -1)
        grads = torch.cat((grads_x, grads_z), dim=1)
        grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def forward(self, x, z, lamb=10):
        z_hat, x_tilde = self.encode(x), self.generate(z)
        data_preds, sample_preds = self.criticize(x, z_hat, x_tilde, z)
        EG_loss = torch.mean(data_preds - sample_preds)
        C_loss = -EG_loss + lamb * self.calculate_grad_penalty(x.data, z_hat.data, x_tilde.data, z.data)
        return C_loss, EG_loss


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    wali = WALI(NLAT).to(device)

    optimizerEG = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()), 
        lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizerC = Adam(wali.get_critic_parameters(), 
        lr=LEARNING_RATE, betas=(BETA1, BETA2))
    
    svhn = datasets.FashionMNIST('./', train=True, download=True, 
                                transform=transforms.Compose(
                                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
    loader = data.DataLoader(svhn, BATCH_SIZE, shuffle=True)
    noise = torch.randn(64, NLAT, 1, 1, device=device)

    curr_iter = C_iter = EG_iter = 0
    C_update, EG_update = True, False

    if not os.path.exists(f'results/BiGAN_{NLAT}'):
        os.makedirs(f'results/BiGAN_{NLAT}')

    model_name = f"bigan_{NLAT}_{datetime.datetime.now().strftime('%H:%M')}"
    writer = SummaryWriter('runs/' + model_name)

    print('Training starts...')

    while curr_iter < ITER:
        for _, (x, _) in enumerate(loader):
            x = x.to(device)

            if curr_iter == 0:
                init_x = x
                curr_iter += 1

            z = torch.randn(x.size(0), NLAT, 1, 1).to(device)
            C_loss, EG_loss = wali(x, z, lamb=LAMBDA)

            if C_update:
                optimizerC.zero_grad()
                C_loss.backward()
                writer.add_scalar('C Loss', C_loss.item(), curr_iter)
                optimizerC.step()
                C_iter += 1

                if C_iter == C_ITERS:
                    C_iter = 0
                    C_update, EG_update = False, True
                    continue

            if EG_update:
                optimizerEG.zero_grad()
                EG_loss.backward()
                writer.add_scalar('G Loss', EG_loss.item(), curr_iter)
                optimizerEG.step()
                EG_iter += 1

                if EG_iter == EG_ITERS:
                    EG_iter = 0
                    C_update, EG_update = True, False
                    curr_iter += 1
                else:
                    continue

            # print training statistics
            if curr_iter % 100 == 0:
                print('[%d/%d]\tW-distance: %.4f\tC-loss: %.4f' % (curr_iter, ITER, EG_loss.item(), C_loss.item()))

                # plot reconstructed images and samples
                wali.eval()
                real_x, rect_x = init_x[:32], wali.reconstruct(init_x[:32]).detach_()
                rect_imgs = torch.cat((real_x.unsqueeze(1), rect_x.unsqueeze(1)), dim=1) 
                rect_imgs = rect_imgs.view(64, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).cpu()
                genr_imgs = wali.generate(noise).detach_().cpu()
                utils.save_image(rect_imgs * 0.5 + 0.5, f'results/BiGAN_{NLAT}/rect_{curr_iter}.png')
                utils.save_image(genr_imgs * 0.5 + 0.5, f'results/BiGAN_{NLAT}/genr_{curr_iter}.png')
                wali.train()

    torch.save(wali.state_dict(), f'models/bigan_{NLAT}.pt')
    print("DONE!")