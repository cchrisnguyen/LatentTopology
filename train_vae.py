import os
import datetime
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

# --- hyper-parameters --- #
BATCH_SIZE = 128
EPOCHS = 50
LOG_INTERVAL = 100
Z_DIM = 432
LEARNING_RATE = 1e-4
INIT_CHANNELS = 8
IMAGE_CHANNELS = 1

class VAE(nn.Module):
    def __init__(self, Z_DIM):
        super(VAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=IMAGE_CHANNELS, out_channels=INIT_CHANNELS, kernel_size=4, 
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=INIT_CHANNELS, out_channels=INIT_CHANNELS*2, kernel_size=4, 
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=INIT_CHANNELS*2, out_channels=INIT_CHANNELS*4, kernel_size=4, 
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=INIT_CHANNELS*4, out_channels=64, kernel_size=3, 
            stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, Z_DIM)
        self.fc_log_var = nn.Linear(128, Z_DIM)
        self.fc2 = nn.Linear(Z_DIM, 512)
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=512, out_channels=INIT_CHANNELS*8, kernel_size=4, 
            stride=1, padding=1
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=INIT_CHANNELS*8, out_channels=INIT_CHANNELS*4, kernel_size=4, 
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=INIT_CHANNELS*4, out_channels=INIT_CHANNELS*2, kernel_size=3, 
            stride=2, padding=0
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=INIT_CHANNELS*2, out_channels=IMAGE_CHANNELS, kernel_size=4, 
            stride=3, padding=0
        )
    
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
    
    def decode(self, z):
        z = self.fc2(z)
        z = z.view(-1, 512, 1, 1)
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        return torch.sigmoid(self.dec4(x))
    
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


# --- defines the loss function --- #
def loss_function(recon_x, x, mu, logvar):

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)

    return BCE + KLD


# --- train and test --- #
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # data: [batch size, 1, 28, 28]
        # label: [batch size] -> we don't use
        optimizer.zero_grad()
        data = data.to(device)
        recon_data, mu, logvar = model(data)
        loss = loss_function(recon_data, data, mu, logvar)
        loss.backward()
        cur_loss = loss.item()
        train_loss += cur_loss
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.*batch_idx / len(train_loader),
                cur_loss/len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)
    ))
    writer.add_scalar('Train loss', train_loss / len(train_loader.dataset), epoch)

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_data, mu, logvar = model(data)
            cur_loss = loss_function(recon_data, data, mu, logvar).item()
            test_loss += cur_loss
            if batch_idx == 0:
                # saves 8 samples of the first batch as an image file to compare input images and reconstructed images
                num_samples = min(BATCH_SIZE, 8)
                comparison = torch.cat(
                    [data[:num_samples], recon_data.view(BATCH_SIZE, 1, 28, 28)[:num_samples]]).cpu()
                save_generated_img(
                    comparison, 'reconstruction', epoch, num_samples)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    writer.add_scalar('Tess loss', test_loss, epoch)

# --- etc. funtions --- #
def save_generated_img(image, name, epoch, nrow=8):
    if not os.path.exists(f'results/VAE_{Z_DIM}'):
        os.makedirs(f'results/VAE_{Z_DIM}')

    if epoch % 5 == 0:
        save_path = f'results/VAE_{Z_DIM}/{name}_{epoch}.png'
        save_image(image, save_path, nrow=nrow)

# --- main function --- #
if __name__ == '__main__':

    # --- model --- #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(Z_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model_name = f"vae_{Z_DIM}_{datetime.datetime.now().strftime('%H:%M')}"
    writer = SummaryWriter('runs/' + model_name)

    # --- data loading --- #
    train_data = datasets.FashionMNIST('./', train=True, download=True,
                                transform=transforms.ToTensor())
    test_data = datasets.FashionMNIST('./', train=False,
                            transform=transforms.ToTensor())

    # pin memory provides improved transfer speed
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            # p(z) = N(0,I), this distribution is used when calculating KLD. So we can sample z from N(0,I)
            sample = torch.randn(16, Z_DIM).to(device)
            sample = model.decode(sample).cpu().view(16, 1, 28, 28)
            save_generated_img(sample, 'sample', epoch, nrow=4)

    torch.save(model.state_dict(), f'models/vae_{Z_DIM}.pt')
    print("DONE!")
    