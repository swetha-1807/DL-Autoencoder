# DL- Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
In practical scenarios, images often contain noise that degrades the performance of computer vision models. A convolutional autoencoder learns compressed representations of images and reconstructs them, which can be used to remove noise.
Dataset: MNIST (28×28 grayscale images of handwritten digits)
Noise: Gaussian noise will be added to simulate real-world scenarios

## DESIGN STEPS

### STEP 1: 

Import required libraries such as PyTorch, torchvision, and matplotlib for building and visualizing the model.

### STEP 2: 

Download the MNIST dataset and convert the images into tensors for training.

### STEP 3: 

Introduce Gaussian noise to the images so the model can learn how to remove it.

### STEP 4: 

Create an encoder with convolution layers and a decoder with transposed convolution layers to reconstruct images.

### STEP 5: 

Use noisy images as input and clean images as output. Apply MSE loss and Adam optimizer during training.

### STEP 6: 

Compare original, noisy, and reconstructed images to check how well the model removes noise.

## PROGRAM

### Name: SWETHA K

### Register Number: 212224230284

```

# Autoencoder Definition
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
      x = self.encoder(x)
      x = self.decoder(x)
      return x


# Initialize model
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training function
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name: BAVYA SRI B")
    print("Register Number: 212224230034")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)


            #forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            #Backward pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):4f}")
# Visualization function
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: BAVYA SRI B ")
    print("Register Number: 212224230034")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


```

### OUTPUT

### Model Summary

![Model Summary](https://github.com/swetha-1807/DL-Autoencoder/blob/main/Screenshot%202026-03-13%20212630.png?raw=true)

### Training loss


## Original vs Noisy Vs Reconstructed Image



## RESULT
Include your result here
