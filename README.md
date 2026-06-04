# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
This code implements a Denoising Autoencoder using PyTorch to clean noisy images from the MNIST dataset. It uses a convolutional neural network architecture, where the encoder compresses the input image into a lower-dimensional representation, and the decoder reconstructs the original image from this compressed form. To train the model to remove noise, Gaussian noise is added to the clean images, and the network learns to recover the original from the noisy version. The training process uses Mean Squared Error (MSE) as the loss function to measure the reconstruction error and the Adam optimizer to update the model weights. The autoencoder is trained over multiple epochs using mini-batches of data for efficiency. After training, the model's performance is visually evaluated by displaying the original, noisy, and denoised images side by side.

## DESIGN STEPS
STEP 1:
Problem Understanding and Dataset Selection

STEP 2:
Preprocessing the Dataset

STEP 3:
Design the Convolutional Autoencoder Architecture

STEP 4:
Compile and Train the Model

STEP 5:
Evaluate the Model

STEP 6:
Visualization and Analysis



## PROGRAM

### Name:Sarankumar.V

### Register Number:212224220089

```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, 7, 7]
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [B, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name: Sarankumar.V")
    print("Register Number:212224220089")

    for epoch in range(epochs):
        running_loss = 0.0

        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")



```

### OUTPUT

### Model Summary

<img width="850" height="589" alt="image" src="https://github.com/user-attachments/assets/ce0fa9b2-a6ce-4993-999f-f0a09c31de1b" />


### Training loss

## Original vs Noisy Vs Reconstructed Image

<img width="961" height="440" alt="image" src="https://github.com/user-attachments/assets/31346e3d-cc30-4f34-9f20-09c992848ae4" />


## RESULT
The convolutional autoencoder model was successfully developed and trained for image denoising. The model effectively removed noise from corrupted images and reconstructed cleaner images with improved visual quality. The encoder extracted important image features, while the decoder restored the denoised images accurately. The performance of the model demonstrated that convolutional autoencoders are effective for image denoising applications.
