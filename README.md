# X-VAE-keras
Simple implementations of different kinds of VAE with tf.keras.

VAE models included are:
1. simple VAE
2. VAE with mmd loss instead of classic kl-divergence loss
3. VQ-VAE with a PixelCNN prior (thanks to Amelie Royer)
4. Categorical VAE (Gumbel-Softmax)
5. VAEGAN
6. Conditional VAE
7. VAE with 2D latent space

P.S: Implementations were made based on the original paper of each model.<br />
Codes were written based on tensorflow=1.15 and its keras module. <br />
All models are tested with MNIST dataset and useful plots are included.
