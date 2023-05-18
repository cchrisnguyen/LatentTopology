# Latent Space Topology of Several Machine Learning Algorithms

This project explores the topology of several machine learning algorithms using the Fashion MNIST dataset. To compare them, manifold learning algorithms were applied to find the underlying structure of the latent space. Specifically, manifold learning was applied on the original data, on lower rank images obtained through RPCA, and on the latent vectors of images obtained through training VAE and BiGAN models in PyTorch.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Scikit-learn
- Tensorboard (optional, for learning process visualization)
- Matplotlib (optional, for result visualization)

## Usage

1. Clone or download the repository to your local machine.

2. Install the required packages using pip or conda:

```
pip install opencv-python numpy scikit-learn tensorboard matplotlib
```

3. Follow the instructions in `demonstration.ipynb`.

4. Reconstructed and generated images will be saved in the `results` folder while trained models can be found under the `models` folder.

## Credits

This project is inspired by the papers:

1. Lin, Z., Chen, M., & Ma, Y. (2010). The augmented lagrange multiplier method for exact recovery of corrupted low-rank matrices. arXiv preprint arXiv:1009.5055.

2. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

3. Donahue, J., Krähenbühl, P., & Darrell, T. (2016). Adversarial feature learning. arXiv preprint arXiv:1605.09782.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).