## A python package for tensor randomized Kaczmarz algorithms


This repository implements several **randomized Kaczmarz-type algorithms for large-scale tensor linear equations under the t-product**, with a (particular) focus on inconsistent tensor problems.

We implement the algorithms proposed in the following papers:

 - ["Tensor randomized extended Kaczmarz methods for large
inconsistent tensor linear equations with t-product", Guang-Xin Huang1](https://doi.org/10.1007/s11075-023-01684-w)
- ["Randomized extended average block Kaczmarz method
for inconsistent tensor equations under t-product",Liyuan An1 Kun Liang1 Han Jiao1 Qilong Liu1](https://doi.org/10.1007/s11075-024-01982-x)
- etc.

To get you understand the importance of randomized Kaczmarz algorithms and its applications, we recommend  you the following survey paper:

- ["A randomized Kaczmarz algorithm with exponential convergence", Thomas Strohmer, Roman Vershynin](https://arxiv.org/abs/math/0702226) (seminal!)
- ["Survey of a Class of Iterative Row-Action Methods: The Kaczmarz Method", Inês A. Ferreira, Juan A. Acebrón, José Monteiro ](https://arxiv.org/abs/2401.02842), 2024, and the references therein.
- etc.

Tensors are multidimenssional arrrays that generalize matrices to higher dimensions, and are intensively used to represent mutlidimensional data in various fields such as signal processing, computet vision, machine learning, etc.  The *t-product* is a specific way to define multiplication between tensors, which allows us to extend many matrix operations and algorithms to the tensor setting. You can read more about the t-product in the following papers:
- ["Third-order tensors as operators on matrices: a theoretical and computational framework with applications to imaging", Misha E. Kilmer, Carla D. Martin, and Kristin R. Muhič](https://epubs.siam.org/doi/10.1137/130912300).
- ["Tensor-tensor products with invertible linear transforms", Canyi Lu, Jiqing Bi, Shuchin Aeron, and Zhouchen Lin](https://arxiv.org/abs/1606.05535).


Several problems can be formulated as tensor linear equations, which can be solve iteratively using a Kaczmarz-type methods. For instance, image inpainting, image deblurring, and video recovery can be formulated as tensor linear equations. Hence, they can be solved using the algorithms implemented in this package.


We will add some example and documentation soon. For now on you can check the detailed examples in the paper above and the references therein. Also  check the repo [Tensor-Tensor-Toolbox](https://github.com/canyilu/Tensor-tensor-product-toolbox) for more details about the t-product and its applications.


## Quick Start
To install the package, you can use pip:

```bash
pip install git+https://github.com/jnlandu/tensor-randomized-kaczmarz-algorithms"
```

Then you can import the package in your Python code:

```python
from trk_algorithms.methods import trek_algorithm
```
to import the Tensor Randomized Extended Kaczmarz (TREK) algorithm.

### Usage
Here is a simple example of how to use the TREK algorithm to solve a tensor linear equation:

```python
import torch
from trk_algorithms.utils import make_tensor_problem
from trk_algorithms.methods import trek_algorithm

# Create test problem
m, n, p, q = 150, 100, 10, 10
noise = 0.01
eta = 0.5
max_iter = 5000
A, X_ls, B = make_tensor_problem(m=m, n=n, p=p, q=q, noise=noise, seed=1)
print(f"Tensor dimensions: A: {A.shape}, B: {B.shape}, X: {X_ls.shape}")

(X_trek, k_trek, hist_trek, x_hist_trek), t_trek = trek_algorithm(A, B, X_ls, T=max_iter)
```
Here are  the  output's details:
- `X_trek`: The approximate solution tensor obtained by the TREK algorithm.
- `k_trek`: The number of iterations taken by the TREK algorithm to converge.
- `hist_trek`: A history dictionary containing information about the convergence process, such as residuals at each iteration.
- `x_hist_trek`: A list of solution tensors at each iteration.
- `t_trek`: The total time taken by the TREK algorithm to converge, i.e. the time take to the RSE to be less than 1e-5 or the maximum number of iterations is reached.


## Requirements & Installation
- Python 3.8 or higher
- PyTorch
- NumPy
- Tensor_toolbox (for t-product operations).

You can install the  tensor_toolbox from [here](https://github.com/jnlandu/tensor-tensor-toolbox-in-python) or 
```bash
pip install git+https://github.com/jnlandu/tensor-tensor-toolbox-in-python
```
which will automaticall install the required dependencies (torch, numpy, etc.).

For developpersm you can clone the repository and install it in editable mode:
```bash
git clone https://github.com/jnlandu/tensor-randomized-kaczmarz-algorithms
cd tensor-randomized-kaczmarz-algorithms
pip install -e .
```
Make sure to install the tensor_toolbox as well. It is advisable to use a virtual environment to avoid dependency conflicts. If you are not use, here is a simple way to create and activate a virtual environment using `venv`:

```bash
python -m venv trk_env
source trk_env/bin/activate  # On Windows use `trk_env\Scripts\activate`
```

## Folder Structure
- `trk_algorithms/`: The main package directory containing the implementation of the algorithms.
  - `methods.py`: Contains the implementations of various randomized Kaczmarz algorithms.
  - `utils.py`: Utility functions for block generation (for block averaging methods) and problem generation.
- `example.ipynb`: A Jupyter notebook demonstrating how to use the package.

## Contributing
Contributions are welcome! If you find a bug or have a feature request, please open an issue or submit a PR.
## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/jnlandu/tensor-randomized-kaczmarz-algorithms/blob/main/LICENSE).










