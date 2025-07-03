# Gravity and magnetic forward modelling of ellipsoids

This repository contains the material generated during Kelly Baker's Summer
Research Project on implementing the gravity and magnetic forward modelling of
ellipsoids.

## Create environment

### Install a Python distribution

Start by installing a Python distribution like
[Miniforge](https://github.com/conda-forge/miniforge).

### Clone this repository

Use `git` to clone this repository using SSH:

```bash
git clone git@github.com:ubcgif/2025-baker-ellipsoids.git
```

or using HTTPS if you are not planning to push changes to the repo:

```bash
git clone https://github.com/ubcgif/2025-baker-ellipsoids
```

> [!IMPORTANT]
> Remember to configure an [SSH key in your GitHub
> account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).
> Follow Software
> Carpentry's Git lesson on [setting up SSH keys in
> GitHub](https://swcarpentry.github.io/git-novice/07-github.html#ssh-background-and-setup)
> for more details.

### Create a new conda environment

Navigate to the cloned repo:

```bash
cd 2025-baker-ellipsoids
```

And create a new conda environment using the `environment.yml` file:

```bash
conda env create -f environment.yml
```

### Activate the environment

Once the new environment is created and all the packages are installed, we can
to activate the environment with:

```bash
conda activate 2025-baker-ellipsoids
```

### Where to start

All example plots and function are within the notebooks folder. \notebooks shows the jupyter notebook examples for how to use and apply the code. \notebooks\functions contains .py files of all functions and tests which build the code.

## License

All code is provided under the [MIT License](LICENSE).
