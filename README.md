<div align="center">

# xhec-mlops-project-student

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

This repository has for purpose to industrialize the [Abalone age prediction](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) Kaggle contest.

<details>
<summary>Details on the Abalone Dataset</summary>

The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age.

**Goal**: predict the age of abalone (column "Rings") from physical measurements ("Shell weight", "Diameter", etc...)

You can download the dataset on the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)

</details>

## Table of Contents

- [xhec-mlops-project-student](#xhec-mlops-project-student)
  - [Table of Contents](#table-of-contents)
  - [Deliverables and Evaluation](#deliverables-and-evaluation)
    - [Deliverables](#deliverables)
    - [Evaluation](#evaluation)
  - [Steps to reproduce to build the deliverable](#steps-to-reproduce-to-build-the-deliverable)
    - [Pull requests in this project](#pull-requests-in-this-project)
    - [Tips to work on this project](#tips-to-work-on-this-project)
- [Setup Instructions](#setup-instructions)
  - [Environment Setup](#environment-setup)
    - [Option 1: Setting up with Conda](#option-1-setting-up-with-conda)
    - [Option 2: Setting up without Conda](#option-2-setting-up-without-conda)


## Deliverables and notation

### Deliverables

The deliverable of this project is a copy of this repository with the industrialization of the Abalone age prediction model. We expect to see:

1. a workflow to train a model using Prefect
- The workflows to train the model and to make the inference (prediction of the age of abalone) are in separate modules and use Prefect `flow` and `task` objects
- The code to get the trained model and encoder is in a separate module and must be reproducible (not necessarily in a docker container)
2. a Prefect deployment to retrain the model regularly
3. an API that runs on a local app and that allows users to make predictions on new data
  - A working API which can be used to make predictions on new data
    - The API can run on a docker container
    - The API has validation on input data (use Pydantic)

### Evaluation

Each of your pull requests will be graded based on the following criteria:

- **Clarity** and quality of code
  - good module structure
  - naming conventions
  - use of docstrings and type hinting
- **Formatting**
  - respect of clear code conventions

  *P.S. you can use a linter and automatic code formatters to help you with that*

- Proper **Functioning** of the code
  - the code must run without bugs

Bseides the evaluation of the pull requests, we will also evaluate:
- **Reproducibility** and clarity of instructions to run the code (we will actually try to run your code)
  - Having a clear README.md with
    - the context of the project
    - the name of the participants and their github users
    - the steps to recreate the Python environment
    - the instructions to run all parts of the code
- Use of *Pull Requests* (see below) to coordinate your collaboration

## Steps to reproduce to build the deliverable

To help you with the structure and order of steps to perform in this project, we created different pull requests templates.
Each branch in this repository corresponds to a future pull request and has an attached markdown file with the instructions to perform the tasks of the pull request.
Each branch starts with a number.
You can follow the order of the branches to build your project and collaborate.

> [!NOTE]
> There are "TODO" in the code of the different branches. Each "TODO" corresponds to a task to perform to build the project.
> [!IMPORTANT]
> Remember to remove all code that is not used before the end of the project (including all TODO tags in the code).

**Please follow these steps**:

- If not done already, create a GitHub account
- If not done already, create a [Kaggle account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) (so you can download the dataset)
- Fork this repository (one person per group)

**WARNING**: make sure to **unselect** the option "Copy the `master` branch only", so you have all the branches in the forked repository.

- Add the different members of your group as admin to your forked repository
- Follow the order of the numbered branches and for each branch:
  - Read the PR_i.md (where i is the number of the branch) file to understand the task to perform
   > [!NOTE]
   > Dont forget to integrate your work from past branches (except for when working on branch #1 obviously (!))
   > ```bash
   > git checkout branch_number_i
   > git pull origin master
   > # At this point, you might have a VIM window opening, you can close it using the command ":wq"
   > git push
   > ```
    - Read and **follow** all the instructions in the the PR instructions file
    - Do as many commits as necessary on the branch_number_i to perform the task indicated in the corresponding markdown file
    - Open **A SINGLE** pull request from this branch to the main branch of your forked repository
    - Once done, merge the pull request in the main branch of your forked repository

### Pull requests in this project

Github [Pull Requests](https://docs.github.com/articles/about-pull-requests) are a way to propose changes to a repository. They have for purpose to integrate the work of *feature branches* into the main branch of the repository, with a collaborative review process.

**PR tips:**

Make sure that you select your own repository when selecting the base repository:

![PR Wrong](assets/PR_wrong.png)

It should rather look like this:

![PR Right](assets/PR_right.png)

### Tips to work on this project

- Use a virtual environment to install the dependencies of the project (conda or virtualenv for instance)

- Once your virtual environment is activated, install pre-commit hooks to automatically format your code before each commit:

```bash
pip install pre-commit
pre-commit install
```

This will guarantee that your code is formatted correctly and of good quality before each commit.

- Use a `requirements.in` file to list the dependencies of your project. You can use the following command to generate a `requirements.txt` file from a `requirements.in` file:

```bash
pip-compile requirements.in
```

# Setup Instructions

## Environment Setup
This project can be set up with or without Conda. Choose the method that works best for your development environment.
### Option 1: Setting up with Conda
Follow these steps to create and activate a Conda environment:
1. **Create the environment** using the `environment.yml` file. This will install all dependencies (both runtime and development):

```bash
conda env create -f environment.yml
```

2. **Activate the environment**:
```bash
conda activate xhec-mlops-env
```
3. **Install Pre-commit hooks** (optional but recommended):
```bash
pre-commit install
```
The pre-commit hooks will run code formatting (black), import sorting (isort), and linting (ruff) automatically before every commit.

### Option 2: Setting up without Conda

If you prefer not to use Conda, you can set up the environment using `pip` with `requirements.txt` and `requirements-dev.txt`:

1. **Create a virtual environment**:
  ```bash
  python -m venv venv
  ```
2. **Activate the virtual environment**:
  - On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

  - On **Windows**:
  ```bash
  venv\Scripts\activate
  ```
3. **Upgrade `pip` and install the project dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```
4. **Install Pre-commit hooks** (optional but recommended):
```bash
pre-commit install
```
The pre-commit hooks will enforce code formatting and quality checks.

## Running Prefect Flows and Deployments
To manage and run your workflows with Prefect, follow these steps:
### Step 1: Sign Up for Prefect Cloud
If you don't already have a Prefect Cloud account, sign up at [Prefect Cloud](https://www.prefect.io/cloud/).

### Step 2: Create a Prefect Cloud Workspace
Once you're logged in to Prefect Cloud:
1. **Create a workspace** (if one doesn't already exist).
2. Note the workspace **API Key** and **Workspace URL**, as you'll need them to connect your local setup to Prefect Cloud.

### Step 3: Log into Prefect Cloud from CLI
You need to log in to Prefect Cloud from your local machine using the Prefect CLI. Run the following command:

```bash
prefect cloud login --key <YOUR_API_KEY>
```
Replace <YOUR_API_KEY> with your Prefect Cloud API key. This key is available in your Prefect Cloud account settings under API Keys.

### Step 4: Run the Deployment Script
Now, you can deploy your Prefect flows to Prefect Cloud. Use the same `deployment.py` script to register your deployment:

```bash
python ./src/modelling/deployment.py
```
This will deploy your flow to the Prefect Cloud workspace.

### Step 5: Trigger a Flow Run
You can trigger flow runs directly from the Prefect Cloud UI or via the CLI.
- Trigger via CLI:
```bash
prefect deployment run "Model Training Deployment"
```

- Trigger via Prefect Cloud UI: Go to your Prefect Cloud workspace and navigate to the Deployments tab. From there, you can trigger your flow manually by selecting the deployment and running it.

### Step 6: Monitor Flow Runs in Prefect Cloud
In Prefect Cloud, you can monitor the status of your flows, view logs, and inspect run results directly from the Prefect Cloud UI. Here’s how to do it:

1. Go to your Prefect Cloud workspace in your browser.
2. Navigate to the Flows or Flow Runs tab.
3. You can view logs, success/failure statuses, and execution details for each flow run.

You can also manage your deployments, schedules, and agents directly from the UI.


## Last Part
We were not able to finish the last part on time. We made a mistake in importing a function from another file, which apparently doesn't exist (utils.py) but it does obviously. If you understand why, we would be happy to learn the reason. Sorry for not finishing on time.
