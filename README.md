# ER-Patient-Admit-Volume
Forecasting patients presenting to the emergency department by month, day of week, and block of hours, with increments of 7, 30, and 60 days out.

# Folders

# Getting Started 
## How to Clone Repo
 - In your terminal, 'cd' to your working directory
 - Run 'git clone https://github.com/alanier1214/ER-Patient-Admit-Volume.git'
 - Open the project in your IDE

## How to Run Program

### Selecting Python Version
Python version 3.14 is recommended, but earlier versions will suffice. 

In your IDE be sure to specify the version of Python you want your interpreter to use.
- Steps if using VS Code (recommended):
  - Open Command Palette:
    - 'Cmd' + 'Shift' + 'P' on Mac
    - 'Ctrl' + 'Shift' + 'P' on Windows
- Search for 'Python: Select Interpreter'
- Select Python version you are running
- Restart VS Code

### Running the Program

# How to Make Changes
Create a branch for making changes to the project.

- Checkout master branch
    - `git checkout master`
- Updating local to be up to date
    - `git pull`
- Make a new branch before making changes
    - `git checkout -b <your-branch-name>`
- Make your changes and update the README
- Stage your changes
    - `git add <file-name>` or `git add .` to stage all changes
- Commit changes with message to identify what was changed
    - `git commit -m "your message"`
- Rebase on master before pushing changes
    - Switch to master by `git switch master`
    - Pull recent changes on master by `git pull`
    - Switch back to your branch by `git switch <your-branch-name>`
    - `git rebase master`
    - `git commit -m <message-about-rebasing>`
- To create a new branch on remote and push your changes
    - `git push origin <your-branch-name>`
- Go to the [GitHub page for this repo]([https://github.com/alanier1214/ER-Patient-Admit-Volume]) and create a Pull Request (PR) to merge changes
    - Click `Compare & pull request`
    - Add a title, comments, etc. with the changes you made
    - Add the team members as reviewers from the pane on the right hand side
    - Click `Create pull request`
- After changes have been reviewed and merged
    - Checkout master branch `git switch master`
    - Pull the newest changes `git pull`
    - Follow the same process for further changes
 
# Reviewing Pull Request Changes

Follow these steps for all Pull Requests:

- Fetch remote branches
    - `git fetch`
- Checkout the branch the PR is for
    - `git checkout <branch-name>`
- Review
    - Review all the files changes
    - Validate the code changes
    - Test code on the data
    - Use GitHub to make comments and give feedback
        - Wait until changes are made by the developer and review again using the steps above 
- When code is satisfactory, approve the Pull Request and merge in the branch using `Rebase and merge` button hidden in the dropdown
