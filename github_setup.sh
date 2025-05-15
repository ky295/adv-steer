#!/bin/bash

# Prompt for GitHub credentials if not provided
read -p "Enter your GitHub email: " email
read -p "Enter your GitHub name: " name
read -p "Enter your GitHub token: " github_token

# Validate inputs
if [ -z "$email" ] || [ -z "$name" ] || [ -z "$github_token" ]; then
    echo "Error: Email, name, and token are required"
    exit 1
fi

# 0) Setup git
git config --global user.email "$email"
git config --global user.name "$name"

# 1) Setup GitHub token authentication
echo "Setting up GitHub token authentication..."
git config --global credential.helper store
echo "https://oauth2:${github_token}@github.com" > /root/.git-credentials
chmod 600 /root/.git-credentials