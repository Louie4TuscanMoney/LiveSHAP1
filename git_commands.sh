#!/bin/bash
# Full git commands to commit and push to GitHub

cd /Users/embrace/Desktop/SHAP

echo "Initializing git repository..."
git init

echo "Adding remote repository..."
git remote add origin https://github.com/Louie4TuscanMoney/LiveSHAP1.git 2>/dev/null || echo "Remote already exists"

echo "Staging all files..."
git add .

echo "Committing changes..."
git commit -m "Initial commit: NBA Live Prediction API with Railway deployment"

echo "Setting main branch..."
git branch -M main

echo "Pushing to GitHub..."
git push -u origin main

echo "Done! Your code is now on GitHub."
