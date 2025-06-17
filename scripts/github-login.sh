#!/bin/bash

# GitHub Container Registry Login Script
# Erstellen Sie einen Personal Access Token mit 'read:packages' Berechtigung

echo "🔐 GitHub Container Registry Login"
echo ""
echo "1. Gehen Sie zu: https://github.com/settings/tokens"
echo "2. 'Generate new token (classic)'"
echo "3. Wählen Sie Scopes: 'read:packages', 'write:packages'"
echo "4. Kopieren Sie den Token"
echo ""

read -p "GitHub Username: " GITHUB_USERNAME
read -s -p "Personal Access Token: " GITHUB_TOKEN
echo ""

echo "🔄 Logging in to GitHub Container Registry..."
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin

if [ $? -eq 0 ]; then
    echo "✅ Login erfolgreich!"
    echo ""
    echo "Jetzt können Sie private Images pullen:"
    echo "docker pull ghcr.io/$GITHUB_USERNAME/scribe-engine:latest"
else
    echo "❌ Login fehlgeschlagen!"
    echo "Prüfen Sie Username und Token."
fi 