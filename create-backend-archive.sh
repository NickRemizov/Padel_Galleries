#!/bin/bash

# Backend Deployment Archive Creator v3.2.7
# Creates galeries.zip for deployment to server

echo "Creating backend deployment archive v3.2.7..."

# Remove old archive if exists
rm -f galeries.zip

# Create temporary directory
mkdir -p /tmp/galeries-build

# Copy SETUP.sh to root
cp SETUP.sh /tmp/galeries-build/

# Copy entire python directory
cp -r python /tmp/galeries-build/

# Make scripts executable
chmod +x /tmp/galeries-build/SETUP.sh
chmod +x /tmp/galeries-build/python/start.sh

# Create archive
cd /tmp/galeries-build
zip -r galeries.zip . -x "*.pyc" -x "__pycache__/*" -x "*.log" -x ".env"

# Move archive to current directory
mv galeries.zip /tmp/tmp.v0.app.galeries/

# Cleanup
cd /tmp/tmp.v0.app.galeries
rm -rf /tmp/galeries-build

echo "✓ Archive created: galeries.zip"
echo "✓ Backend version: v3.2.7"
echo ""
echo "Upload to server and run:"
echo "  cd /home/nickr"
echo "  sudo rm -rf python"
echo "  unzip -o galeries.zip"
echo "  chmod +x SETUP.sh"
echo "  sudo ./SETUP.sh"
echo "  cd python && ./start.sh"
