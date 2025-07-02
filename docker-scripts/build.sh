#!/bin/bash

# ASAG-AGENT-ARABIC Docker Build Script
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🐳 ASAG-AGENT-ARABIC Docker Build Script${NC}"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from template...${NC}"
    cp .env.docker .env
    echo -e "${RED}❗ Please edit .env file with your actual API keys before running!${NC}"
    exit 1
fi

# Build production image
echo -e "${YELLOW}🔨 Building production Docker image...${NC}"
docker build -t asag-arabic:latest .

# Build development image
echo -e "${YELLOW}🔨 Building development Docker image...${NC}"
docker build -f Dockerfile.dev -t asag-arabic:dev .

echo -e "${GREEN}✅ Docker images built successfully!${NC}"

# Show built images
echo -e "${YELLOW}📋 Built images:${NC}"
docker images | grep asag-arabic

echo -e "${GREEN}🚀 Ready to run! Use:${NC}"
echo -e "  Production: ${YELLOW}docker-compose up -d${NC}"
echo -e "  Development: ${YELLOW}docker-compose -f docker-compose.dev.yml up -d${NC}"