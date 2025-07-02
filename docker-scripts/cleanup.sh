#!/bin/bash

# ASAG-AGENT-ARABIC Docker Cleanup Script
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🧹 ASAG-AGENT-ARABIC Docker Cleanup Script${NC}"

# Function to confirm action
confirm() {
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Operation cancelled${NC}"
        exit 1
    fi
}

# Stop all containers
echo -e "${YELLOW}🛑 Stopping all containers...${NC}"
docker-compose down
docker-compose -f docker-compose.dev.yml down 2>/dev/null || true

# Remove containers
echo -e "${YELLOW}🗑️  Removing containers...${NC}"
docker-compose rm -f
docker-compose -f docker-compose.dev.yml rm -f 2>/dev/null || true

# Remove images
echo -e "${RED}⚠️  This will remove ASAG Docker images${NC}"
confirm
docker rmi asag-arabic:latest asag-arabic:dev 2>/dev/null || true

# Remove volumes (optional)
echo -e "${RED}⚠️  This will remove ALL data (databases, cache, etc.)${NC}"
confirm
docker volume rm $(docker volume ls -q | grep asag) 2>/dev/null || true

# Remove networks
docker network rm asag-network asag-dev-network 2>/dev/null || true

# Clean up unused Docker resources
echo -e "${YELLOW}🧽 Cleaning up unused Docker resources...${NC}"
docker system prune -f

echo -e "${GREEN}✅ Cleanup completed!${NC}"