#!/bin/bash

# ASAG-AGENT-ARABIC Docker Deployment Script
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 ASAG-AGENT-ARABIC Deployment Script${NC}"

# Function to check if service is healthy
check_service_health() {
    local service_name=$1
    local max_attempts=30
    local attempt=1
    
    echo -e "${YELLOW}🔍 Checking $service_name health...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps $service_name | grep -q "healthy\|Up"; then
            echo -e "${GREEN}✅ $service_name is healthy${NC}"
            return 0
        fi
        
        echo -e "${YELLOW}   Attempt $attempt/$max_attempts - $service_name not ready yet...${NC}"
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}❌ $service_name failed to become healthy${NC}"
    return 1
}

# Stop existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker-compose down

# Pull latest images
echo -e "${YELLOW}📥 Pulling latest images...${NC}"
docker-compose pull

# Start services
echo -e "${YELLOW}🚀 Starting services...${NC}"
docker-compose up -d

# Check service health
check_service_health "mongo"
check_service_health "chromadb"
check_service_health "asag-app"

# Show running containers
echo -e "${GREEN}📋 Running containers:${NC}"
docker-compose ps

# Show logs
echo -e "${YELLOW}📝 Recent logs:${NC}"
docker-compose logs --tail=20

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "Application available at: ${YELLOW}http://localhost${NC}"