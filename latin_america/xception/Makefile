# Makefile for managing the Docker Compose environment

# --- Configuration ---
# Use the current directory name as the project name for Docker Compose.
PROJECT_NAME := $(shell basename $(CURDIR))

# Define the main service name for commands like 'shell' and 'logs'.
# Change 'app' to your main service's name in docker-compose.yml.
APP_SERVICE_NAME := app

# --- Commands ---

# .PHONY ensures that 'make' will execute the command instead of looking for a file with the same name.
.PHONY: help up down start stop restart build rebuild logs ps shell clean

# If 'make' is run without a target, it will execute the first one in the file. We set 'help' as the default.
.DEFAULT_GOAL := help

help: ## Display this help message with available commands
	@echo "Usage: make [command]"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

up: ## Start containers in detached mode (background)
	@echo "🚀 Starting up containers..."
	@docker-compose up -d

down: ## Stop and remove containers, networks, and anonymous volumes
	@echo "🛑 Stopping and removing containers..."
	@docker-compose down

start: ## Start containers that have already been created
	@echo "▶️  Starting containers..."
	@docker-compose start

stop: ## Stop containers without removing them
	@echo "⏹️  Stopping containers..."
	@docker-compose stop

restart: ## Restart all services
	@echo "🔄 Restarting containers..."
	@docker-compose restart

build: ## Build or rebuild the images for the services
	@echo "🛠️  Building images..."
	@docker-compose build

rebuild: ## Force the image rebuild without using cache and start the containers
	@echo "💥 Forcing image rebuild and starting services..."
	@docker-compose build --no-cache
	@make up

logs: ## Display the logs from all services in real-time
	@echo "📜 Tailing logs..."
	@docker-compose logs -f

ps: ## List the running containers for the project
	@echo "📊 Container status:"
	@docker-compose ps

shell: ## Access the terminal (bash) of the main container
	@echo "💻 Accessing the terminal of the '$(APP_SERVICE_NAME)' service..."
	@docker-compose exec $(APP_SERVICE_NAME) bash

clean: ## Remove all unused containers, networks, volumes, and images (USE WITH CAUTION!)
	@echo "🧹 Cleaning up the Docker environment..."
	@docker-compose down -v --remove-orphans
	@docker system prune -af
	@docker volume prune -f