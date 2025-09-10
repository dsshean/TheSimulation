# TheSimulation Docker Container
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    redis-server \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for the dashboard
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/states \
    /app/data/life_summaries \
    /app/data/narrative_images \
    /app/logs/events \
    /var/log/supervisor \
    && chmod -R 755 /app/data \
    && chmod -R 755 /app/logs

# Back to app root  
WORKDIR /app

# Copy supervisor configuration
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy Redis configuration
COPY docker/redis.conf /etc/redis/redis.conf

# Copy startup script
COPY docker/startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

# Expose ports
EXPOSE 6379 8765 8766 1420

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='localhost', port=6379); r.ping()" || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_API_KEY=""
ENV ENABLE_WEB_VISUALIZATION=true
ENV REDIS_HOST=localhost
ENV REDIS_PORT=6379

# Volume for persistent data
VOLUME ["/app/data", "/app/logs"]

# Start supervisor to manage all services
CMD ["/usr/local/bin/startup.sh"]