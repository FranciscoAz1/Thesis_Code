# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Build the TypeScript code
RUN npm run build

# Runtime stage
FROM node:20-alpine

# Install ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates

WORKDIR /app

# Copy package files and install production dependencies
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy built application from builder stage
COPY --from=builder /app/dist ./dist

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S mcp -u 1001

# Create logs directory and set permissions
RUN mkdir -p logs && chown -R mcp:nodejs logs

USER mcp

# Expose port (if using HTTP transport)
EXPOSE 3000

# Run the server
CMD ["node", "dist/main.js"]