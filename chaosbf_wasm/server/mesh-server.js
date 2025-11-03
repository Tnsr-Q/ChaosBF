// ChaosBF v5.0 - Mesh Server (Node.js WebSocket)
// TODO: Coordinate distributed island migration
// Feature flag: ENABLE_MESH

// const WebSocket = require('ws');

// Max message size to prevent DoS attacks (1MB)
const MAX_MESSAGE_SIZE = 1_000_000;
// WebSocket ready state constants (WebSocket module not imported yet)
const WEBSOCKET_OPEN = 1;

class MeshServer {
    constructor(port = 8080) {
        this.port = port;
        this.wss = null;
        this.clients = new Map();
        this.globalPool = [];  // Global elite pool
        this.stats = { total_clients: 0, total_individuals: 0 };
    }

    start() {
        // TODO: Initialize WebSocket server
        // this.wss = new WebSocket.Server({ port: this.port });
        // this.wss.on('connection', (ws, req) => { this.handleConnection(ws, req); });

        console.log(`MeshServer listening on port ${this.port} (TODO)`);
    }

    handleConnection(ws, req) {
        const clientId = this.generateClientId();
        this.clients.set(clientId, { ws, island_id: clientId, connected_at: Date.now() });
        this.stats.total_clients++;

        console.log(`Client ${clientId} connected`);

        ws.on('message', (data) => {
            try {
                // Handle Buffer or string payloads - convert Buffer to UTF-8 string
                let msgStr;
                if (Buffer.isBuffer(data)) {
                    // Check size before converting
                    if (data.length > MAX_MESSAGE_SIZE) {
                        console.warn(`Message from ${clientId} exceeds size limit (${data.length} bytes)`);
                        return;
                    }
                    msgStr = data.toString('utf8');
                } else if (typeof data === 'string') {
                    // Check size of string
                    if (data.length > MAX_MESSAGE_SIZE) {
                        console.warn(`Message from ${clientId} exceeds size limit (${data.length} chars)`);
                        return;
                    }
                    msgStr = data;
                } else {
                    console.warn(`Invalid message type from ${clientId}`);
                    return;
                }

                // Parse JSON after size check
                const msg = JSON.parse(msgStr);
                this.handleMessage(clientId, msg);
            } catch (err) {
                console.error(`Failed to parse message from ${clientId}:`, err.message);
            }
        });

        ws.on('close', () => {
            this.clients.delete(clientId);
            this.stats.total_clients--;
            console.log(`Client ${clientId} disconnected`);
        });

        // Send welcome message
        ws.send(JSON.stringify({ type: 'welcome', client_id: clientId }));
    }

    handleMessage(clientId, msg) {
        switch (msg.type) {
            case 'individual':
                // Add to global pool
                this.globalPool.push({
                    genome: msg.genome,
                    fitness: msg.fitness,
                    descriptors: msg.descriptors,
                    source: clientId,
                    timestamp: Date.now()
                });
                this.stats.total_individuals++;

                // Trim pool if too large (keep best N)
                if (this.globalPool.length > 1000) {
                    this.globalPool.sort((a, b) => b.fitness - a.fitness);
                    this.globalPool = this.globalPool.slice(0, 500);
                }
                break;

            case 'request_migrant': {
                // Wrap in braces to scope const/let declarations
                // Send random individual from pool (excluding sender's own)
                const candidates = this.globalPool.filter(ind => ind.source !== clientId);
                if (candidates.length > 0) {
                    const migrant = candidates[Math.floor(Math.random() * candidates.length)];
                    const client = this.clients.get(clientId);
                    if (client && client.ws.readyState === WEBSOCKET_OPEN) {
                        client.ws.send(JSON.stringify({
                            type: 'migrant',
                            genome: migrant.genome,
                            fitness: migrant.fitness,
                            descriptors: migrant.descriptors
                        }));
                    }
                }
                break;
            }

            default:
                console.warn(`Unknown message type from ${clientId}:`, msg.type);
        }
    }

    generateClientId() {
        return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    broadcastStats() {
        const statsMsg = JSON.stringify({
            type: 'stats',
            ...this.stats,
            pool_size: this.globalPool.length
        });

        this.clients.forEach(client => {
            if (client.ws.readyState === WEBSOCKET_OPEN) {
                client.ws.send(statsMsg);
            }
        });
    }
}

// TODO: Uncomment when implementing
// const server = new MeshServer(8080);
// server.start();
// setInterval(() => server.broadcastStats(), 5000);

module.exports = { MeshServer };
