/**
 * SwarmP2P — Shared P2P module for all gpubench demos.
 * Handles: WebRTC mesh, signaling, room overflow, custom rooms, elite exchange.
 *
 * Usage:
 *   const p2p = new SwarmP2P({
 *     defaultRoom: 'flappy-demo',
 *     genomeSize: 22,
 *     signalingUrl: SwarmP2P.DEFAULT_RELAY,  // or custom URL
 *     onEliteReceived: (genome, fitness, from) => { ... },
 *     onStatusChange: (status) => { ... },
 *     onLog: (msg, cls) => { ... },
 *   });
 *   p2p.connect();                       // join default room
 *   p2p.connect('my-custom-room');        // join custom room
 *   p2p.broadcastElite(genome, fitness);  // send best to all peers
 *   p2p.destroy();                        // cleanup
 */

class SwarmP2P {
    static MAX_PEERS = 20;
    static RATE_LIMIT_MS = 500;
    static DEFAULT_RELAY = 'wss://swarm-engine-production-5b98.up.railway.app';

    constructor(opts) {
        this.genomeSize = opts.genomeSize || 22;
        this.defaultRoom = opts.defaultRoom || 'swarm-demo';
        this.signalingUrl = opts.signalingUrl || SwarmP2P.DEFAULT_RELAY;
        this.onEliteReceived = opts.onEliteReceived || (() => {});
        this.onStatusChange = opts.onStatusChange || (() => {});
        this.onLog = opts.onLog || ((msg) => console.log(`[SwarmP2P] ${msg}`));

        this.nodeId = Math.random().toString(36).slice(2, 8);
        this.peers = new Map();          // targetId -> {pc, dc, lastEliteTime}
        this.candidateQueue = new Map();  // targetId -> [candidates]
        this.ws = null;
        this.isConnected = false;
        this.lastBroadcastTime = 0;
        this.localBestScore = -Infinity;
        this.currentRoom = null;
        this._roomSuffix = 0;
        this._sentCount = 0;
        this._recvCount = 0;
    }

    // ─── Public API ───

    connect(room) {
        this.currentRoom = room || this._getRoomFromURL() || this.defaultRoom;
        this._connectWS();
    }

    broadcastElite(genome, fitness) {
        this.localBestScore = Math.max(this.localBestScore, fitness);
        const now = Date.now();
        if (now - this.lastBroadcastTime < SwarmP2P.RATE_LIMIT_MS) return;
        this.lastBroadcastTime = now;

        // Binary: Float32[0]=fitness, Float32[1..N]=genome
        const buf = new Float32Array(1 + this.genomeSize);
        buf[0] = fitness;
        buf.set(genome, 1);

        let sent = 0;
        for (const [, p] of this.peers) {
            if (p.dc?.readyState === 'open') { p.dc.send(buf.buffer); sent++; }
        }
        // Relay fallback for peers without direct WebRTC
        if (sent === 0 && this.ws?.readyState === 1) {
            this.ws.send(JSON.stringify({
                type: 'elite', fitness,
                genome: Array.from(genome),
                fromNode: this.nodeId
            }));
            sent = 1;
        }
        if (sent > 0) {
            this._sentCount++;
            this.onLog(`Broadcast elite (score: ${Math.floor(fitness)}) to ${sent} peer(s)`, 'p2p');
            this._emitStatus();
        }
    }

    destroy() {
        for (const [, p] of this.peers) { p.dc?.close(); p.pc?.close(); }
        this.peers.clear();
        this.ws?.close();
    }

    get status() {
        const openPeers = [...this.peers.values()].filter(p => p.dc?.readyState === 'open').length;
        return {
            nodeId: this.nodeId,
            room: this.currentRoom,
            peers: openPeers,
            connected: openPeers > 0,
            sent: this._sentCount,
            received: this._recvCount,
            signalingConnected: this.ws?.readyState === 1,
        };
    }

    // ─── URL params ───

    _getRoomFromURL() {
        try {
            const params = new URLSearchParams(window.location.search);
            return params.get('room') || null;
        } catch { return null; }
    }

    _getRelayFromURL() {
        try {
            const params = new URLSearchParams(window.location.search);
            return params.get('sig') || null;
        } catch { return null; }
    }

    // ─── WebSocket signaling ───

    _connectWS() {
        const url = this._getRelayFromURL() || this.signalingUrl;
        this.onLog(`Node ${this.nodeId} connecting to relay...`, 'p2p');

        try {
            this.ws = new WebSocket(url);
        } catch (e) {
            this.onLog('Failed to connect to relay: ' + e.message, 'p2p');
            return;
        }

        this.ws.onopen = () => {
            this.onLog(`Joining room: ${this.currentRoom}`, 'p2p');
            this.ws.send(JSON.stringify({ type: 'join', room: this.currentRoom }));
        };

        this.ws.onclose = () => {
            this.onLog('Signaling disconnected — reconnecting in 3s', 'p2p');
            this._emitStatus();
            setTimeout(() => this._connectWS(), 3000);
        };

        this.ws.onerror = () => {
            this.onLog('Relay connection error', 'p2p');
        };

        this.ws.onmessage = async (e) => {
            let msg;
            try { msg = JSON.parse(e.data); } catch { return; }

            // ─── Room full → auto-overflow ───
            if (msg.type === 'error' && msg.msg === 'Room full') {
                this._roomSuffix++;
                this.currentRoom = this.defaultRoom + '-' + this._roomSuffix;
                this.onLog(`Room full — joining overflow: ${this.currentRoom}`, 'p2p');
                this.ws.send(JSON.stringify({ type: 'join', room: this.currentRoom }));
                this._emitStatus();
                return;
            }

            // ─── Joined room ───
            if (msg.type === 'joined') {
                this.currentRoom = msg.room || this.currentRoom;
                this.onLog(`Joined room "${this.currentRoom}" (${msg.peers} peers)`, 'p2p');
                // Hello handshake: announce ourselves so existing peers connect
                this.ws.send(JSON.stringify({
                    type: 'candidate', isHello: true,
                    from: this.nodeId, room: this.currentRoom
                }));
                this._emitStatus();
            }

            if (msg.type === 'peer_joined') {
                this.onLog('New peer joining...', 'p2p');
            }

            if (msg.type === 'peer_left') {
                this.onLog('Peer left', 'p2p');
                this._emitStatus();
            }

            // ─── Hello from another peer → create targeted offer ───
            if (msg.type === 'candidate' && msg.isHello) {
                if (msg.from !== this.nodeId) await this._createOffer(msg.from);
                return;
            }

            // ─── Relay elite (WebSocket fallback) ───
            if (msg.type === 'elite' || msg.type === 'elite_genome') {
                if (msg.fromNode === this.nodeId) return; // ignore own echo
                const genome = Array.isArray(msg.genome) ? new Float32Array(msg.genome) : null;
                if (genome) this._handleElite(genome, msg.fitness || 0, 'relay');
                return;
            }

            // ─── Targeted signaling (only process if addressed to us) ───
            if (msg.to && msg.to !== this.nodeId) return;

            if (msg.type === 'offer') await this._handleOffer(msg.offer, msg.from);
            if (msg.type === 'answer') await this._handleAnswer(msg.answer, msg.from);
            if (msg.type === 'candidate' && !msg.isHello) this._addCandidate(msg.candidate, msg.from);
        };
    }

    // ─── WebRTC ───

    async _createOffer(targetId) {
        if (this.peers.has(targetId) || this.peers.size >= SwarmP2P.MAX_PEERS) return;
        const pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
        const dc = pc.createDataChannel('genome', { ordered: false, maxRetransmits: 0 });
        this.peers.set(targetId, { pc, dc, lastEliteTime: 0 });
        this._setupDC(dc, targetId);
        this._setupPC(pc, targetId);
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        this.ws.send(JSON.stringify({ type: 'offer', offer: pc.localDescription, from: this.nodeId, to: targetId }));
    }

    async _handleOffer(offer, targetId) {
        if (this.peers.has(targetId) || this.peers.size >= SwarmP2P.MAX_PEERS) return;
        const pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
        this.peers.set(targetId, { pc, dc: null, lastEliteTime: 0 });
        this._setupPC(pc, targetId);
        pc.ondatachannel = (e) => {
            this.peers.get(targetId).dc = e.channel;
            this._setupDC(e.channel, targetId);
        };
        await pc.setRemoteDescription(new RTCSessionDescription(offer));
        this._flushCandidates(targetId);
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        this.ws.send(JSON.stringify({ type: 'answer', answer: pc.localDescription, from: this.nodeId, to: targetId }));
    }

    async _handleAnswer(answer, targetId) {
        const p = this.peers.get(targetId);
        if (p && p.pc.signalingState === 'have-local-offer') {
            await p.pc.setRemoteDescription(new RTCSessionDescription(answer));
            this._flushCandidates(targetId);
        }
    }

    _addCandidate(c, targetId) {
        const p = this.peers.get(targetId);
        if (p && p.pc.remoteDescription) {
            p.pc.addIceCandidate(new RTCIceCandidate(c));
        } else {
            if (!this.candidateQueue.has(targetId)) this.candidateQueue.set(targetId, []);
            this.candidateQueue.get(targetId).push(c);
        }
    }

    _flushCandidates(targetId) {
        const p = this.peers.get(targetId);
        const queue = this.candidateQueue.get(targetId) || [];
        for (const c of queue) {
            if (p?.pc.remoteDescription) p.pc.addIceCandidate(new RTCIceCandidate(c));
        }
        this.candidateQueue.delete(targetId);
    }

    _setupPC(pc, targetId) {
        pc.onicecandidate = (e) => {
            if (e.candidate) {
                this.ws.send(JSON.stringify({
                    type: 'candidate', candidate: e.candidate,
                    from: this.nodeId, to: targetId
                }));
            }
        };
        pc.onconnectionstatechange = () => {
            if (pc.connectionState === 'connected') {
                this.onLog(`WebRTC connected to ${targetId}`, 'p2p');
                this._emitStatus();
            }
            if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
                this.peers.delete(targetId);
                this._emitStatus();
            }
        };
    }

    _setupDC(dc, targetId) {
        dc.binaryType = 'arraybuffer';
        dc.onopen = () => {
            this.onLog(`Data channel open with ${targetId}`, 'p2p');
            this._emitStatus();
        };
        dc.onclose = () => this._emitStatus();
        dc.onmessage = (e) => {
            if (!(e.data instanceof ArrayBuffer)) return;
            const view = new Float32Array(e.data);
            if (view.length !== this.genomeSize + 1) return;
            const fitness = view[0];
            if (!isFinite(fitness)) return;
            for (let i = 1; i <= this.genomeSize; i++) { if (!isFinite(view[i])) return; }
            // Per-peer rate limiting
            const peer = this.peers.get(targetId);
            if (peer) {
                const now = Date.now();
                if (peer.lastEliteTime && (now - peer.lastEliteTime) < SwarmP2P.RATE_LIMIT_MS) return;
                peer.lastEliteTime = now;
            }
            this._handleElite(view.slice(1), fitness, targetId);
        };
    }

    // ─── Elite handling ───

    _handleElite(genome, fitness, from) {
        // Always accept — different islands evolve on different landscapes,
        // scores aren't comparable. Every foreign genome adds diversity.
        this._recvCount++;
        this.onLog(`Received elite (score: ${Math.floor(fitness)}) from ${from}`, 'p2p');
        this.onEliteReceived(genome, fitness, from);
        this._emitStatus();
    }

    _emitStatus() {
        this.onStatusChange(this.status);
    }
}

// Export for both module and script tag usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SwarmP2P;
} else if (typeof window !== 'undefined') {
    window.SwarmP2P = SwarmP2P;
}
