// ChaosBF v5.0 - Gallery for Snapshot Browsing
// TODO: Implement IndexedDB storage and replay
// Feature flag: ENABLE_GALLERY

const DB_NAME = 'chaosbf_snapshots';
const DB_VERSION = 1;
const STORE_NAME = 'snapshots';

class GalleryDB {
    constructor() {
        this.db = null;
    }

    async open() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(DB_NAME, DB_VERSION);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve();
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains(STORE_NAME)) {
                    const store = db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
                    store.createIndex('timestamp', 'timestamp', { unique: false });
                    store.createIndex('run_id', 'run_id', { unique: false });
                }
            };
        });
    }

    async saveSnapshot(snapshot) {
        // TODO: Save snapshot to IndexedDB
        // snapshot: { run_id, timestamp, step, e, t, s, f, lambda, code, tape, ... }
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([STORE_NAME], 'readwrite');
            const store = transaction.objectStore(STORE_NAME);
            const request = store.add(snapshot);

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getAllSnapshots() {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([STORE_NAME], 'readonly');
            const store = transaction.objectStore(STORE_NAME);
            const request = store.getAll();

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getSnapshot(id) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([STORE_NAME], 'readonly');
            const store = transaction.objectStore(STORE_NAME);
            const request = store.get(id);

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async deleteSnapshot(id) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([STORE_NAME], 'readwrite');
            const store = transaction.objectStore(STORE_NAME);
            const request = store.delete(id);

            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }
}

// Gallery UI
const gallery = new GalleryDB();
let currentSnapshot = null;

async function init() {
    try {
        await gallery.open();
        console.log('Gallery DB opened');
        await loadSnapshots();
    } catch (err) {
        console.error('Failed to open gallery DB:', err);
        document.getElementById('snapshot-list').innerHTML = `
            <div style="color: #ff0000; padding: 20px; border: 1px solid #ff0000;">
                Error: ${err.message}<br>
                <small>Gallery feature requires IndexedDB support</small>
            </div>
        `;
    }
}

async function loadSnapshots() {
    const snapshots = await gallery.getAllSnapshots();
    const listEl = document.getElementById('snapshot-list');

    if (snapshots.length === 0) {
        listEl.innerHTML = '<p style="grid-column: 1/-1;">No snapshots yet. Run simulations to create snapshots.</p>';
        return;
    }

    listEl.innerHTML = '';
    snapshots.reverse().forEach(snapshot => {
        const card = createSnapshotCard(snapshot);
        listEl.appendChild(card);
    });
}

function createSnapshotCard(snapshot) {
    const card = document.createElement('div');
    card.className = 'snapshot-card';
    card.onclick = () => viewSnapshot(snapshot);

    const date = new Date(snapshot.timestamp);
    const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();

    card.innerHTML = `
        <h3>${snapshot.run_id || 'Untitled Run'}</h3>
        <div class="meta">
            <div>Date: ${dateStr}</div>
            <div>Step: ${snapshot.step || 0}</div>
        </div>
        <div class="metrics">
            E=${(snapshot.e || 0).toFixed(1)}
            T=${(snapshot.t || 0).toFixed(2)}
            S=${(snapshot.s || 0).toFixed(2)}
            λ=${(snapshot.lambda_estimate || 0).toFixed(2)}
        </div>
    `;

    return card;
}

function viewSnapshot(snapshot) {
    currentSnapshot = snapshot;
    const viewer = document.getElementById('viewer');
    const details = document.getElementById('viewer-details');

    const date = new Date(snapshot.timestamp);
    const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();

    details.innerHTML = `
        <h2>${snapshot.run_id || 'Untitled Run'}</h2>
        <div style="margin-top: 15px; line-height: 1.8;">
            <div><strong>Date:</strong> ${dateStr}</div>
            <div><strong>Step:</strong> ${snapshot.step || 0}</div>
            <div><strong>Energy:</strong> ${(snapshot.e || 0).toFixed(2)}</div>
            <div><strong>Temperature:</strong> ${(snapshot.t || 0).toFixed(3)}</div>
            <div><strong>Entropy:</strong> ${(snapshot.s || 0).toFixed(3)}</div>
            <div><strong>Free Energy:</strong> ${(snapshot.f || 0).toFixed(2)}</div>
            <div><strong>Lambda:</strong> ${(snapshot.lambda_estimate || 0).toFixed(3)}</div>
        </div>
        <div style="margin-top: 15px;">
            <strong>Code:</strong>
            <pre style="background: #000; padding: 10px; margin-top: 5px; overflow-x: auto; border: 1px solid #003300;">
${snapshot.code_hash || 'N/A'}
            </pre>
        </div>
        <div style="margin-top: 15px;">
            <button onclick="replaySnapshot(${snapshot.id})" style="padding: 8px 16px; background: #003300; border: 1px solid #00ff00; color: #00ff00; cursor: pointer; font-family: inherit;">
                ▶ Replay
            </button>
            <button onclick="deleteSnapshotUI(${snapshot.id})" style="padding: 8px 16px; background: #330000; border: 1px solid #ff0000; color: #ff0000; cursor: pointer; font-family: inherit; margin-left: 10px;">
                🗑 Delete
            </button>
        </div>
    `;

    viewer.classList.add('active');
}

window.closeViewer = function() {
    document.getElementById('viewer').classList.remove('active');
    currentSnapshot = null;
};

window.replaySnapshot = function(id) {
    // TODO: Load snapshot and redirect to main UI with restore
    alert('Replay not implemented yet. Will load snapshot and restore state.');
};

window.deleteSnapshotUI = async function(id) {
    if (confirm('Delete this snapshot?')) {
        await gallery.deleteSnapshot(id);
        closeViewer();
        await loadSnapshots();
    }
};

// Initialize on load
init();
