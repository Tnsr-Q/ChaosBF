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
        // Create safe error message without innerHTML
        const listEl = document.getElementById('snapshot-list');
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = 'color: #ff0000; padding: 20px; border: 1px solid #ff0000;';
        
        const errorMsg = document.createElement('div');
        errorMsg.textContent = `Error: ${err.message}`;
        errorDiv.appendChild(errorMsg);
        
        const errorDetail = document.createElement('small');
        errorDetail.textContent = 'Gallery feature requires IndexedDB support';
        errorDiv.appendChild(document.createElement('br'));
        errorDiv.appendChild(errorDetail);
        
        listEl.replaceChildren();
        listEl.appendChild(errorDiv);
    }
}

async function loadSnapshots() {
    const snapshots = await gallery.getAllSnapshots();
    const listEl = document.getElementById('snapshot-list');

    if (snapshots.length === 0) {
        // Clear and add no-data message using safe DOM API
        listEl.replaceChildren();
        const noDataMsg = document.createElement('p');
        noDataMsg.style.gridColumn = '1/-1';
        noDataMsg.textContent = 'No snapshots yet. Run simulations to create snapshots.';
        listEl.appendChild(noDataMsg);
        return;
    }

    // Clear existing content using replaceChildren
    listEl.replaceChildren();
    snapshots.reverse().forEach(snapshot => {
        const card = createSnapshotCard(snapshot);
        listEl.appendChild(card);
    });
}

function createSnapshotCard(snapshot) {
    const card = document.createElement('div');
    card.className = 'snapshot-card';
    // Attach click handler with addEventListener instead of inline
    card.addEventListener('click', () => viewSnapshot(snapshot));

    const date = new Date(snapshot.timestamp);
    const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();

    // Create elements safely without innerHTML for untrusted data
    const title = document.createElement('h3');
    title.textContent = snapshot.run_id || 'Untitled Run';
    card.appendChild(title);

    const meta = document.createElement('div');
    meta.className = 'meta';
    
    const dateDiv = document.createElement('div');
    dateDiv.textContent = `Date: ${dateStr}`;
    meta.appendChild(dateDiv);
    
    const stepDiv = document.createElement('div');
    stepDiv.textContent = `Step: ${Number(snapshot.step || 0)}`;
    meta.appendChild(stepDiv);
    
    card.appendChild(meta);

    const metrics = document.createElement('div');
    metrics.className = 'metrics';
    // Coerce to numbers before toFixed
    metrics.textContent = `E=${Number(snapshot.e || 0).toFixed(1)} ` +
                          `T=${Number(snapshot.t || 0).toFixed(2)} ` +
                          `S=${Number(snapshot.s || 0).toFixed(2)} ` +
                          `λ=${Number(snapshot.lambda_estimate || 0).toFixed(2)}`;
    card.appendChild(metrics);

    return card;
}

function viewSnapshot(snapshot) {
    currentSnapshot = snapshot;
    const viewer = document.getElementById('viewer');
    const details = document.getElementById('viewer-details');

    const date = new Date(snapshot.timestamp);
    const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();

    // Build details panel safely without innerHTML for untrusted data
    // Clear existing content using replaceChildren
    details.replaceChildren();
    
    const title = document.createElement('h2');
    title.textContent = snapshot.run_id || 'Untitled Run';
    details.appendChild(title);

    const detailsDiv = document.createElement('div');
    detailsDiv.style.cssText = 'margin-top: 15px; line-height: 1.8;';
    
    const fields = [
        ['Date', dateStr],
        ['Step', Number(snapshot.step || 0)],
        ['Energy', Number(snapshot.e || 0).toFixed(2)],
        ['Temperature', Number(snapshot.t || 0).toFixed(3)],
        ['Entropy', Number(snapshot.s || 0).toFixed(3)],
        ['Free Energy', Number(snapshot.f || 0).toFixed(2)],
        ['Lambda', Number(snapshot.lambda_estimate || 0).toFixed(3)]
    ];
    
    fields.forEach(([label, value]) => {
        const fieldDiv = document.createElement('div');
        const strong = document.createElement('strong');
        strong.textContent = `${label}: `;
        fieldDiv.appendChild(strong);
        fieldDiv.appendChild(document.createTextNode(value));
        detailsDiv.appendChild(fieldDiv);
    });
    
    details.appendChild(detailsDiv);

    const codeSection = document.createElement('div');
    codeSection.style.marginTop = '15px';
    const codeLabel = document.createElement('strong');
    codeLabel.textContent = 'Code:';
    codeSection.appendChild(codeLabel);
    
    const codePre = document.createElement('pre');
    codePre.style.cssText = 'background: #000; padding: 10px; margin-top: 5px; overflow-x: auto; border: 1px solid #003300;';
    codePre.textContent = snapshot.code_hash || 'N/A';
    codeSection.appendChild(codePre);
    details.appendChild(codeSection);

    const buttonSection = document.createElement('div');
    buttonSection.style.marginTop = '15px';
    
    const replayBtn = document.createElement('button');
    replayBtn.textContent = '▶ Replay';
    replayBtn.style.cssText = 'padding: 8px 16px; background: #003300; border: 1px solid #00ff00; color: #00ff00; cursor: pointer; font-family: inherit;';
    replayBtn.addEventListener('click', () => replaySnapshot(snapshot.id));
    buttonSection.appendChild(replayBtn);
    
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '🗑 Delete';
    deleteBtn.style.cssText = 'padding: 8px 16px; background: #330000; border: 1px solid #ff0000; color: #ff0000; cursor: pointer; font-family: inherit; margin-left: 10px;';
    deleteBtn.addEventListener('click', () => deleteSnapshotUI(snapshot.id));
    buttonSection.appendChild(deleteBtn);
    
    details.appendChild(buttonSection);

    viewer.classList.add('active');
    // Move focus to close button for accessibility
    const closeBtn = document.getElementById('viewer-close');
    if (closeBtn) {
        closeBtn.focus();
    }
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

// Add keyboard handler for Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        const viewer = document.getElementById('viewer');
        if (viewer.classList.contains('active')) {
            closeViewer();
        }
    }
});

// Add click handler for close button
document.addEventListener('DOMContentLoaded', () => {
    const closeBtn = document.getElementById('viewer-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', closeViewer);
    }
});

// Initialize on load
init();
