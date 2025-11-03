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
        const listEl = document.getElementById('snapshot-list');
        listEl.textContent = ''; // Clear existing content
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = 'color: #ff0000; padding: 20px; border: 1px solid #ff0000;';
        const errorText = document.createTextNode(`Error: ${err.message}`);
        const errorSmall = document.createElement('small');
        errorSmall.textContent = 'Gallery feature requires IndexedDB support';
        errorDiv.appendChild(errorText);
        errorDiv.appendChild(document.createElement('br'));
        errorDiv.appendChild(errorSmall);
        listEl.appendChild(errorDiv);
    }
}

async function loadSnapshots() {
    const snapshots = await gallery.getAllSnapshots();
    const listEl = document.getElementById('snapshot-list');

    if (snapshots.length === 0) {
        listEl.textContent = '';
        const emptyMsg = document.createElement('p');
        emptyMsg.style.gridColumn = '1/-1';
        emptyMsg.textContent = 'No snapshots yet. Run simulations to create snapshots.';
        listEl.appendChild(emptyMsg);
        return;
    }

    listEl.textContent = ''; // Clear existing content
    snapshots.reverse().forEach(snapshot => {
        const card = createSnapshotCard(snapshot);
        listEl.appendChild(card);
    });
}

function createSnapshotCard(snapshot) {
    const card = document.createElement('div');
    card.className = 'snapshot-card';
    // Use addEventListener instead of inline onclick
    card.addEventListener('click', () => viewSnapshot(snapshot));

    const date = new Date(snapshot.timestamp);
    const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();

    // Create DOM elements safely without innerHTML
    const title = document.createElement('h3');
    title.textContent = snapshot.run_id || 'Untitled Run';
    card.appendChild(title);

    const meta = document.createElement('div');
    meta.className = 'meta';
    
    const dateDiv = document.createElement('div');
    dateDiv.textContent = `Date: ${dateStr}`;
    meta.appendChild(dateDiv);

    const stepDiv = document.createElement('div');
    stepDiv.textContent = `Step: ${Number(snapshot.step ?? 0)}`;
    meta.appendChild(stepDiv);

    card.appendChild(meta);

    const metrics = document.createElement('div');
    metrics.className = 'metrics';
    // Coerce to Number before toFixed
    const e = Number(snapshot.e || 0).toFixed(1);
    const t = Number(snapshot.t || 0).toFixed(2);
    const s = Number(snapshot.s || 0).toFixed(2);
    const lambda = Number(snapshot.lambda_estimate || 0).toFixed(2);
    metrics.textContent = `E=${e} T=${t} S=${s} λ=${lambda}`;
    card.appendChild(metrics);

    return card;
}

function viewSnapshot(snapshot) {
    currentSnapshot = snapshot;
    const viewer = document.getElementById('viewer');
    const details = document.getElementById('viewer-details');

    const date = new Date(snapshot.timestamp);
    const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();

    // Create DOM elements safely without innerHTML
    details.textContent = ''; // Clear existing content

    const title = document.createElement('h2');
    title.textContent = snapshot.run_id || 'Untitled Run';
    details.appendChild(title);

    const detailsDiv = document.createElement('div');
    detailsDiv.style.cssText = 'margin-top: 15px; line-height: 1.8;';

    const fields = [
        { label: 'Date', value: dateStr },
        { label: 'Step', value: Number(snapshot.step ?? 0) },
        { label: 'Energy', value: Number(snapshot.e || 0).toFixed(2) },
        { label: 'Temperature', value: Number(snapshot.t || 0).toFixed(3) },
        { label: 'Entropy', value: Number(snapshot.s || 0).toFixed(3) },
        { label: 'Free Energy', value: Number(snapshot.f || 0).toFixed(2) },
        { label: 'Lambda', value: Number(snapshot.lambda_estimate || 0).toFixed(3) }
    ];

    fields.forEach(field => {
        const fieldDiv = document.createElement('div');
        const strong = document.createElement('strong');
        strong.textContent = `${field.label}: `;
        fieldDiv.appendChild(strong);
        fieldDiv.appendChild(document.createTextNode(field.value));
        detailsDiv.appendChild(fieldDiv);
    });

    details.appendChild(detailsDiv);

    const codeDiv = document.createElement('div');
    codeDiv.style.marginTop = '15px';
    const codeLabel = document.createElement('strong');
    codeLabel.textContent = 'Code:';
    codeDiv.appendChild(codeLabel);
    const codePre = document.createElement('pre');
    codePre.style.cssText = 'background: #000; padding: 10px; margin-top: 5px; overflow-x: auto; border: 1px solid #003300;';
    codePre.textContent = snapshot.code_hash || 'N/A';
    codeDiv.appendChild(codePre);
    details.appendChild(codeDiv);

    const buttonDiv = document.createElement('div');
    buttonDiv.style.marginTop = '15px';

    const replayBtn = document.createElement('button');
    replayBtn.textContent = '▶ Replay';
    replayBtn.style.cssText = 'padding: 8px 16px; background: #003300; border: 1px solid #00ff00; color: #00ff00; cursor: pointer; font-family: inherit;';
    replayBtn.addEventListener('click', () => replaySnapshot(snapshot.id));
    buttonDiv.appendChild(replayBtn);

    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '🗑 Delete';
    deleteBtn.style.cssText = 'padding: 8px 16px; background: #330000; border: 1px solid #ff0000; color: #ff0000; cursor: pointer; font-family: inherit; margin-left: 10px;';
    deleteBtn.addEventListener('click', () => deleteSnapshotUI(snapshot.id));
    buttonDiv.appendChild(deleteBtn);

    details.appendChild(buttonDiv);

    viewer.classList.add('active');
    // Set focus to close button for accessibility
    const closeBtn = document.getElementById('viewer-close');
    if (closeBtn) {
        closeBtn.focus();
    }
}

window.closeViewer = function() {
    const viewer = document.getElementById('viewer');
    viewer.classList.remove('active');
    currentSnapshot = null;
    // Restore focus to document body or previously focused element
    document.body.focus();
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

// Attach close button listener directly (script is loaded as module, so DOM is ready)
const closeBtn = document.getElementById('viewer-close');
if (closeBtn) {
    closeBtn.addEventListener('click', closeViewer);
}

// Add Escape key listener to close viewer
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
        const viewer = document.getElementById('viewer');
        if (viewer.classList.contains('active')) {
            closeViewer();
        }
    }
});
