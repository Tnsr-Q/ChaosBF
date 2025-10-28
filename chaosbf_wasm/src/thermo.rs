use crate::state::SimState;

/// Calculate local Shannon entropy over a byte window
pub fn local_entropy(bytes: &[u8]) -> f32 {
    if bytes.is_empty() {
        return 0.0;
    }

    // Build histogram
    let mut hist = [0u16; 256];
    for &b in bytes {
        hist[b as usize] = hist[b as usize].saturating_add(1);
    }

    let n = bytes.len() as f32;
    let mut h = 0.0;

    for &count in hist.iter() {
        if count == 0 { continue; }
        let p = (count as f32) / n;
        h -= p * p.ln(); // nats (natural log)
    }

    h
}

/// Landauer window-based entropy cost update
pub fn landauer_update(
    state: &mut SimState,
    old_win: &[u8],
    new_win: &[u8],
    credit_factor: f32
) {
    let h0 = local_entropy(old_win);
    let h1 = local_entropy(new_win);
    let dh = h1 - h0;

    if dh < 0.0 {
        // Erasure cost → energy penalty proportional to |ΔH|
        state.e += dh * state.t;
    } else {
        // Partial credit for entropy increase
        state.e += credit_factor * dh * state.t;
    }
}
