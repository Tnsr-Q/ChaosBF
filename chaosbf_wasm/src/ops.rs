#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Op {
    Gt, Lt, Plus, Minus, LBr, RBr, Dot, Comma,
    Caret, Vee, Colon, Semi, Q, Star, At, Eq, Bang,
    LCurly, RCurly, Hash, Percent, Tilde, Unknown
}

#[inline]
pub fn op_from_byte(b: u8) -> Op {
    match b {
        b'>' => Op::Gt,    b'<' => Op::Lt,
        b'+' => Op::Plus,  b'-' => Op::Minus,
        b'[' => Op::LBr,   b']' => Op::RBr,
        b'.' => Op::Dot,   b',' => Op::Comma,
        b'^' => Op::Caret, b'v' => Op::Vee,
        b':' => Op::Colon, b';' => Op::Semi,
        b'?' => Op::Q,     b'*' => Op::Star,
        b'@' => Op::At,    b'=' => Op::Eq,
        b'!' => Op::Bang,  b'{' => Op::LCurly,
        b'}' => Op::RCurly,b'#' => Op::Hash,
        b'%' => Op::Percent,b'~'=> Op::Tilde,
        _    => Op::Unknown
    }
}

#[inline]
pub fn delta_e(op: Op, depth: usize, slocal: f32) -> f32 {
    let leak = if matches!(op, Op::LBr | Op::RBr | Op::LCurly | Op::RCurly) {
        1.0 + (depth as f32) / 3.0
    } else {
        0.0
    };

    let base = match op {
        Op::Gt | Op::Lt => -1.0,
        Op::Plus        => -2.0,
        Op::Minus       =>  1.0,
        Op::Dot | Op::Comma => -1.0,
        Op::Caret | Op::Vee => -1.0,
        Op::Colon       =>  0.0,
        Op::Semi        => -slocal,  // Entropy-dependent cost
        Op::Q           => -2.0,
        Op::Star        => -10.0,
        Op::At          => -6.0,
        Op::Eq          =>  0.0,
        Op::Bang        => -1.0,
        Op::LCurly      => -2.0,
        Op::RCurly      =>  0.0,
        Op::Hash        =>  0.0,
        Op::Percent     => -1.0,
        Op::Tilde       =>  5.0,
        _               =>  0.0,
    };
    base - leak
}

// Available ops for mutation
pub const OPS: &[u8] = b"><+-[].,^v:;?*@=!{}#%~";
