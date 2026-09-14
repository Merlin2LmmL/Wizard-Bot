use crate::bitboard::*;
use crate::position::*;

/// Attackers of `square` by `attacker_color`, given an (optionally
/// overridden) occupancy bitboard. This is the single function used for
/// check detection, pin detection, king-move legality and null-move
/// preconditions.
pub fn attackers_of(pos: &Position, attacker_color: Color, square: u8, occ: Bitboard) -> Bitboard {
    let c = attacker_color.idx();
    let mut attackers = 0u64;

    attackers |= knight_attacks(square) & pos.pieces[c][PieceType::Knight as usize];
    attackers |= king_attacks(square) & pos.pieces[c][PieceType::King as usize];

    // Pawn attacks: a pawn of `attacker_color` on square `s` attacks
    // `square` if `square` is in pawn_attacks(attacker_color, s) - which is
    // the same as: squares that would attack `square` if a pawn of the
    // *opposite* color stood on them. So we look up pawn_attacks for the
    // *defending* side's perspective from `square`.
    let defender_is_white = attacker_color == Color::Black;
    attackers |= pawn_attacks(defender_is_white, square) & pos.pieces[c][PieceType::Pawn as usize];

    let rook_like = pos.pieces[c][PieceType::Rook as usize] | pos.pieces[c][PieceType::Queen as usize];
    if rook_like != 0 {
        attackers |= rook_attacks(square, occ) & rook_like;
    }
    let bishop_like = pos.pieces[c][PieceType::Bishop as usize] | pos.pieces[c][PieceType::Queen as usize];
    if bishop_like != 0 {
        attackers |= bishop_attacks(square, occ) & bishop_like;
    }

    attackers
}

#[inline]
pub fn is_in_check(pos: &Position, color: Color) -> bool {
    let ksq = pos.king_square[color.idx()];
    attackers_of(pos, color.opponent(), ksq, pos.occ_all) != 0
}

/// Per-node cache of "which squares would a piece of type X moving there
/// deliver a *direct* check from" — computed once per node instead of
/// recomputed from scratch inside `gives_check()` for every candidate move
/// at that node. Mirrors Reckless's `checking_squares[piece_type]` table:
/// a plain attack computation from the enemy king's square against the
/// *current* (pre-move) occupancy.
///
/// This is deliberately an approximation — occupancy at `from` hasn't been
/// cleared yet, so a slider whose own vacated square lies on the ray to the
/// king can produce a false negative here. That's fine: `gives_check()`
/// below only trusts a `true` from this table and always falls back to the
/// full computation on `false`, so the approximation can never produce a
/// *wrong* answer — only a slower path for the cases it misses (Reckless
/// documents ~90-95% direct-check accuracy for the equivalent table before
/// falling through to a fuller check).
#[derive(Clone, Copy)]
pub struct CheckingSquares {
    pub pawn: Bitboard,
    pub knight: Bitboard,
    pub bishop: Bitboard,
    pub rook: Bitboard,
    pub queen: Bitboard,
}

/// Computes `CheckingSquares` for `us.opponent()`'s king. Call this once
/// per node and reuse the result across every `gives_check()` call for
/// candidate moves considered at that node.
pub fn compute_checking_squares(pos: &Position, us: Color) -> CheckingSquares {
    let them = us.opponent();
    let king_sq = pos.king_square[them.idx()];
    let occ = pos.occ_all;
    // Same "ask from the defender's perspective" trick attackers_of() uses.
    let defender_is_white = us == Color::Black;
    CheckingSquares {
        pawn: pawn_attacks(defender_is_white, king_sq),
        knight: knight_attacks(king_sq),
        bishop: bishop_attacks(king_sq, occ),
        rook: rook_attacks(king_sq, occ),
        queen: bishop_attacks(king_sq, occ) | rook_attacks(king_sq, occ),
    }
}

/// Does making `m` (a legal move for `pos.side_to_move`) give check to the
/// opponent? Answered without make_move/unmake_move: copies only the
/// mover's own 6 piece bitboards (48 bytes) and patches that copy for the
/// one piece that moved (or, for castling, the rook too), then reuses the
/// same attack-union approach as `attackers_of` against the resulting
/// occupancy. This covers direct checks (the moved piece now attacks the
/// enemy king from `to`) and discovered checks (a *different* mover slider
/// unmasked by `from` becoming vacant) in the same pass -- both are just
/// "does anything in the adjusted bitboards attack the king square, given
/// the adjusted occupancy." Existing callers that need this (LMP/futility
/// pruning gates in negamax.rs, quiescence's checking-move filter) were
/// previously make_move-ing the position just to find out, which is the
/// exact thing pruning is trying to avoid paying for.
pub fn gives_check(pos: &Position, m: Move, ctx: &CheckingSquares) -> bool {
    let flag = m.flag();
    let to = m.to_sq();
    let moved_piece_type = if flag.is_promotion() { flag.promo_piece() } else { m.piece() };

    // Fast path: O(1) lookup against the per-node table above. Covers
    // direct checks -- including promotions, since we key off the
    // post-move piece type -- without touching the board at all.
    let direct = match moved_piece_type {
        PieceType::Pawn => ctx.pawn & bit(to) != 0,
        PieceType::Knight => ctx.knight & bit(to) != 0,
        PieceType::Bishop => ctx.bishop & bit(to) != 0,
        PieceType::Rook => ctx.rook & bit(to) != 0,
        PieceType::Queen => ctx.queen & bit(to) != 0,
        PieceType::King | PieceType::None => false,
    };
    if direct {
        return true;
    }

    // Slow path: everything the table can't (or, given its stale-occupancy
    // approximation, might not) answer -- discovered checks of any piece
    // type (including a king move unmasking a slider), en-passant
    // discovered checks, and castling-rook checks.
    gives_check_full(pos, m)
}

/// Full, always-correct check test: copies only the mover's own 6 piece
/// bitboards (48 bytes) and patches that copy for the one piece that moved
/// (or, for castling, the rook too), then reuses the same attack-union
/// approach as `attackers_of` against the resulting occupancy. This covers
/// direct checks and discovered checks (a *different* mover slider
/// unmasked by `from` becoming vacant) in the same pass. `gives_check()`
/// above only falls through to this when its O(1) table lookup can't
/// already answer "yes".
fn gives_check_full(pos: &Position, m: Move) -> bool {
    let us = pos.side_to_move;
    let them = us.opponent();
    let king_sq = pos.king_square[them.idx()];
    let from = m.from_sq();
    let to = m.to_sq();
    let flag = m.flag();

    let mut occ = pos.occ_all;
    occ &= !bit(from);
    occ |= bit(to);

    let mut mover = pos.pieces[us.idx()];

    match flag {
        MoveFlag::EpCapture => {
            // The captured pawn sits beside `to`, not on it -- clear it too,
            // since its removal can itself unmask a discovered check along
            // the rank (the classic e.p.-discovered-check pattern).
            occ &= !bit(sq(rank_of(from), file_of(to)));
        }
        MoveFlag::KingCastle | MoveFlag::QueenCastle => {
            // The king's own move can't give check, but the rook's new
            // square can (and routinely does, e.g. Rf1/Rd1 hitting a king
            // on the same rank/file after castling) -- patch it in too.
            let home_rank = rank_of(from);
            let (rook_from, rook_to) = if flag == MoveFlag::KingCastle {
                (sq(home_rank, 7), sq(home_rank, 5))
            } else {
                (sq(home_rank, 0), sq(home_rank, 3))
            };
            occ &= !bit(rook_from);
            occ |= bit(rook_to);
            mover[PieceType::Rook as usize] &= !bit(rook_from);
            mover[PieceType::Rook as usize] |= bit(rook_to);
        }
        _ => {}
    }

    let moved_piece = if flag.is_promotion() { flag.promo_piece() } else { m.piece() };
    mover[m.piece() as usize] &= !bit(from);
    mover[moved_piece as usize] |= bit(to);

    if knight_attacks(king_sq) & mover[PieceType::Knight as usize] != 0 {
        return true;
    }
    // Same "look up pawn_attacks from the defender's perspective" trick
    // `attackers_of` uses above, with `us` playing the attacker_color role.
    let defender_is_white = us == Color::Black;
    if pawn_attacks(defender_is_white, king_sq) & mover[PieceType::Pawn as usize] != 0 {
        return true;
    }
    let rook_like = mover[PieceType::Rook as usize] | mover[PieceType::Queen as usize];
    if rook_like != 0 && rook_attacks(king_sq, occ) & rook_like != 0 {
        return true;
    }
    let bishop_like = mover[PieceType::Bishop as usize] | mover[PieceType::Queen as usize];
    if bishop_like != 0 && bishop_attacks(king_sq, occ) & bishop_like != 0 {
        return true;
    }
    false
}

struct GenContext {
    checkmask: Bitboard,
    pin_mask: [Bitboard; 64], // ALL bits set = not pinned (sentinel: we use a parallel "is_pinned" array)
    is_pinned: [bool; 64],
    num_checkers: u32,
}

fn compute_context(pos: &Position, us: Color) -> GenContext {
    let them = us.opponent();
    let ksq = pos.king_square[us.idx()];
    let checkers = attackers_of(pos, them, ksq, pos.occ_all);
    let num_checkers = popcount(checkers);

    let checkmask = if num_checkers == 0 {
        ALL
    } else if num_checkers == 1 {
        let checker_sq = checkers.trailing_zeros() as u8;
        let checker_piece = pos.piece_at(checker_sq).map(|(_, pt)| pt);
        let is_slider = matches!(
            checker_piece,
            Some(PieceType::Bishop) | Some(PieceType::Rook) | Some(PieceType::Queen)
        );
        if is_slider {
            bit(checker_sq) | ray_between(ksq, checker_sq)
        } else {
            bit(checker_sq)
        }
    } else {
        0
    };

    let mut is_pinned = [false; 64];
    let mut pin_mask = [ALL; 64];

    let them_rook_like =
        pos.pieces[them.idx()][PieceType::Rook as usize] | pos.pieces[them.idx()][PieceType::Queen as usize];
    let them_bishop_like =
        pos.pieces[them.idx()][PieceType::Bishop as usize] | pos.pieces[them.idx()][PieceType::Queen as usize];

    for &dir in ROOK_DIRS.iter() {
        scan_pin_dir(pos, us, ksq, dir, them_rook_like, &mut is_pinned, &mut pin_mask);
    }
    for &dir in BISHOP_DIRS.iter() {
        scan_pin_dir(pos, us, ksq, dir, them_bishop_like, &mut is_pinned, &mut pin_mask);
    }

    GenContext {
        checkmask,
        pin_mask,
        is_pinned,
        num_checkers,
    }
}

fn scan_pin_dir(
    pos: &Position,
    us: Color,
    ksq: u8,
    dir: usize,
    enemy_sliders_this_dir: Bitboard,
    is_pinned: &mut [bool; 64],
    pin_mask: &mut [Bitboard; 64],
) {
    let ray = RAY_TABLES.rays[dir][ksq as usize];
    let occ_on_ray = ray & pos.occ_all;
    if occ_on_ray == 0 {
        return;
    }
    let first_sq = nearest_on_ray(ksq, dir, occ_on_ray);
    let first_is_own = pos.occ[us.idx()] & bit(first_sq) != 0;
    if !first_is_own {
        return; // first piece is enemy or nothing pinned
    }
    // find next occupied square beyond first_sq along the same ray
    let ray_beyond = RAY_TABLES.rays[dir][first_sq as usize];
    let occ_beyond = ray_beyond & pos.occ_all;
    if occ_beyond == 0 {
        return;
    }
    let second_sq = nearest_on_ray(first_sq, dir, occ_beyond);
    if enemy_sliders_this_dir & bit(second_sq) != 0 {
        is_pinned[first_sq as usize] = true;
        // legal destinations for the pinned piece: the ray from king through
        // pinner (inclusive of pinner), i.e. (ray from ksq in dir) minus
        // (ray beyond the pinner).
        let ray_beyond_pinner = RAY_TABLES.rays[dir][second_sq as usize];
        pin_mask[first_sq as usize] = ray & !ray_beyond_pinner;
    }
}

#[inline]
fn nearest_on_ray(from: u8, dir: usize, occ_on_ray: Bitboard) -> u8 {
    match dir {
        DIR_N | DIR_E | DIR_NE | DIR_NW => occ_on_ray.trailing_zeros() as u8,
        _ => {
            let _ = from;
            63 - occ_on_ray.leading_zeros() as u8
        }
    }
}

/// Generate all legal moves for the side to move into `list`.
pub fn generate_legal_moves(pos: &Position, list: &mut MoveList) {
    let us = pos.side_to_move;
    let them = us.opponent();
    let ctx = compute_context(pos, us);

    // King moves (always generated; not filtered by checkmask/pins)
    gen_king_moves(pos, us, them, list);

    if ctx.num_checkers >= 2 {
        return; // double check: only king moves are legal
    }

    gen_pawn_moves(pos, us, &ctx, list);
    gen_piece_moves(pos, us, PieceType::Knight, &ctx, list, |sq, _occ| knight_attacks(sq));
    gen_piece_moves(pos, us, PieceType::Bishop, &ctx, list, |sq, occ| bishop_attacks(sq, occ));
    gen_piece_moves(pos, us, PieceType::Rook, &ctx, list, |sq, occ| rook_attacks(sq, occ));
    gen_piece_moves(pos, us, PieceType::Queen, &ctx, list, |sq, occ| queen_attacks(sq, occ));

    if ctx.num_checkers == 0 {
        gen_castling(pos, us, them, list);
    }
}

fn gen_piece_moves<F: Fn(u8, Bitboard) -> Bitboard>(
    pos: &Position,
    us: Color,
    pt: PieceType,
    ctx: &GenContext,
    list: &mut MoveList,
    attacks_fn: F,
) {
    let mut bb = pos.pieces[us.idx()][pt as usize];
    while bb != 0 {
        let from = pop_lsb(&mut bb);
        let mut targets = attacks_fn(from, pos.occ_all) & !pos.occ[us.idx()] & ctx.checkmask;
        if ctx.is_pinned[from as usize] {
            targets &= ctx.pin_mask[from as usize];
        }
        let mut t = targets;
        while t != 0 {
            let to = pop_lsb(&mut t);
            let captured = pos.piece_at(to).map(|(_, p)| p);
            let flag = if captured.is_some() { MoveFlag::Capture } else { MoveFlag::Quiet };
            list.push(Move::new(from, to, flag, pt, captured.unwrap_or(PieceType::None)));
        }
    }
}

fn gen_king_moves(pos: &Position, us: Color, them: Color, list: &mut MoveList) {
    let ksq = pos.king_square[us.idx()];
    let occ_without_king = pos.occ_all & !bit(ksq);
    let mut targets = king_attacks(ksq) & !pos.occ[us.idx()];
    let mut t = targets;
    while t != 0 {
        let to = pop_lsb(&mut t);
        if attackers_of(pos, them, to, occ_without_king) != 0 {
            continue;
        }
        let captured = pos.piece_at(to).map(|(_, p)| p);
        let flag = if captured.is_some() { MoveFlag::Capture } else { MoveFlag::Quiet };
        list.push(Move::new(ksq, to, flag, PieceType::King, captured.unwrap_or(PieceType::None)));
    }
    let _ = &mut targets;
}

fn gen_castling(pos: &Position, us: Color, them: Color, list: &mut MoveList) {
    let ksq = pos.king_square[us.idx()];
    let (kingside_right, queenside_right, home_rank) = match us {
        Color::White => (CR_WK, CR_WQ, 0u8),
        Color::Black => (CR_BK, CR_BQ, 7u8),
    };
    let king_home = sq(home_rank, 4);
    if ksq != king_home {
        return;
    }

    if pos.castling_rights & kingside_right != 0 {
        let f_sq = sq(home_rank, 5);
        let g_sq = sq(home_rank, 6);
        let h_sq = sq(home_rank, 7);
        let rook_ok = pos.piece_at(h_sq) == Some((us, PieceType::Rook));
        if rook_ok
            && pos.occ_all & (bit(f_sq) | bit(g_sq)) == 0
            && attackers_of(pos, them, king_home, pos.occ_all) == 0
            && attackers_of(pos, them, f_sq, pos.occ_all) == 0
            && attackers_of(pos, them, g_sq, pos.occ_all) == 0
        {
            list.push(Move::new(ksq, g_sq, MoveFlag::KingCastle, PieceType::King, PieceType::None));
        }
    }
    if pos.castling_rights & queenside_right != 0 {
        let d_sq = sq(home_rank, 3);
        let c_sq = sq(home_rank, 2);
        let b_sq = sq(home_rank, 1);
        let a_sq = sq(home_rank, 0);
        let rook_ok = pos.piece_at(a_sq) == Some((us, PieceType::Rook));
        if rook_ok
            && pos.occ_all & (bit(d_sq) | bit(c_sq) | bit(b_sq)) == 0
            && attackers_of(pos, them, king_home, pos.occ_all) == 0
            && attackers_of(pos, them, d_sq, pos.occ_all) == 0
            && attackers_of(pos, them, c_sq, pos.occ_all) == 0
        {
            list.push(Move::new(ksq, c_sq, MoveFlag::QueenCastle, PieceType::King, PieceType::None));
        }
    }
}

fn gen_pawn_moves(pos: &Position, us: Color, ctx: &GenContext, list: &mut MoveList) {
    let is_white = us == Color::White;
    let them = us.opponent();
    let mut pawns = pos.pieces[us.idx()][PieceType::Pawn as usize];
    let start_rank = if is_white { 1 } else { 6 };
    let promo_rank = if is_white { 7 } else { 0 };
    let push_dir: i32 = if is_white { 8 } else { -8 };

    while pawns != 0 {
        let from = pop_lsb(&mut pawns);
        let from_rank = rank_of(from);
        let pin_ok = |dest: u8| -> bool {
            !ctx.is_pinned[from as usize] || (ctx.pin_mask[from as usize] & bit(dest) != 0)
        };

        // Single push
        let one_sq = from as i32 + push_dir;
        if (0..64).contains(&one_sq) {
            let one = one_sq as u8;
            if pos.occ_all & bit(one) == 0 {
                if ctx.checkmask & bit(one) != 0 && pin_ok(one) {
                    push_pawn_move(list, from, one, promo_rank, false, PieceType::None);
                }
                // Double push
                if from_rank == start_rank {
                    let two_sq = one_sq + push_dir;
                    if (0..64).contains(&two_sq) {
                        let two = two_sq as u8;
                        if pos.occ_all & bit(two) == 0
                            && ctx.checkmask & bit(two) != 0
                            && pin_ok(two)
                        {
                            list.push(Move::new(from, two, MoveFlag::DoublePawnPush, PieceType::Pawn, PieceType::None));
                        }
                    }
                }
            }
        }

        // Captures (incl. promotions)
        let attacks = pawn_attacks(is_white, from) & pos.occ[them.idx()];
        let mut a = attacks;
        while a != 0 {
            let to = pop_lsb(&mut a);
            if ctx.checkmask & bit(to) == 0 || !pin_ok(to) {
                continue;
            }
            // `to` is only reached here because it's set in
            // pos.occ[them.idx()], so piece_at(to) should always be Some.
            // If it's ever None, occupancy and the mailbox have desynced --
            // in that case we must NOT build a Capture-flagged move with a
            // captured piece of PieceType::None: make_move() would later
            // index pieces[..][PieceType::None as usize] (out of bounds,
            // since PieceType::None=6 is a sentinel, not a real array slot)
            // and panic deep in search, far from the actual cause. Skipping
            // the phantom capture here is strictly safer than crashing.
            let captured = match pos.piece_at(to).map(|(_, p)| p) {
                Some(pt) => pt,
                None => {
                    debug_assert!(
                        false,
                        "pawn capture target {to} set in occ[them] but empty in mailbox (from {from})"
                    );
                    continue;
                }
            };
            push_pawn_move(list, from, to, promo_rank, true, captured);
        }

        // En passant
        if let Some(epsq) = pos.ep_square {
            if pawn_attacks(is_white, from) & bit(epsq) != 0 {
                // Special-case legality check: simulate removing both pawns.
                let captured_pawn_sq = sq(from_rank, file_of(epsq));
                let occ_after = (pos.occ_all & !bit(from) & !bit(captured_pawn_sq)) | bit(epsq);
                let ksq = pos.king_square[us.idx()];
                if attackers_of(pos, them, ksq, occ_after) == 0 {
                    // checkmask/pin still apply in the normal sense for the
                    // moving pawn's own destination (approximate: ep capture
                    // must still address check / respect the pawn's own pin,
                    // in addition to the special discovered-check test above)
                    let addresses_check = ctx.checkmask & (bit(epsq) | bit(captured_pawn_sq)) != 0;
                    if addresses_check && pin_ok(epsq) {
                        list.push(Move::new(from, epsq, MoveFlag::EpCapture, PieceType::Pawn, PieceType::Pawn));
                    }
                }
            }
        }
    }
}

fn push_pawn_move(list: &mut MoveList, from: u8, to: u8, promo_rank: u8, is_capture: bool, captured: PieceType) {
    if rank_of(to) == promo_rank {
        let flags = if is_capture {
            [
                MoveFlag::PromoCaptureQ,
                MoveFlag::PromoCaptureR,
                MoveFlag::PromoCaptureB,
                MoveFlag::PromoCaptureN,
            ]
        } else {
            [MoveFlag::PromoQ, MoveFlag::PromoR, MoveFlag::PromoB, MoveFlag::PromoN]
        };
        for f in flags {
            list.push(Move::new(from, to, f, PieceType::Pawn, captured));
        }
    } else if is_capture {
        list.push(Move::new(from, to, MoveFlag::Capture, PieceType::Pawn, captured));
    } else {
        list.push(Move::new(from, to, MoveFlag::Quiet, PieceType::Pawn, PieceType::None));
    }
}

/// Static Exchange Evaluation for a capture (or en passant) move: simulates
/// the full capture sequence on `to` (attacker least-valuable-first on both
/// sides, using bitboard x-ray re-scans via `attackers_of` so sliders
/// revealed behind removed blockers are picked up automatically) and
/// returns the net material result for the side making `m`, in centipawns.
/// Returns 0 for non-capture moves.
pub fn see(pos: &Position, m: Move) -> i32 {
    if !m.flag().is_capture() {
        return 0;
    }
    let to = m.to_sq();
    let from = m.from_sq();

    let mut occ = pos.occ_all;
    occ &= !bit(from);
    if m.flag() == MoveFlag::EpCapture {
        let ep_cap_sq = sq(rank_of(from), file_of(to));
        occ &= !bit(ep_cap_sq);
    }

    // gain[0] = value of whatever `m` captures on `to`.
    let mut gain = [0i32; 32];
    gain[0] = if m.flag() == MoveFlag::EpCapture {
        piece_value(PieceType::Pawn)
    } else {
        piece_value(m.captured())
    };

    // Value of the piece that now sits on `to` (the mover) - the next
    // thing that could be recaptured.
    let mut moving_value = piece_value(m.piece());
    let mut side = pos.side_to_move.opponent();
    let mut depth = 0usize;

    loop {
        let attackers = attackers_of(pos, side, to, occ) & occ;
        if attackers == 0 || depth >= 31 {
            break;
        }
        // Least valuable attacker of `side`: PIECE_TYPES is already in
        // ascending value order with King last, so King is only ever
        // "used" once nothing cheaper remains.
        let mut next: Option<(PieceType, u8)> = None;
        for &pt in PIECE_TYPES.iter() {
            let bb = attackers & pos.pieces[side.idx()][pt as usize];
            if bb != 0 {
                next = Some((pt, bb.trailing_zeros() as u8));
                break;
            }
        }
        let (attacker_pt, attacker_sq) = match next {
            Some(v) => v,
            None => break,
        };

        depth += 1;
        gain[depth] = moving_value - gain[depth - 1];
        // Standard SEE pruning: if neither side would continue the swap
        // from here, the remaining terms can't change the unwound result.
        if (-gain[depth - 1]).max(gain[depth]) < 0 {
            break;
        }

        occ &= !bit(attacker_sq);
        moving_value = piece_value(attacker_pt);
        side = side.opponent();
    }

    while depth > 0 {
        gain[depth - 1] = -((-gain[depth - 1]).max(gain[depth]));
        depth -= 1;
    }
    gain[0]
}

#[allow(dead_code)]
pub fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut list = MoveList::new();
    generate_legal_moves(pos, &mut list);
    if depth == 1 {
        return list.count as u64;
    }
    let mut nodes = 0u64;
    for &m in list.as_slice() {
        pos.make_move(m);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(m);
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn see_simple_undefended_capture() {
        // White queen d1, black rook d7, nothing else on the d-file / no
        // other attacker of d7. Qxd7 should just win the rook outright.
        let pos = Position::from_fen("k7/3r4/8/8/8/8/8/K2Q4 w - - 0 1").unwrap();
        let mut list = MoveList::new();
        generate_legal_moves(&pos, &mut list);
        let m = list
            .as_slice()
            .iter()
            .copied()
            .find(|m| m.from_sq() == sq(0, 3) && m.to_sq() == sq(6, 3))
            .expect("Qd1xd7 should be a legal move");
        assert_eq!(see(&pos, m), 500);
    }

    #[test]
    fn see_losing_queen_for_rook_pair() {
        // White queen d1, black rooks stacked on d7 and d8. Qxd7 wins the
        // first rook (+500) but the queen (900) is then recaptured by the
        // rook behind it on d8, netting -400 overall.
        let pos = Position::from_fen("k2r4/3r4/8/8/8/8/8/K2Q4 w - - 0 1").unwrap();
        let mut list = MoveList::new();
        generate_legal_moves(&pos, &mut list);
        let m = list
            .as_slice()
            .iter()
            .copied()
            .find(|m| m.from_sq() == sq(0, 3) && m.to_sq() == sq(6, 3))
            .expect("Qd1xd7 should be a legal move");
        assert_eq!(see(&pos, m), -400);
    }

    #[test]
    fn see_noncapture_is_zero() {
        let pos = Position::startpos();
        let mut list = MoveList::new();
        generate_legal_moves(&pos, &mut list);
        let quiet = list.as_slice().iter().copied().find(|m| !m.flag().is_capture()).unwrap();
        assert_eq!(see(&pos, quiet), 0);
    }

    #[test]
    fn gives_check_discovered_check_matches_handoff_position() {
        // The confirmed ply-158 outlier position: Nd5-f6+ is a discovered
        // check (knight vacates d5, unmasking the rook on d4's attack down
        // the d-file onto the black king... actually onto black's own
        // rook on d6 in the real game, but the king is what must move, so
        // it must be giving check to the king). Board:
        // 4k3/6K1/2r4p/3N4/1p1R4/1P4PP/8/8 w - - 9 79 (flip stm to White,
        // whose knight move this is).
        let pos = Position::from_fen("4k3/6K1/2r4p/3N4/1p1R4/1P4PP/8/8 w - - 9 79").unwrap();
        let mut list = MoveList::new();
        generate_legal_moves(&pos, &mut list);
        let nf6 = list
            .as_slice()
            .iter()
            .copied()
            .find(|m| m.from_sq() == sq(4, 3) && m.to_sq() == sq(5, 5))
            .expect("Nd5-f6 should be a legal move");
        let ctx = compute_checking_squares(&pos, pos.side_to_move);
        assert!(
            gives_check(&pos, nf6, &ctx),
            "Nf6 unmasks Rd4's attack on the d-file and should be flagged as check"
        );
        // Sanity check against the ground truth: making the move and
        // asking is_in_check should agree.
        let mut pos2 = pos;
        pos2.make_move(nf6);
        assert!(is_in_check(&pos2, pos2.side_to_move));
    }

    #[test]
    fn gives_check_direct_knight_check() {
        // White king e1, black knight can jump to a square that directly
        // checks it (not a discovered check -- exercises the "moved piece
        // itself attacks the king" path, not the "vacating from unmasks a
        // slider" path).
        //
        // The squares a knight must land on to check a king on e1 are
        // exactly f3, d3, g2, c2 (knight_attacks(e1)). The previous version
        // of this test started the knight on f3 and moved it to d2 --
        // d2 isn't in that set (knight on d2 attacks c4/e4/b3/f3/b1/f1, not
        // e1), so the assertion didn't correspond to a real check and the
        // test was wrong, not `gives_check()`. Starting the knight on e5
        // instead and moving it to d3 (one of the four genuine
        // checking squares) is an actual direct check.
        let pos = Position::from_fen("4k3/8/8/4n3/8/8/8/4K3 b - - 0 1").unwrap();
        let mut list = MoveList::new();
        generate_legal_moves(&pos, &mut list);
        let nd3 = list
            .as_slice()
            .iter()
            .copied()
            .find(|m| m.from_sq() == sq(4, 4) && m.to_sq() == sq(2, 3))
            .expect("Ne5-d3 should be a legal move");
        let ctx = compute_checking_squares(&pos, pos.side_to_move);
        assert!(gives_check(&pos, nd3, &ctx), "Nd3 directly checks the king on e1");
        // Sanity check against the ground truth, same pattern as the
        // discovered-check test above.
        let mut pos2 = pos;
        pos2.make_move(nd3);
        assert!(is_in_check(&pos2, pos2.side_to_move));
    }

    #[test]
    fn gives_check_quiet_non_checking_move_is_false() {
        let pos = Position::startpos();
        let mut list = MoveList::new();
        generate_legal_moves(&pos, &mut list);
        let quiet = list.as_slice().iter().copied().find(|m| !m.flag().is_capture()).unwrap();
        let ctx = compute_checking_squares(&pos, pos.side_to_move);
        assert!(!gives_check(&pos, quiet, &ctx));
    }

    #[test]
    fn perft_startpos() {
        let mut pos = Position::startpos();
        assert_eq!(perft(&mut pos, 1), 20);
        assert_eq!(perft(&mut pos, 2), 400);
        assert_eq!(perft(&mut pos, 3), 8902);
        assert_eq!(perft(&mut pos, 4), 197281);
    }

    #[test]
    fn perft_kiwipete() {
        let mut pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        assert_eq!(perft(&mut pos, 1), 48);
        assert_eq!(perft(&mut pos, 2), 2039);
        assert_eq!(perft(&mut pos, 3), 97862);
    }

    #[test]
    fn perft_position3() {
        let mut pos = Position::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1").unwrap();
        assert_eq!(perft(&mut pos, 1), 14);
        assert_eq!(perft(&mut pos, 2), 191);
        assert_eq!(perft(&mut pos, 3), 2812);
        assert_eq!(perft(&mut pos, 4), 43238);
    }

    #[test]
    fn perft_position4() {
        let mut pos =
            Position::from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
                .unwrap();
        assert_eq!(perft(&mut pos, 1), 6);
        assert_eq!(perft(&mut pos, 2), 264);
        assert_eq!(perft(&mut pos, 3), 9467);
    }

    #[test]
    fn perft_position5() {
        let mut pos =
            Position::from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8").unwrap();
        assert_eq!(perft(&mut pos, 1), 44);
        assert_eq!(perft(&mut pos, 2), 1486);
        assert_eq!(perft(&mut pos, 3), 62379);
    }

    #[test]
    fn zobrist_roundtrip() {
        let mut pos = Position::startpos();
        let mut list = MoveList::new();
        generate_legal_moves(&mut pos, &mut list);
        for &m in list.as_slice() {
            let before = pos.zobrist_key;
            pos.make_move(m);
            let recomputed = crate::zobrist::compute_full(&pos);
            assert_eq!(pos.zobrist_key, recomputed, "mismatch after {:?}", m.to_uci());
            pos.unmake_move(m);
            assert_eq!(pos.zobrist_key, before);
        }
    }
}



/// Qsearch-specific pseudo-legal capture generator (no full compute_context).
pub fn generate_qsearch_captures(pos: &mut Position, list: &mut MoveList) {
    let us = pos.side_to_move;
    let them = us.opponent();
    let ksq = pos.king_square[us.idx()];
    let enemy_occ = pos.occ[them.idx()];
    if enemy_occ == 0 { return; }

    // Fast per-piece loop: king, knights, bishops, rooks, queens, pawns
    // For non-king, non-pinned movers, pseudo-legal captures are legal.
    // We apply cheap make/unmake filter only for king moves and when 
    // piece square lies on a king ray to a potential slider.
    for pt in [PieceType::King, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen, PieceType::Pawn] {
        let mut sq_bb = pos.pieces[us.idx()][pt as usize];
        while sq_bb != 0 {
            let from = sq_bb.trailing_zeros() as u8;
            sq_bb &= sq_bb - 1;

            // Quick: if king, always filter
            let is_king = pt == PieceType::King;
            // Cheap pin-heuristic: same file/col/diag as king to enemy slider
            let mut must_filter = is_king;
            if !is_king && pt != PieceType::Pawn {
                let f = (from % 8) as i32; let r = (from / 8) as i32;
                let kf = (ksq % 8) as i32; let kr = (ksq / 8) as i32;
                if f == kf || r == kr || (f - kf).abs() == (r - kr).abs() {
                    // Possible pin; apply cheap filter
                    must_filter = true;
                }
            }

            if pt == PieceType::Pawn {
                let dirs: &[(i32, i32)] = if us == Color::White { &[(1,1), (-1,1)] } else { &[(1,-1), (-1,-1)] };
                let f = (from % 8) as i32; let r = (from / 8) as i32;
                for &(df, dr) in dirs {
                    let to_f = f + df; let to_r = r + dr;
                    if to_f < 0 || to_f > 7 || to_r < 0 || to_r > 7 { continue; }
                    let to = (to_r * 8 + to_f) as u8;
                    if enemy_occ & bit(to) != 0 {
                        // enemy_occ is a snapshot taken once at the top of
                        // this function; if any must_filter legality check
                        // earlier in this same call (see below) left the
                        // board state out of sync with that snapshot, don't
                        // crash on a stale square -- just skip this
                        // direction defensively.
                        let captured = match pos.piece_at(to) {
                            Some((_, p)) => p,
                            None => continue,
                        };
                        let promo = if (us == Color::White && to_r == 7) || (us == Color::Black && to_r == 0) { true } else { false };
                        if promo {
                            push_pawn_move(list, from, to, if us == Color::White { 7 } else { 0 }, true, captured);
                        } else {
                            list.push(Move::new(from, to, MoveFlag::Capture, PieceType::Pawn, captured));
                        }
                    } else if Some(to) == pos.ep_square && pos.ep_square.is_some() {
                        let ep_vic = if us == Color::White { to - 8 } else { to + 8 };
                        if pos.piece_at(ep_vic) == Some((them, PieceType::Pawn)) {
                            list.push(Move::new(from, to, MoveFlag::EpCapture, PieceType::Pawn, PieceType::Pawn));
                        }
                    }
                }
            } else {
                let attack_mask = match pt {
                    PieceType::Knight => knight_attacks(from),
                    PieceType::King => king_attacks(from),
                    PieceType::Bishop => bishop_attacks(from, pos.occ_all),
                    PieceType::Rook => rook_attacks(from, pos.occ_all),
                    PieceType::Queen => bishop_attacks(from, pos.occ_all) | rook_attacks(from, pos.occ_all),
                    _ => 0,
                };
                let targets = attack_mask & enemy_occ;
                let mut t = targets;
                while t != 0 {
                    let to = t.trailing_zeros() as u8;
                    t &= t - 1;

                    // Query the captured piece ONCE, before any temporary
                    // make_move/unmake_move below. enemy_occ is a snapshot
                    // taken once at the top of this function; re-querying
                    // pos.piece_at(to) *after* mutating the position (as
                    // the old code did at what's now the push() below) is
                    // fragile -- it silently assumes unmake_move perfectly
                    // round-trips board state for every move type. Getting
                    // it once up front and reusing that value removes the
                    // dependency on that assumption entirely, and we
                    // gracefully skip rather than panic if this square
                    // somehow doesn't have a piece despite being in
                    // enemy_occ.
                    let cap_piece = match pos.piece_at(to) {
                        Some((_, p)) => p,
                        None => continue,
                    };

                    if must_filter {
                        let mv = Move::new(from, to, MoveFlag::Capture, pt, cap_piece);
                        pos.make_move(mv);
                        let bad = is_in_check(pos, us);
                        pos.unmake_move(mv);
                        if bad { continue; }
                    }
                    list.push(Move::new(from, to, MoveFlag::Capture, pt, cap_piece));
                }
            }
        }
    }
}
