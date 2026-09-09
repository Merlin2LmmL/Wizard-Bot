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
