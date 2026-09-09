//! Bitboard primitives: square indexing, precomputed knight/king/pawn
//! attack tables, and sliding-piece attacks via runtime-generated fixed-shift
//! magic bitboards (single multiply + shift + table index per lookup).
//!
//! The classical ray-fill-with-first-blocker-scan approach is kept as
//! `sliding_attacks_ray_scan` — it's the reference implementation used to
//! populate the magic tables at startup, and it stays available for testing
//! / debugging so the two implementations can be cross-checked.

pub type Bitboard = u64;

pub const ALL: Bitboard = u64::MAX;
pub const EMPTY: Bitboard = 0;

pub const FILE_A: Bitboard = 0x0101_0101_0101_0101;

#[inline(always)]
pub const fn sq(rank: u8, file: u8) -> u8 {
    rank * 8 + file
}

#[inline(always)]
pub const fn rank_of(square: u8) -> u8 {
    square / 8
}

#[inline(always)]
pub const fn file_of(square: u8) -> u8 {
    square % 8
}

#[inline(always)]
pub const fn bit(square: u8) -> Bitboard {
    1u64 << square
}

#[inline(always)]
pub fn pop_lsb(bb: &mut Bitboard) -> u8 {
    let s = bb.trailing_zeros() as u8;
    *bb &= *bb - 1;
    s
}

#[inline(always)]
pub fn popcount(bb: Bitboard) -> u32 {
    bb.count_ones()
}

/// 8 ray directions: 0=N,1=S,2=E,3=W,4=NE,5=NW,6=SE,7=SW
pub const DIR_N: usize = 0;
pub const DIR_S: usize = 1;
pub const DIR_E: usize = 2;
pub const DIR_W: usize = 3;
pub const DIR_NE: usize = 4;
pub const DIR_NW: usize = 5;
pub const DIR_SE: usize = 6;
pub const DIR_SW: usize = 7;

/// Orthogonal directions come first (rook-type), diagonal second (bishop-type).
pub const ROOK_DIRS: [usize; 4] = [DIR_N, DIR_S, DIR_E, DIR_W];
pub const BISHOP_DIRS: [usize; 4] = [DIR_NE, DIR_NW, DIR_SE, DIR_SW];

const fn step(square: i32, dir: usize) -> i32 {
    let r = (square / 8) as i32;
    let f = (square % 8) as i32;
    match dir {
        0 => {
            if r < 7 {
                square + 8
            } else {
                -1
            }
        }
        1 => {
            if r > 0 {
                square - 8
            } else {
                -1
            }
        }
        2 => {
            if f < 7 {
                square + 1
            } else {
                -1
            }
        }
        3 => {
            if f > 0 {
                square - 1
            } else {
                -1
            }
        }
        4 => {
            if r < 7 && f < 7 {
                square + 9
            } else {
                -1
            }
        }
        5 => {
            if r < 7 && f > 0 {
                square + 7
            } else {
                -1
            }
        }
        6 => {
            if r > 0 && f < 7 {
                square - 7
            } else {
                -1
            }
        }
        7 => {
            if r > 0 && f > 0 {
                square - 9
            } else {
                -1
            }
        }
        _ => -1,
    }
}

/// RAYS[dir][sq] = bitboard of all squares from `sq` (exclusive) to the
/// edge of the board along `dir`.
pub struct RayTables {
    pub rays: [[Bitboard; 64]; 8],
    pub knight: [Bitboard; 64],
    pub king: [Bitboard; 64],
    /// pawn_attacks[color][sq]: color 0 = white, 1 = black
    pub pawn_attacks: [[Bitboard; 64]; 2],
}

pub static RAY_TABLES: RayTables = build_ray_tables();

const fn build_ray_tables() -> RayTables {
    let mut rays = [[0u64; 64]; 8];
    let mut dir = 0;
    while dir < 8 {
        let mut s = 0;
        while s < 64 {
            let mut mask: u64 = 0;
            let mut cur = s as i32;
            loop {
                let nxt = step(cur, dir);
                if nxt < 0 {
                    break;
                }
                mask |= 1u64 << nxt;
                cur = nxt;
            }
            rays[dir][s] = mask;
            s += 1;
        }
        dir += 1;
    }

    let mut knight = [0u64; 64];
    let mut king = [0u64; 64];
    let mut pawn_attacks = [[0u64; 64]; 2];

    let mut s = 0usize;
    while s < 64 {
        let r = (s / 8) as i32;
        let f = (s % 8) as i32;

        // Knight
        let deltas: [(i32, i32); 8] = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2),
        ];
        let mut i = 0;
        let mut nb: u64 = 0;
        while i < 8 {
            let (dr, df) = deltas[i];
            let nr = r + dr;
            let nf = f + df;
            if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                nb |= 1u64 << (nr * 8 + nf);
            }
            i += 1;
        }
        knight[s] = nb;

        // King
        let kdeltas: [(i32, i32); 8] = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ];
        let mut kb: u64 = 0;
        let mut j = 0;
        while j < 8 {
            let (dr, df) = kdeltas[j];
            let nr = r + dr;
            let nf = f + df;
            if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                kb |= 1u64 << (nr * 8 + nf);
            }
            j += 1;
        }
        king[s] = kb;

        // Pawn attacks: white attacks upward (rank+1), black attacks downward (rank-1)
        let mut wb: u64 = 0;
        if r < 7 {
            if f > 0 {
                wb |= 1u64 << ((r + 1) * 8 + (f - 1));
            }
            if f < 7 {
                wb |= 1u64 << ((r + 1) * 8 + (f + 1));
            }
        }
        pawn_attacks[0][s] = wb;

        let mut bb2: u64 = 0;
        if r > 0 {
            if f > 0 {
                bb2 |= 1u64 << ((r - 1) * 8 + (f - 1));
            }
            if f < 7 {
                bb2 |= 1u64 << ((r - 1) * 8 + (f + 1));
            }
        }
        pawn_attacks[1][s] = bb2;

        s += 1;
    }

    RayTables {
        rays,
        knight,
        king,
        pawn_attacks,
    }
}

/// Sliding attack generation along a set of directions, stopping at (and
/// including) the first blocker in each direction. This is the classical
/// "ray fill with first-blocker trim" technique, equivalent to o^(o-2r).
/// Kept as the reference implementation for magic-table construction and
/// for differential testing against the magic-based lookups below.
#[inline]
pub fn sliding_attacks_ray_scan(square: u8, occ: Bitboard, dirs: &[usize; 4]) -> Bitboard {
    let mut attacks: Bitboard = 0;
    for &dir in dirs.iter() {
        let full_ray = RAY_TABLES.rays[dir][square as usize];
        attacks |= full_ray;
        let blockers = full_ray & occ;
        if blockers != 0 {
            // Find nearest blocker along this ray direction and remove
            // everything beyond it.
            let nearest = nearest_blocker(square, dir, blockers);
            let beyond = RAY_TABLES.rays[dir][nearest as usize];
            attacks &= !beyond;
        }
    }
    attacks
}

/// Given a set of blocker bits all lying along `dir` from `square`, find
/// the one closest to `square`.
#[inline]
fn nearest_blocker(_square: u8, dir: usize, blockers: Bitboard) -> u8 {
    match dir {
        DIR_N | DIR_E | DIR_NE | DIR_NW => blockers.trailing_zeros() as u8,
        _ => 63 - blockers.leading_zeros() as u8,
    }
}

// ---------------------------------------------------------------------
// Magic bitboards
//
// Fixed-shift ("plain") magic bitboards for rook/bishop attack lookup:
// index = ((occ & mask) * magic) >> shift, then a single table read.
// This replaces the 4-direction ray-scan loop with one multiply, one
// shift, and one array index per call — rook/bishop attacks are the
// hottest functions in the engine (movegen, attackers_of, check
// detection, eval mobility), so this is the highest-leverage classical
// speedup available without going to PEXT/BMI2 intrinsics.
//
// The 128 magic numbers themselves are baked in as compile-time constants
// (see MAGIC_NUMBERS below) rather than found by trial-and-error at
// startup. Searching for them by brute force is the expensive part (was
// ~300-400ms of blocking, synchronous rejection-sampling — a real problem
// given this engine runs in WASM and that cost would land on the first
// move of every session). Baking them in leaves only the *cheap* half of
// the old startup work: for each square, walk its ~2^bits occupancy
// subsets once (Carry-Rippler) and drop each precomputed attack set
// straight into its table slot. No candidates are rejected, nothing is
// searched — this is a single deterministic pass per square, on the
// order of low single-digit milliseconds total.
//
// If the ray tables ever change (they shouldn't — they're a fixed
// geometric fact about an 8x8 board), the constants can be regenerated
// with the `regen_magic_numbers` test below.
// ---------------------------------------------------------------------

struct MagicEntry {
    mask: Bitboard,
    magic: u64,
    shift: u32,
    attacks: Vec<Bitboard>,
}

struct MagicTables {
    rook: Vec<MagicEntry>,
    bishop: Vec<MagicEntry>,
}

static MAGIC_TABLES: std::sync::OnceLock<MagicTables> = std::sync::OnceLock::new();

#[inline]
fn magic_tables() -> &'static MagicTables {
    MAGIC_TABLES.get_or_init(build_magic_tables)
}

/// The relevant-occupancy mask for a slider on `square` moving along
/// `dirs`: every square a blocker could occupy that actually changes the
/// attack set. The outermost square in each direction is excluded, since
/// whether it's occupied or not never changes the resulting attack
/// bitboard (the ray already terminates there either way).
fn relevant_mask(square: u8, dirs: &[usize; 4]) -> Bitboard {
    let mut mask: Bitboard = 0;
    let r0 = rank_of(square) as i32;
    let f0 = file_of(square) as i32;
    for &dir in dirs.iter() {
        let (dr, df) = dir_delta(dir);
        let mut r = r0 + dr;
        let mut f = f0 + df;
        let mut squares: Vec<u8> = Vec::new();
        while (0..8).contains(&r) && (0..8).contains(&f) {
            squares.push(sq(r as u8, f as u8));
            r += dr;
            f += df;
        }
        // Drop the farthest square (the edge square) along this ray.
        if squares.len() > 1 {
            for &s in &squares[..squares.len() - 1] {
                mask |= bit(s);
            }
        }
    }
    mask
}

const fn dir_delta(dir: usize) -> (i32, i32) {
    match dir {
        DIR_N => (1, 0),
        DIR_S => (-1, 0),
        DIR_E => (0, 1),
        DIR_W => (0, -1),
        DIR_NE => (1, 1),
        DIR_NW => (1, -1),
        DIR_SE => (-1, 1),
        DIR_SW => (-1, -1),
        _ => (0, 0),
    }
}

/// Pre-searched fixed-shift magic multipliers, one per square, generated
/// offline by exhaustively-verified random search (see `regen_magic_numbers`
/// below) and baked in here so startup never has to search for them.
///
/// Regenerate with: `cargo test regen_magic_numbers -- --ignored --nocapture`
#[rustfmt::skip]
const ROOK_MAGICS: [u64; 64] = [
    0x1080004008801020, 0x0840092002C03000, 0x1900200010400900, 0x0880100008000480,
    0x4200100420080200, 0x8100020100080400, 0x0200040110886200, 0x0200008040220411,
    0x0404800084400220, 0x0000401000402000, 0x0086001081220440, 0x0408800800100280,
    0x000A001201040820, 0x8848800200840080, 0x4001000100040200, 0x0442000102105084,
    0x9080010020804100, 0x0040404000201009, 0x0000808010002009, 0x2200090021D00100,
    0x0008008008040080, 0x0004004002010040, 0x0011040008015042, 0x00000A0001768104,
    0x0000800080204009, 0x2010004140002001, 0x9800200280100080, 0x1000100080080080,
    0x0050500500080100, 0x0000020080040080, 0x0C10010400420810, 0x1040008200005104,
    0x01808240088004A0, 0x0882804004802000, 0x0880402001001100, 0x2000210409001000,
    0x2000480131001500, 0x0000800400800200, 0x000002380C001003, 0x4600084882000431,
    0x0080002000504000, 0x0300500020004002, 0x0040408200220011, 0x0010040008004040,
    0x0000080004008080, 0x0010040002008080, 0x2012004881020004, 0x8300842444820011,
    0x0088403882010200, 0x0820400080210100, 0x0110910040A00300, 0x0801100280080480,
    0x0242009008200600, 0x1002000489500200, 0x0040800200010080, 0x0091800041000080,
    0x0000209300488001, 0x04C1002414824001, 0x020020000B001041, 0x7000100004200901,
    0x8002002004100802, 0x30010002084C0007, 0x0888221800813004, 0x4000002840840112,
];
#[rustfmt::skip]
const BISHOP_MAGICS: [u64; 64] = [
    0x20C0090901061081, 0x0024040094030104, 0x8210810200290200, 0x0011040484620000,
    0x0081104002221000, 0x0009012011001350, 0x0081010802400380, 0x0000420210010408,
    0x0008105002280050, 0x0001028484040044, 0x2A00880810408804, 0x7020022282000100,
    0x0084040420100A50, 0x000401010840E000, 0x2020020210420888, 0x0008084202012010,
    0x2010400810018800, 0x0445122008020840, 0x0804100808002008, 0x0008002104110100,
    0x0061005820080800, 0x2001000200820100, 0x480C210084010800, 0x3004442500480420,
    0x1010102240048100, 0x00182009084220A3, 0x8803090A10004205, 0x0208080040202020,
    0x000C044084010040, 0x00A1010002004106, 0x6008210020640202, 0x1600902112860801,
    0x00042008C1220200, 0x010C042002440140, 0x5022080200040820, 0x0402004042940100,
    0x0860108400008020, 0x000C080022021000, 0x0264080652822100, 0x4005031221010401,
    0x0004502410008400, 0x000500B010A20400, 0x0415094050080800, 0x080000201800A104,
    0x4022A80304000110, 0x4012140802028020, 0x40200104010100A0, 0x12810806008B0C41,
    0x0020441008080000, 0x2002120084045420, 0x0704020062080002, 0x0000001084040001,
    0x0322200891240200, 0xF040200210024800, 0x0140824832008042, 0x000210020A004602,
    0x0083042805141020, 0x002C12009A011000, 0x0041A00044140400, 0x00004004020A0202,
    0x0000140010020210, 0x2864160811012200, 0x2060080841082A17, 0xA010041108003100,
];

/// Build one square's attack table from its baked-in magic number. No
/// search happens here: `magic` is already known-good, so this is a
/// single Carry-Rippler pass over the ~2^bits occupancy subsets (at most
/// 4096 for a rook on an open file/rank, far fewer on average), each
/// costing one reference ray-scan and one store. In debug builds this
/// also verifies there are no index collisions, so a stale constant (e.g.
/// after an edit to the ray tables) fails loudly in tests instead of
/// silently producing wrong attacks.
fn build_magic_entry(square: u8, dirs: &[usize; 4], magic: u64) -> MagicEntry {
    let mask = relevant_mask(square, dirs);
    let bits = popcount(mask);
    let shift = 64 - bits;
    let size = 1usize << bits;

    let mut attacks = vec![0 as Bitboard; size];
    #[cfg(debug_assertions)]
    let mut filled = vec![false; size];

    let mut subset: Bitboard = 0;
    loop {
        let idx = ((subset.wrapping_mul(magic)) >> shift) as usize;
        let att = sliding_attacks_ray_scan(square, subset, dirs);

        #[cfg(debug_assertions)]
        {
            if filled[idx] {
                debug_assert_eq!(
                    attacks[idx], att,
                    "stale/invalid baked magic for square {square}: index collision"
                );
            }
            filled[idx] = true;
        }
        attacks[idx] = att;

        subset = subset.wrapping_sub(mask) & mask;
        if subset == 0 {
            break;
        }
    }

    MagicEntry {
        mask,
        magic,
        shift,
        attacks,
    }
}

fn build_magic_tables() -> MagicTables {
    let mut rook = Vec::with_capacity(64);
    let mut bishop = Vec::with_capacity(64);
    for s in 0..64u8 {
        rook.push(build_magic_entry(s, &ROOK_DIRS, ROOK_MAGICS[s as usize]));
        bishop.push(build_magic_entry(s, &BISHOP_DIRS, BISHOP_MAGICS[s as usize]));
    }

    MagicTables { rook, bishop }
}

#[inline(always)]
pub fn rook_attacks(square: u8, occ: Bitboard) -> Bitboard {
    let e = &magic_tables().rook[square as usize];
    let idx = (((occ & e.mask).wrapping_mul(e.magic)) >> e.shift) as usize;
    e.attacks[idx]
}

#[inline(always)]
pub fn bishop_attacks(square: u8, occ: Bitboard) -> Bitboard {
    let e = &magic_tables().bishop[square as usize];
    let idx = (((occ & e.mask).wrapping_mul(e.magic)) >> e.shift) as usize;
    e.attacks[idx]
}

#[inline(always)]
pub fn queen_attacks(square: u8, occ: Bitboard) -> Bitboard {
    rook_attacks(square, occ) | bishop_attacks(square, occ)
}

#[inline(always)]
pub fn knight_attacks(square: u8) -> Bitboard {
    RAY_TABLES.knight[square as usize]
}

#[inline(always)]
pub fn king_attacks(square: u8) -> Bitboard {
    RAY_TABLES.king[square as usize]
}

#[inline(always)]
pub fn pawn_attacks(color_is_white: bool, square: u8) -> Bitboard {
    RAY_TABLES.pawn_attacks[if color_is_white { 0 } else { 1 }][square as usize]
}

/// Squares strictly between `a` and `b` along a straight/diagonal line, or
/// 0 if they are not aligned. Used for check-block masks.
pub fn ray_between(a: u8, b: u8) -> Bitboard {
    for &dir in ROOK_DIRS.iter().chain(BISHOP_DIRS.iter()) {
        let ray = RAY_TABLES.rays[dir][a as usize];
        if ray & bit(b) != 0 {
            // ray from a in this dir includes b; the "between" squares are
            // the ray from a up to (excluding) b.
            let beyond_b = RAY_TABLES.rays[dir][b as usize];
            return ray & !beyond_b & !bit(b);
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knight_center() {
        // Knight on d4 (rank3,file3 -> sq = 3*8+3=27) has 8 moves
        let s = sq(3, 3);
        assert_eq!(popcount(knight_attacks(s)), 8);
    }

    #[test]
    fn rook_open_board() {
        let s = sq(0, 0); // a1
        let attacks = rook_attacks(s, EMPTY);
        assert_eq!(popcount(attacks), 14);
    }

    #[test]
    fn rook_blocked() {
        let s = sq(0, 0); // a1
        let occ = bit(sq(0, 3)) | bit(sq(3, 0)); // d1, a4
        let attacks = rook_attacks(s, occ);
        // Should include up to and including blockers, not beyond
        assert!(attacks & bit(sq(0, 3)) != 0);
        assert!(attacks & bit(sq(0, 4)) == 0);
        assert!(attacks & bit(sq(3, 0)) != 0);
        assert!(attacks & bit(sq(4, 0)) == 0);
    }

    #[test]
    fn ray_between_works() {
        let a = sq(0, 0);
        let b = sq(0, 4);
        let between = ray_between(a, b);
        assert_eq!(between, bit(sq(0, 1)) | bit(sq(0, 2)) | bit(sq(0, 3)));
    }

    /// Cross-check every baked magic-table lookup against the reference
    /// ray-scan implementation across a spread of occupancies, so a bad
    /// constant (typo, stale value after a ray-table edit, etc.) is caught
    /// in CI rather than producing silently wrong attacks at runtime.
    #[test]
    fn magic_matches_ray_scan_reference() {
        let mut occs: Vec<Bitboard> = vec![EMPTY, ALL];
        // A spread of pseudo-random occupancies, deterministic across runs.
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        for _ in 0..2000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            occs.push(x);
        }

        for s in 0..64u8 {
            for &occ in &occs {
                assert_eq!(
                    rook_attacks(s, occ),
                    sliding_attacks_ray_scan(s, occ, &ROOK_DIRS),
                    "rook magic mismatch on square {s}, occ {occ:#018x}"
                );
                assert_eq!(
                    bishop_attacks(s, occ),
                    sliding_attacks_ray_scan(s, occ, &BISHOP_DIRS),
                    "bishop magic mismatch on square {s}, occ {occ:#018x}"
                );
            }
        }
    }

    /// Not run by default (it's a brute-force search, same cost as the old
    /// startup path). Regenerates ROOK_MAGICS/BISHOP_MAGICS from scratch —
    /// only needed if the ray tables/board geometry ever change. Run with:
    /// `cargo test regen_magic_numbers -- --ignored --nocapture`
    /// and paste the printed arrays back into the constants above.
    #[test]
    #[ignore]
    fn regen_magic_numbers() {
        struct Rng(u64);
        impl Rng {
            fn new(seed: u64) -> Self {
                Rng(seed | 1)
            }
            fn next_u64(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                self.0 = x;
                x.wrapping_mul(0x2545_F491_4F6C_DD1D)
            }
            fn sparse_u64(&mut self) -> u64 {
                self.next_u64() & self.next_u64() & self.next_u64()
            }
        }

        fn find_magic(square: u8, dirs: &[usize; 4], rng: &mut Rng) -> u64 {
            let mask = relevant_mask(square, dirs);
            let bits = popcount(mask);
            let shift = 64 - bits;
            let size = 1usize << bits;

            let mut occs: Vec<Bitboard> = Vec::with_capacity(size);
            let mut atts: Vec<Bitboard> = Vec::with_capacity(size);
            let mut subset: Bitboard = 0;
            loop {
                occs.push(subset);
                atts.push(sliding_attacks_ray_scan(square, subset, dirs));
                subset = subset.wrapping_sub(mask) & mask;
                if subset == 0 {
                    break;
                }
            }

            struct Slot {
                epoch: u32,
                value: Bitboard,
            }
            let mut table: Vec<Slot> =
                (0..size).map(|_| Slot { epoch: 0, value: 0 }).collect();
            let mut epoch: u32 = 0;

            loop {
                let magic = rng.sparse_u64();
                if popcount(mask.wrapping_mul(magic) & 0xFF00_0000_0000_0000) < 6 {
                    continue;
                }
                epoch += 1;
                let mut ok = true;
                for i in 0..occs.len() {
                    let idx = ((occs[i].wrapping_mul(magic)) >> shift) as usize;
                    let slot = &mut table[idx];
                    if slot.epoch != epoch {
                        slot.epoch = epoch;
                        slot.value = atts[i];
                    } else if slot.value != atts[i] {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    return magic;
                }
            }
        }

        let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
        let mut rook = Vec::with_capacity(64);
        let mut bishop = Vec::with_capacity(64);
        for s in 0..64u8 {
            rook.push(find_magic(s, &ROOK_DIRS, &mut rng));
            bishop.push(find_magic(s, &BISHOP_DIRS, &mut rng));
        }

        println!("const ROOK_MAGICS: [u64; 64] = [");
        for chunk in rook.chunks(4) {
            let line: Vec<String> = chunk.iter().map(|m| format!("0x{:016X}", m)).collect();
            println!("    {},", line.join(", "));
        }
        println!("];");
        println!("const BISHOP_MAGICS: [u64; 64] = [");
        for chunk in bishop.chunks(4) {
            let line: Vec<String> = chunk.iter().map(|m| format!("0x{:016X}", m)).collect();
            println!("    {},", line.join(", "));
        }
        println!("];");
    }
}
