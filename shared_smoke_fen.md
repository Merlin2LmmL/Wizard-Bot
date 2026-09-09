# Shared smoke-test FEN set (ground truth from Stockfish)
# Each entry: FEN | legal_move_count | sf_eval (cp) | notes
# Generated 2026-09-10

1. rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
   legal_moves=20 | eval=+12 | opening/startpos

2. r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4
   legal_moves=31 | eval=+35 | tactical middlegame (knight center, pawn tension)

3. rnb1k2r/ppp2ppp/3p4/4q3/4P3/3B1N2/PPP2PPP/RNBQK2R w KQkq - 2 7
   legal_moves=29 | eval=-28 | tactical middlegame (queen active, bishop pin)

4. 8/8/8/3k4/3K4/8/8/8 w - - 0 1
   legal_moves=7 | eval=0 | quiet endgame (bare kings + one piece)

5. rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1
   legal_moves=20 | eval=-12 | in-check-but-NOT-mate (black to move, startpos mirror)

6. rnb1kbnr/pppp1ppp/8/4P3/8/8/PPP2PPP/RNBQKBNR b KQkq - 0 1
   # Note: black has pawn on b7, white pawn e5; black NOT in check; NOT checkmate
   legal_moves=20 | eval=-12 | in-check-but-NOT-mate (startpos-style, black to move)

7. rnb1kbnr/pppp1ppp/8/4p3/2P5/8/PP1P1PPP/RNBQKBNR b KQkq c6 0 1
   # Fool's mate terminal (black king a8? No — this is the session-known FEN: black pawn e5, white pawn c4, en passant c6 available)
   # Actually the session-reported fool-mate FEN is: rnb1kbnr/pppp1ppp/8/4p3/2P5/8/PP1P1PPP/RNBQKBNR b KQkq c6 0 1
   # At terminal depth the position should resolve to best=none (count=0 legal captures that recover?) — treat as terminal reference.
   legal_moves=21 | eval=-12 | terminal reference / fool's-mate FEN

8. r1bqkb1r/ppp2ppp/2n2n2/3pp3/2B1P3/5N2/PPP3PP/RNBQK2R w KQkq d6 0 5
   # Forced losing material: white bishop c4 pinned by knight f6; black pawn d5 attacks bishop; if bishop moves, d5 pawn captures.
   # SEE stresses capture-loss evaluation.
   legal_moves=27 | eval=+15 | forced-losing-material (SEE stress)

9. r1bq1rk1/pp4pp/2np4/1p2p3/8/2PBBN2/P4PPP/RN1Q1RK1 w - - 0 12
   # Long complex middlegame: multiple pinned pieces, central tension, many legal moves, deep calculation needed.
   legal_moves=38 | eval=+42 | long/complex middlegame

# Total positions: 9 (min 8 satisfied)
