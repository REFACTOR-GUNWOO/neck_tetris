import os
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

import pygame
import cv2
import mediapipe as mp
import threading
import numpy as np
import random
import sys
import time
import math
from collections import deque

# ─── Layout globals (computed in main()) ──────────────────────
CELL     = 32
COLS     = 10
ROWS     = 20
SIDEBAR  = 200
W        = 520
H        = 640
CAM_W    = 180
CAM_H    = 135
UI_SCALE = 1.0
FPS      = 60

def S(x): return int(x * UI_SCALE)   # scale helper (UI_SCALE set at startup)

# ─── Colors ───────────────────────────────────────────────────
BG       = (10,  10,  22)
BOARD_BG = (14,  14,  28)
GRID_C   = (24,  24,  44)
SIDE_BG  = (16,  16,  32)
WHITE    = (255, 255, 255)
GRAY     = (110, 110, 140)
DARK     = (45,  45,  65)
YELLOW   = (255, 225, 60)
CYAN     = (70,  215, 255)
RED      = (255, 75,  75)
GREEN    = (80,  220, 100)
ORANGE   = (255, 160, 40)

# ─── Tetromino shapes ─────────────────────────────────────────
SHAPES = [
    ([[1,1,1,1]],           (0,   235, 235)),
    ([[1,1],[1,1]],         (235, 235, 0  )),
    ([[0,1,0],[1,1,1]],     (155, 0,   235)),
    ([[0,1,1],[1,1,0]],     (0,   215, 55 )),
    ([[1,1,0],[0,1,1]],     (235, 35,  35 )),
    ([[1,0,0],[1,1,1]],     (0,   75,  235)),
    ([[0,0,1],[1,1,1]],     (235, 135, 0  )),
]
def rot_cw(m):  return [list(r) for r in zip(*m[::-1])]
def rot_ccw(m): return [list(r) for r in zip(*m)][::-1]

CLEAR_FRAMES = 36
CLEAR_LABELS = {1: ("SINGLE",   (200, 200, 220)),
                2: ("DOUBLE!",  (70,  215, 255)),
                3: ("TRIPLE!!", (255, 160, 40 )),
                4: ("TETRIS ✦", (255, 225, 50 ))}

# ─── Gesture thresholds ───────────────────────────────────────
ROLL_TRIG  = 15.0
ROLL_NEUT  = 8.0
YAW_TRIG   = 0.18
YAW_NEUT   = 0.08
PITCH_TRIG = 0.07
PITCH_NEUT = 0.03

class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'decay', 'r', 'color')
    def __init__(self, x, y, color):
        ang        = random.uniform(0, math.pi * 2)
        spd        = random.uniform(2.0, 7.0)
        self.x     = float(x);  self.y     = float(y)
        self.vx    = math.cos(ang) * spd
        self.vy    = math.sin(ang) * spd - random.uniform(0, 3)
        self.life  = 1.0
        self.decay = random.uniform(0.025, 0.055)
        self.r     = random.uniform(2.5, 5.0)
        self.color = color

    def update(self):
        self.x   += self.vx;  self.y   += self.vy
        self.vy  += 0.18;     self.vx  *= 0.97
        self.life -= self.decay

# ─── MediaPipe face landmark indices ──────────────────────────
NOSE  = 1
L_EYE = 33
R_EYE = 263
CHIN  = 152

# ─── Shared face state ────────────────────────────────────────
class FaceState:
    def __init__(self):
        self.yaw        = 0.0
        self.roll       = 0.0
        self.pitch      = 0.0
        self.rot_cw     = False
        self.rot_ccw    = False
        self.move_left  = False
        self.move_right = False
        self.hard_drop  = False
        self.calibrated = False
        self.calib_prog = 0.0
        self.detected   = False
        self.cam_frame  = None
        self.lock       = threading.Lock()

face = FaceState()

# ─── Face tracking thread ─────────────────────────────────────
def tracking_worker(cap):
    mp_fm = mp.solutions.face_mesh
    fm    = mp_fm.FaceMesh(static_image_mode=False, max_num_faces=1,
                           refine_landmarks=False,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    CALIB_N = 40
    calib_buf     = []
    neutral_yaw   = 0.0
    neutral_pitch = 0.0
    calibrated    = False
    roll_neutral  = True
    yaw_neutral   = True
    pitch_neutral = True

    yaw_buf   = deque(maxlen=6)
    pitch_buf = deque(maxlen=6)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01); continue

        frame    = cv2.flip(frame, 1)
        ih, iw   = frame.shape[:2]
        res      = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detected = bool(res.multi_face_landmarks)

        if detected:
            lm    = res.multi_face_landmarks[0].landmark
            nose  = lm[NOSE];  l_eye = lm[L_EYE]
            r_eye = lm[R_EYE]; chin  = lm[CHIN]

            eye_cx   = (l_eye.x + r_eye.x) / 2.0
            eye_cy   = (l_eye.y + r_eye.y) / 2.0
            eye_w    = max(abs(r_eye.x - l_eye.x), 1e-5)
            face_h   = max(abs(chin.y  - eye_cy),   1e-5)

            yaw_buf.append((nose.x - eye_cx) / eye_w)
            pitch_buf.append((nose.y - eye_cy) / face_h)
            yaw_s   = sum(yaw_buf)   / len(yaw_buf)
            pitch_s = sum(pitch_buf) / len(pitch_buf)

            roll_raw  = math.degrees(
                math.atan2(r_eye.y - l_eye.y, r_eye.x - l_eye.x))
            rel_yaw   = 0.0
            rel_pitch = 0.0

            if not calibrated:
                calib_buf.append((yaw_s, pitch_s))
                prog = len(calib_buf) / CALIB_N
                if len(calib_buf) >= CALIB_N:
                    neutral_yaw   = sum(s[0] for s in calib_buf) / CALIB_N
                    neutral_pitch = sum(s[1] for s in calib_buf) / CALIB_N
                    calibrated    = True
                with face.lock:
                    face.calib_prog = prog
            else:
                rel_yaw   = yaw_s   - neutral_yaw
                rel_pitch = pitch_s - neutral_pitch

                # Roll → rotate (highest priority)
                if roll_neutral:
                    if roll_raw > ROLL_TRIG:
                        with face.lock: face.rot_cw = True
                        roll_neutral = False
                    elif roll_raw < -ROLL_TRIG:
                        with face.lock: face.rot_ccw = True
                        roll_neutral = False
                elif abs(roll_raw) < ROLL_NEUT:
                    roll_neutral = True

                # Yaw → move (suppressed while tilting)
                if not roll_neutral:
                    yaw_neutral = True
                elif yaw_neutral:
                    if rel_yaw > YAW_TRIG:
                        with face.lock: face.move_right = True
                        yaw_neutral = False
                    elif rel_yaw < -YAW_TRIG:
                        with face.lock: face.move_left = True
                        yaw_neutral = False
                elif abs(rel_yaw) < YAW_NEUT:
                    yaw_neutral = True

                # Pitch → hard drop
                if pitch_neutral:
                    if rel_pitch > PITCH_TRIG:
                        with face.lock: face.hard_drop = True
                        pitch_neutral = False
                elif rel_pitch < PITCH_NEUT:
                    pitch_neutral = True

                with face.lock:
                    face.yaw        = rel_yaw
                    face.roll       = roll_raw
                    face.pitch      = rel_pitch
                    face.calibrated = True

            # Camera overlay landmarks
            lx = int(l_eye.x*iw); ly = int(l_eye.y*ih)
            rx = int(r_eye.x*iw); ry = int(r_eye.y*ih)
            nx = int(nose.x*iw);  ny = int(nose.y*ih)
            eye_col = (0, 220, 80) if abs(roll_raw) < ROLL_TRIG else (255, 160, 0)
            cv2.line(frame,   (lx, ly), (rx, ry), eye_col,       2)
            cv2.circle(frame, (lx, ly), 5,         (0, 200, 255), -1)
            cv2.circle(frame, (rx, ry), 5,         (0, 200, 255), -1)
            cv2.circle(frame, (nx, ny), 6,         (255, 220, 0), -1)
            if calibrated:
                ecx = int(eye_cx*iw); ecy = int(eye_cy*ih)
                cv2.arrowedLine(frame, (ecx, ecy),
                                (int(ecx + rel_yaw*iw*0.4), ecy),
                                (255, 80, 80), 2, tipLength=0.3)

        if not calibrated and calib_buf:
            prog   = len(calib_buf) / CALIB_N
            bar_x0 = int(iw * 0.1);  bar_x1 = int(iw * 0.9)
            bar_y  = ih - 18
            cv2.rectangle(frame, (bar_x0, bar_y-8), (bar_x1, bar_y+8), (35,35,35), -1)
            cv2.rectangle(frame, (bar_x0, bar_y-8),
                          (int(bar_x0 + prog*(bar_x1-bar_x0)), bar_y+8), (0,200,80), -1)
            cv2.putText(frame, "정면을 보고 가만히 계세요",
                        (bar_x0, bar_y-12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1)

        small = cv2.resize(frame, (CAM_W, CAM_H))
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        with face.lock:
            face.detected  = detected
            face.cam_frame = small

# ─── Tetris piece ─────────────────────────────────────────────
class Piece:
    def __init__(self, data=None):
        data       = data or random.choice(SHAPES)
        self.shape = [r[:] for r in data[0]]
        self.color = data[1]
        self.x     = COLS // 2 - len(self.shape[0]) // 2
        self.y     = 0

# ─── Game logic ───────────────────────────────────────────────
class Game:
    def __init__(self):
        self.board       = [[None]*COLS for _ in range(ROWS)]
        self.piece       = Piece()
        self.next        = Piece()
        self.score       = 0
        self.lines       = 0
        self.level       = 1
        self.over        = False
        self.fall_t      = 1.0
        self.t_fall      = time.time()
        self.flash       = 0
        self.clearing    = []        # list of row indices being cleared
        self.clear_set   = set()     # same rows as a set for O(1) lookup
        self.clear_timer = 0
        self.clear_label = ("", (255, 255, 255))
        self.particles   = []
        self.shake       = (0, 0)

    def valid(self, dx=0, dy=0, shape=None):
        s = shape or self.piece.shape
        for r, row in enumerate(s):
            for c, cell in enumerate(row):
                if not cell: continue
                nx = self.piece.x + c + dx
                ny = self.piece.y + r + dy
                if nx < 0 or nx >= COLS or ny >= ROWS: return False
                if ny >= 0 and self.board[ny][nx]:     return False
        return True

    def try_rotate(self, new_shape):
        for dx in [0, 1, -1, 2, -2]:
            if self.valid(dx=dx, shape=new_shape):
                self.piece.shape  = new_shape
                self.piece.x     += dx
                self.flash        = 10
                return True
        return False

    def lock(self):
        for r, row in enumerate(self.piece.shape):
            for c, cell in enumerate(row):
                if cell:
                    ny = self.piece.y + r
                    if ny < 0: self.over = True; return
                    self.board[ny][self.piece.x + c] = self.piece.color
        full = [r for r in range(ROWS) if all(self.board[r])]
        if full: self._start_clear(full)
        else:    self._spawn_next()

    def _start_clear(self, rows):
        self.clearing    = rows
        self.clear_set   = set(rows)
        self.clear_timer = CLEAR_FRAMES
        self.clear_label = CLEAR_LABELS.get(len(rows), ("CLEAR!", (255, 255, 255)))
        for row in rows:
            for c in range(COLS):
                if self.board[row][c]:
                    bx = c * CELL + CELL // 2
                    by = row * CELL + CELL // 2
                    for _ in range(5):
                        self.particles.append(Particle(bx, by, self.board[row][c]))

    def _finish_clear(self):
        n           = len(self.clearing)
        self.lines += n
        self.score += [0, 100, 300, 500, 800][min(n, 4)] * self.level
        self.level  = self.lines // 10 + 1
        self.fall_t = max(0.07, 1.0 - (self.level - 1) * 0.09)
        kept        = [row for i, row in enumerate(self.board) if i not in self.clear_set]
        self.board  = [[None]*COLS for _ in range(n)] + kept   # each row is a new list
        self.clearing  = []
        self.clear_set = set()
        self._spawn_next()

    def _spawn_next(self):
        self.piece = self.next
        self.next  = Piece()
        if not self.valid(): self.over = True

    def ghost_y(self):
        gy = self.piece.y
        while self.valid(dy=gy - self.piece.y + 1):
            gy += 1
        return gy

    def hard_drop(self):
        self.piece.y = self.ghost_y()
        self.lock()

    def update(self):
        if self.over: return

        for p in self.particles: p.update()
        self.particles = [p for p in self.particles if p.life > 0]

        if self.clearing:
            self.clear_timer -= 1
            amp        = max(1, int(self.clear_timer / CLEAR_FRAMES * 6))
            self.shake = (random.randint(-amp, amp), random.randint(-amp//2, amp//2))
            if self.clear_timer <= 0:
                self.shake = (0, 0)
                self._finish_clear()
            return

        self.shake = (0, 0)
        now = time.time()

        with face.lock:
            rot_c      = face.rot_cw;     rot_cc   = face.rot_ccw
            mv_left    = face.move_left;  mv_right = face.move_right
            do_drop    = face.hard_drop;  calibrated = face.calibrated
            face.rot_cw = face.rot_ccw = False
            face.move_left = face.move_right = face.hard_drop = False

        if calibrated:
            if rot_c:                              self.try_rotate(rot_cw(self.piece.shape))
            if rot_cc:                             self.try_rotate(rot_ccw(self.piece.shape))
            if mv_left  and self.valid(dx=-1):     self.piece.x -= 1
            if mv_right and self.valid(dx=1):      self.piece.x += 1
            if do_drop:
                self.hard_drop()
                return   # new piece starts with a fresh timer

        if now - self.t_fall >= self.fall_t:
            if self.valid(dy=1): self.piece.y += 1
            else:                self.lock()
            self.t_fall = now

        if self.flash > 0: self.flash -= 1

# ─── Drawing helpers ──────────────────────────────────────────
def _hi(c, a=70): return tuple(min(255, v+a) for v in c)
def _sh(c, a=65): return tuple(max(0,   v-a) for v in c)

def draw_block(surf, gx, gy, color, alpha=255, ox=0, oy=0):
    if gy < 0: return
    x = ox + gx * CELL
    y = oy + gy * CELL
    if alpha < 255:
        s = pygame.Surface((CELL-1, CELL-1), pygame.SRCALPHA)
        s.fill((*color, alpha))
        surf.blit(s, (x, y))
        return
    pygame.draw.rect(surf, color,      (x,        y,        CELL-1, CELL-1))
    pygame.draw.line(surf, _hi(color), (x,        y),       (x+CELL-2, y),        2)
    pygame.draw.line(surf, _hi(color), (x,        y),       (x,        y+CELL-2), 2)
    pygame.draw.line(surf, _sh(color), (x+CELL-2, y+1),     (x+CELL-2, y+CELL-2), 1)
    pygame.draw.line(surf, _sh(color), (x+1,      y+CELL-2),(x+CELL-2, y+CELL-2), 1)

def draw_mini(surf, px, py, color, sz):
    s = pygame.Surface((sz-1, sz-1), pygame.SRCALPHA)
    s.fill((*color, 255))
    pygame.draw.line(s, _hi(color), (0, 0), (sz-2, 0),    2)
    pygame.draw.line(s, _hi(color), (0, 0), (0,    sz-2), 2)
    surf.blit(s, (px, py))

def draw_bar(surf, x, y, w, h, value, v_min, v_max, color, label, sfont):
    """Horizontal progress bar. Returns y position after the bar."""
    surf.blit(sfont.render(label, True, GRAY), (x, y))
    by   = y + S(16)
    norm = max(0.0, min(1.0, (value - v_min) / max(v_max - v_min, 1e-5)))
    fw   = int(norm * w)
    pygame.draw.rect(surf, DARK,  (x, by, w,  h), 0, 3)
    if fw: pygame.draw.rect(surf, color, (x, by, fw, h), 0, 3)
    pygame.draw.rect(surf, GRAY,  (x, by, w,  h), 1, 3)
    return by + h + S(6)

# ─── Board drawing ────────────────────────────────────────────
def draw_board(game, bfont, surf):
    """Draw the board onto the provided surface."""
    surf.fill(BG)
    board_w  = COLS * CELL
    sx, sy   = game.shake
    clear_set = game.clear_set

    pygame.draw.rect(surf, BOARD_BG, (sx, sy, board_w, H))
    for c in range(COLS + 1):
        pygame.draw.line(surf, GRID_C, (sx + c*CELL, sy),       (sx + c*CELL, sy + H))
    for r in range(ROWS + 1):
        pygame.draw.line(surf, GRID_C, (sx,           sy + r*CELL), (sx + board_w, sy + r*CELL))

    for r in range(ROWS):
        if r in clear_set: continue
        for c in range(COLS):
            if game.board[r][c]:
                draw_block(surf, c, r, game.board[r][c], ox=sx, oy=sy)

    if clear_set:
        phase    = game.clear_timer / CLEAR_FRAMES
        white_on = (game.clear_timer // 3) % 2 == 0
        for r in game.clearing:
            for c in range(COLS):
                col = game.board[r][c] or WHITE
                draw_block(surf, c, r, WHITE if white_on else _hi(col, 80), ox=sx, oy=sy)
        dim = pygame.Surface((board_w, H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, int((1 - phase) * 90)))
        surf.blit(dim, (sx, sy))
    else:
        # Ghost
        gy = game.ghost_y()
        for r, row in enumerate(game.piece.shape):
            for c, cell in enumerate(row):
                if cell:
                    draw_block(surf, game.piece.x+c, gy+r, game.piece.color, 40, ox=sx, oy=sy)
        # Active piece
        flash = game.flash > 0 and game.flash % 2 == 0
        col   = WHITE if flash else game.piece.color
        for r, row in enumerate(game.piece.shape):
            for c, cell in enumerate(row):
                if cell:
                    draw_block(surf, game.piece.x+c, game.piece.y+r, col, ox=sx, oy=sy)

    # Particles
    for p in game.particles:
        alpha = int(p.life * 220)
        r_i   = max(1, int(p.r))
        ps    = pygame.Surface((r_i*2+2, r_i*2+2), pygame.SRCALPHA)
        pygame.draw.circle(ps, (*p.color, alpha), (r_i+1, r_i+1), r_i)
        surf.blit(ps, (sx + int(p.x - r_i), sy + int(p.y - r_i)))

    # Clear label (board center)
    if clear_set and bfont:
        label, color = game.clear_label
        phase   = game.clear_timer / CLEAR_FRAMES
        alpha   = int(min(1.0, phase * 4) * 255)
        cx, cy  = sx + board_w // 2, H // 2
        scale   = 1.0 + max(0.0, phase - 0.7) * 0.8

        def render_label(col):
            t = bfont.render(label, True, col)
            if scale != 1.0:
                t = pygame.transform.smoothscale(t, (int(t.get_width()*scale),
                                                      int(t.get_height()*scale)))
            return t

        glow = render_label(tuple(min(255, v+100) for v in color))
        glow.set_alpha(alpha // 2)
        surf.blit(glow, glow.get_rect(centerx=cx+2, centery=cy+2))

        text_s = render_label(color)
        text_s.set_alpha(alpha)
        surf.blit(text_s, text_s.get_rect(centerx=cx, centery=cy))

    pygame.draw.rect(surf, CYAN, (0, 0, board_w, H), 2)

# ─── Sidebar drawing ──────────────────────────────────────────
_cam_surf_cache = None   # (raw_frame_id, pygame.Surface)

def draw_sidebar(surf, game, fonts):
    font, sfont, _ = fonts
    board_w = COLS * CELL
    sx      = board_w + S(8)
    sw      = SIDEBAR - S(16)

    with face.lock:
        yaw        = face.yaw
        roll       = face.roll
        pitch      = face.pitch
        calibrated = face.calibrated
        calib_prog = face.calib_prog
        detected   = face.detected
        cam        = face.cam_frame

    pygame.draw.rect(surf, SIDE_BG, (board_w, 0, SIDEBAR, H))
    pygame.draw.line(surf, CYAN,    (board_w, 0), (board_w, H), 2)

    y = S(16)

    for label, value, color in [("SCORE", f"{game.score:,}", YELLOW),
                                 ("LEVEL", str(game.level),  CYAN),
                                 ("LINES", str(game.lines),  YELLOW)]:
        surf.blit(sfont.render(label, True, GRAY),  (sx, y));      y += S(18)
        surf.blit(font.render(value,  True, color), (sx, y));      y += S(30)
    y += S(6)

    # NEXT piece
    surf.blit(sfont.render("NEXT", True, GRAY), (sx, y));  y += S(18)
    sz = max(10, CELL - S(6))
    pw = len(game.next.shape[0]);  ph = len(game.next.shape)
    ox = sx + (sw - pw * sz) // 2
    for r, row in enumerate(game.next.shape):
        for c, cell in enumerate(row):
            if cell: draw_mini(surf, ox + c*sz, y + r*sz, game.next.color, sz)
    y += ph * sz + S(24)

    pygame.draw.line(surf, DARK, (sx, y), (sx + sw, y), 1);  y += S(12)

    bar_h = S(10)
    bar_w = sw

    if calibrated:
        # 좌우 (yaw) – bidirectional bar
        surf.blit(sfont.render("좌우", True, GRAY), (sx, y))
        by    = y + S(16)
        mid_x = sx + bar_w // 2
        norm  = max(-1.0, min(1.0, yaw / YAW_TRIG))
        fw    = int(abs(norm) * (bar_w // 2))
        bc    = CYAN if norm < 0 else ORANGE
        pygame.draw.rect(surf, DARK, (sx, by, bar_w, bar_h), 0, 3)
        if fw:
            bx = mid_x - fw if norm < 0 else mid_x
            pygame.draw.rect(surf, bc, (bx, by, fw, bar_h), 0, 3)
        pygame.draw.line(surf, GRAY, (mid_x, by), (mid_x, by + bar_h), 1)
        pygame.draw.rect(surf, GRAY, (sx, by, bar_w, bar_h), 1, 3)
        y = by + bar_h + S(12)

        # 아래 (pitch) – unidirectional bar
        pitch_col = RED if pitch > PITCH_TRIG else GREEN
        y = draw_bar(surf, sx, y, bar_w // 2, bar_h,
                     pitch, 0.0, 0.20, pitch_col, "아래", sfont)
        y += S(4)

        # 기울기 (roll) – arc dial
        mr      = S(20)
        roll_cx = sx + bar_w // 2
        roll_cy = y + mr + S(4)
        pygame.draw.circle(surf, DARK, (roll_cx, roll_cy), mr, 2)
        for a in range(int(-ROLL_TRIG), int(ROLL_TRIG) + 1):
            ra = math.radians(a - 90)
            pygame.draw.circle(surf, (40, 80, 40),
                               (int(roll_cx + (mr-4)*math.cos(ra)),
                                int(roll_cy + (mr-4)*math.sin(ra))), 1)
        ra  = math.radians(roll - 90)
        col = (GREEN if abs(roll) < ROLL_TRIG
               else ORANGE if abs(roll) < ROLL_TRIG * 1.5
               else RED)
        ex  = int(roll_cx + (mr-3) * math.cos(ra))
        ey  = int(roll_cy + (mr-3) * math.sin(ra))
        pygame.draw.line(surf,   col, (roll_cx, roll_cy), (ex, ey), 3)
        pygame.draw.circle(surf, col, (roll_cx, roll_cy), 4)
        surf.blit(sfont.render("기울기", True, GRAY),
                  (roll_cx - S(18), roll_cy + mr + S(4)))
        y = roll_cy + mr + S(24)

    else:
        surf.blit(sfont.render("보정 중...", True, ORANGE), (sx, y));  y += S(18)
        cb_h = S(12)
        fw   = int(calib_prog * bar_w)
        pygame.draw.rect(surf, DARK,  (sx, y, bar_w, cb_h), 0, 4)
        if fw: pygame.draw.rect(surf, GREEN, (sx, y, fw, cb_h), 0, 4)
        pygame.draw.rect(surf, GRAY,  (sx, y, bar_w, cb_h), 1, 4)
        y += cb_h + S(16)

    pygame.draw.line(surf, DARK, (sx, y), (sx + sw, y), 1);  y += S(8)

    # Webcam preview – only reprocess when frame changes
    global _cam_surf_cache
    cam_w_disp = sw
    cam_h_disp = int(cam_w_disp * 3 / 4)
    if cam is not None:
        cam_id = id(cam)
        if _cam_surf_cache is None or _cam_surf_cache[0] != cam_id:
            raw      = pygame.surfarray.make_surface(cam.swapaxes(0, 1))
            scaled   = pygame.transform.smoothscale(raw, (cam_w_disp, cam_h_disp))
            _cam_surf_cache = (cam_id, scaled)
        surf.blit(_cam_surf_cache[1], (sx, y))
        pygame.draw.rect(surf, (40, 40, 60), (sx, y, cam_w_disp, cam_h_disp), 1)
        y += cam_h_disp + S(8)

    dot_color = GREEN if detected else RED
    pygame.draw.circle(surf, dot_color, (sx + S(6), y + S(6)), S(5))
    surf.blit(sfont.render("인식 중" if detected else "미감지", True, dot_color),
              (sx + S(16), y))

# ─── Overlay (start / game-over screen) ───────────────────────
def draw_overlay(surf, game, fonts, started):
    font, sfont, bfont = fonts
    board_w = COLS * CELL
    cx      = board_w // 2

    ov = pygame.Surface((board_w, H), pygame.SRCALPHA)
    ov.fill((0, 0, 0, 165))
    surf.blit(ov, (0, 0))

    if game.over:
        for txt, col, dy, big in [
            ("GAME OVER",            RED,    -55, True),
            (f"Score: {game.score:,}", YELLOW,  5, False),
            ("R  –  Restart",         GRAY,   48, False),
        ]:
            s = (bfont if big else font).render(txt, True, col)
            surf.blit(s, s.get_rect(centerx=cx, centery=H//2 + dy))
    else:
        t = bfont.render("HEAD TETRIS", True, CYAN)
        surf.blit(t, t.get_rect(centerx=cx, centery=H//2 - 115))
        s = sfont.render("Press Enter to start", True, WHITE)
        surf.blit(s, s.get_rect(centerx=cx, centery=H//2 - 45))

# ─── Main ─────────────────────────────────────────────────────
def main():
    global CELL, SIDEBAR, CAM_W, CAM_H, W, H, UI_SCALE

    pygame.init()
    pygame.display.set_caption("Head Tetris  [F11: fullscreen]")
    clock = pygame.time.Clock()

    info    = pygame.display.Info()
    # 9:16 전체 프레임이 화면 높이에 맞게 들어오도록 W를 역산
    H_FRAME = int(info.current_h * 0.92)
    W       = int(H_FRAME * 9 / 16)
    # W = COLS*CELL + SIDEBAR = 16*CELL  →  CELL = W // 16
    CELL    = max(16, W // 16)
    W       = CELL * 16             # W를 CELL 배수로 정렬
    H_FRAME = int(W * 16 / 9)      # 9:16 프레임 높이 (정렬)
    H       = CELL * ROWS           # 게임 콘텐츠 높이
    SIDEBAR = CELL * 6
    CAM_W   = SIDEBAR - 16
    CAM_H   = int(CAM_W * 3 / 4)
    UI_SCALE = CELL / 32

    PAD_Y   = (H_FRAME - H) // 2   # 위아래 여백

    def mkfont(names, size, bold=False):
        for n in names:
            f = pygame.font.SysFont(n, round(size * UI_SCALE), bold=bold)
            if f: return f
        return pygame.font.Font(None, round(size * UI_SCALE))

    font  = mkfont(["applegothic", "malgunGothic", "arial"], 23, bold=True)
    sfont = mkfont(["applegothic", "malgunGothic", "arial"], 13)
    bfont = mkfont(["applegothic", "malgunGothic", "arial"], 36, bold=True)
    fonts = (font, sfont, bfont)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    threading.Thread(target=tracking_worker, args=(cap,), daemon=True).start()

    game       = Game()
    started    = False
    fullscreen = False
    screen     = pygame.display.set_mode((W, H_FRAME), pygame.RESIZABLE)
    canvas     = pygame.Surface((W, H))       # game content
    frame      = pygame.Surface((W, H_FRAME)) # 9:16 output frame

    def toggle_fullscreen():
        nonlocal fullscreen, screen
        fullscreen = not fullscreen
        screen = pygame.display.set_mode(
            (0, 0) if fullscreen else (W, H_FRAME),
            pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE)

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if fullscreen: toggle_fullscreen()
                    else:          pygame.quit(); sys.exit()
                if event.key == pygame.K_F11:
                    toggle_fullscreen(); continue
                if not started:
                    if event.key == pygame.K_RETURN: started = True
                    continue
                if event.key == pygame.K_r:
                    game = Game(); continue
                if not game.over:
                    if event.key == pygame.K_LEFT  and game.valid(dx=-1): game.piece.x -= 1
                    if event.key == pygame.K_RIGHT and game.valid(dx=1):  game.piece.x += 1
                    if event.key == pygame.K_UP:    game.try_rotate(rot_cw(game.piece.shape))
                    if event.key == pygame.K_z:     game.try_rotate(rot_ccw(game.piece.shape))
                    if event.key == pygame.K_DOWN  and game.valid(dy=1):  game.piece.y += 1
                    if event.key == pygame.K_SPACE: game.hard_drop()

        if started and not game.over:
            game.update()

        draw_board(game, bfont, canvas)
        draw_sidebar(canvas, game, fonts)
        if not started or game.over:
            draw_overlay(canvas, game, fonts, started)

        # Compose: game canvas centred vertically in 9:16 frame
        frame.fill(BG)
        frame.blit(canvas, (0, PAD_Y))

        sw, sh   = screen.get_size()
        scale    = min(sw / W, sh / H_FRAME)
        scaled_w = int(W * scale);  scaled_h = int(H_FRAME * scale)
        scaled   = pygame.transform.smoothscale(frame, (scaled_w, scaled_h))
        screen.fill((0, 0, 0))
        screen.blit(scaled, ((sw - scaled_w) // 2, (sh - scaled_h) // 2))
        pygame.display.flip()


if __name__ == "__main__":
    main()
