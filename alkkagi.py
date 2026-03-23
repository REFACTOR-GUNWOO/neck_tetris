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

# ─── Layout ───────────────────────────────────────────────────
GAME_W  = 640
GAME_H  = 520
SIDEBAR = 260
CAM_W   = 240
CAM_H   = 175
W       = GAME_W + SIDEBAR
H       = GAME_H
FPS     = 60

# ─── Physics ──────────────────────────────────────────────────
FRICTION    = 0.988
WALL_DAMP   = 0.65
STOP_SPEED  = 0.10
MAX_SPEED   = 20.0
PLAYER_R    = 22
TARGET_R    = 17

# ─── Colors ───────────────────────────────────────────────────
BG         = (14, 11, 7)
FELT       = (22, 62, 22)
FELT_LINE  = (28, 76, 28)
GOLD       = (185, 140, 50)
WHITE      = (255, 255, 255)
GRAY       = (110, 110, 130)
YELLOW     = (255, 215, 45)
CYAN       = (70, 210, 255)
RED        = (230, 60, 60)

MARBLE_PALETTE = [
    (220, 70, 70),  (70, 145, 225), (70, 195, 90),
    (230, 150, 40), (170, 70, 230), (225, 70, 150),
    (70, 195, 195), (200, 120, 60),
]

# ─── Level layouts ────────────────────────────────────────────
def _make_levels():
    cx = GAME_W // 2
    levels = []

    # Lv1 – triangle
    pos = []
    for row in range(4):
        for col in range(row + 1):
            pos.append((cx - row*44//2 + col*44, 80 + row*48))
    levels.append(pos)

    # Lv2 – two staggered rows
    pos = [(cx - 110 + i*55, 90)  for i in range(5)]
    pos += [(cx - 82  + i*55, 150) for i in range(4)]
    levels.append(pos)

    # Lv3 – 4×4 grid
    pos = [(cx - 105 + j*70, 75 + i*70) for i in range(3) for j in range(4)]
    levels.append(pos)

    return levels

LEVELS = _make_levels()

# ─── Shared hand state ────────────────────────────────────────
class HandState:
    def __init__(self):
        self.aim_angle      = -math.pi / 2
        self.velocity       = 0.0
        self.flick_detected = False
        self.flick_angle    = -math.pi / 2
        self.flick_power    = 0.5
        self.detected       = False
        self.cam_frame      = None
        self.lock           = threading.Lock()

hand = HandState()

# ─── Hand tracking thread ─────────────────────────────────────
def tracking_worker(cap):
    mp_hands = mp.solutions.hands
    det = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, model_complexity=0,
        min_detection_confidence=0.6, min_tracking_confidence=0.5)

    hist     = deque(maxlen=15)
    cooldown = 0
    FLICK_T  = 0.9    # normalized units/sec
    POWER_S  = 0.60

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01); continue

        frame = cv2.flip(frame, 1)
        ih, iw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = det.process(rgb)

        detected  = False
        aim_angle = -math.pi / 2
        velocity  = 0.0

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            detected = True
            wrist = lm[0];  tip = lm[8]

            # Aim direction: wrist → index fingertip
            aim_angle = math.atan2(tip.y - wrist.y, tip.x - wrist.x)

            now = time.time()
            hist.append((tip.x, tip.y, now))

            # Velocity from 5-frame window
            if len(hist) >= 5:
                h = list(hist)
                dt = h[-1][2] - h[-5][2]
                if dt > 0.001:
                    vx = (h[-1][0] - h[-5][0]) / dt
                    vy = (h[-1][1] - h[-5][1]) / dt
                    velocity = math.sqrt(vx*vx + vy*vy)

                    if velocity > FLICK_T and cooldown <= 0:
                        power = min(velocity * POWER_S, 1.0)
                        angle = math.atan2(vy, vx)
                        with hand.lock:
                            hand.flick_detected = True
                            hand.flick_angle    = angle
                            hand.flick_power    = power
                        cooldown = 45

            if cooldown > 0:
                cooldown -= 1

            # Draw skeleton
            for conn in mp_hands.HAND_CONNECTIONS:
                a, b = conn
                ax = int(lm[a].x*iw); ay = int(lm[a].y*ih)
                bx = int(lm[b].x*iw); by = int(lm[b].y*ih)
                cv2.line(frame, (ax, ay), (bx, by), (50, 180, 50), 2)
            for i in [0, 8]:
                px = int(lm[i].x*iw); py = int(lm[i].y*ih)
                cv2.circle(frame, (px, py), 6, (0, 230, 170), -1)

            # Power bar overlay on frame
            bar_w = int(min(velocity / FLICK_T, 1.0) * int(iw * 0.75))
            bar_col = (0, 220, 80) if bar_w < int(iw*0.5) else (255, 160, 0)
            cv2.rectangle(frame, (8, ih-22), (int(iw*0.75)+8, ih-9), (35, 35, 35), -1)
            if bar_w > 0:
                cv2.rectangle(frame, (8, ih-22), (8+bar_w, ih-9), bar_col, -1)
            cv2.putText(frame, "POWER", (8, ih-27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        small = cv2.resize(frame, (CAM_W, CAM_H))
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        with hand.lock:
            hand.detected  = detected
            hand.aim_angle = aim_angle
            hand.velocity  = velocity
            hand.cam_frame = small

# ─── Marble ───────────────────────────────────────────────────
class Marble:
    def __init__(self, x, y, r, color, is_player=False):
        self.x = float(x);  self.y = float(y)
        self.vx = 0.0;       self.vy = 0.0
        self.r  = r
        self.color     = color
        self.is_player = is_player
        self.alive     = True
        self.flash     = 0   # frames to flash white after collision

    def update(self):
        self.x += self.vx;  self.y += self.vy
        self.vx *= FRICTION; self.vy *= FRICTION

        if self.is_player:
            if self.x - self.r < 0:
                self.x = self.r;            self.vx =  abs(self.vx) * WALL_DAMP
            if self.x + self.r > GAME_W:
                self.x = GAME_W - self.r;   self.vx = -abs(self.vx) * WALL_DAMP
            if self.y - self.r < 0:
                self.y = self.r;            self.vy =  abs(self.vy) * WALL_DAMP
            if self.y + self.r > GAME_H:
                self.y = GAME_H - self.r;   self.vy = -abs(self.vy) * WALL_DAMP

        if self.flash > 0:
            self.flash -= 1

        if math.sqrt(self.vx*self.vx + self.vy*self.vy) < STOP_SPEED:
            self.vx = self.vy = 0.0

    @property
    def moving(self):
        return math.sqrt(self.vx*self.vx + self.vy*self.vy) > STOP_SPEED


def collide(a, b):
    dx = b.x - a.x;  dy = b.y - a.y
    d  = math.sqrt(dx*dx + dy*dy)
    if d == 0 or d >= a.r + b.r:
        return False
    nx = dx/d;  ny = dy/d
    overlap = (a.r + b.r - d) * 0.5
    a.x -= nx*overlap;  a.y -= ny*overlap
    b.x += nx*overlap;  b.y += ny*overlap
    dot = (a.vx - b.vx)*nx + (a.vy - b.vy)*ny
    if dot > 0:
        a.vx -= dot*nx;  a.vy -= dot*ny
        b.vx += dot*nx;  b.vy += dot*ny
        a.flash = b.flash = 7
    return True

# ─── Particle ─────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color, speed_mul=1.0):
        ang = random.uniform(0, math.pi * 2)
        spd = random.uniform(1.5, 5.0) * speed_mul
        self.x = float(x);  self.y = float(y)
        self.vx = math.cos(ang) * spd
        self.vy = math.sin(ang) * spd
        self.life  = 1.0
        self.decay = random.uniform(0.028, 0.065)
        self.r     = random.uniform(2.0, 4.5)
        self.color = color

    def update(self):
        self.x += self.vx;  self.y += self.vy
        self.vy += 0.10
        self.life -= self.decay

# ─── Drawing ──────────────────────────────────────────────────
def draw_marble(surf, m):
    ix, iy, r = int(m.x), int(m.y), m.r
    base = WHITE if (m.flash > 0 and m.flash % 2 == 0) else m.color

    # Drop shadow
    sh = pygame.Surface((r*2+8, r*2+8), pygame.SRCALPHA)
    pygame.draw.circle(sh, (0, 0, 0, 55), (r+5, r+5), r)
    surf.blit(sh, (ix - r + 1, iy - r + 1))

    # Base + shading via layered circles
    pygame.draw.circle(surf, base, (ix, iy), r)
    dark = tuple(max(0, c - 85) for c in base)
    pygame.draw.circle(surf, dark, (ix + 4, iy + 4), r - 2)
    pygame.draw.circle(surf, base, (ix,     iy    ), r - 3)

    # Specular highlight
    hx = ix - r // 3;  hy = iy - r // 3
    pygame.draw.circle(surf, WHITE, (hx, hy), max(2, r // 4))
    pygame.draw.circle(surf, WHITE, (hx + 2, hy + 2), max(1, r // 8))

    # Rim
    border = tuple(max(0, c - 50) for c in base)
    pygame.draw.circle(surf, border, (ix, iy), r, 2)


def draw_aim_ray(surf, sx, sy, angle, power):
    """Dotted aim ray that bounces off walls."""
    x, y   = float(sx), float(sy)
    dx, dy = math.cos(angle), math.sin(angle)
    step   = 10
    total  = 0
    max_d  = int(80 + power * 380)

    while total < max_d:
        nx = x + dx * step
        ny = y + dy * step

        if nx < 0:        nx = 0;      dx =  abs(dx)
        elif nx > GAME_W: nx = GAME_W; dx = -abs(dx)
        if ny < 0:        ny = 0;      dy =  abs(dy)
        elif ny > GAME_H: ny = GAME_H; dy = -abs(dy)

        if total % 18 < 11:  # dotted gap
            alpha = int(210 * (1.0 - total / max_d))
            dot   = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(dot, (*YELLOW, alpha), (3, 3), 3)
            surf.blit(dot, (int(nx) - 3, int(ny) - 3))

        x, y   = nx, ny
        total += step


def draw_power_ring(surf, cx, cy, r, power):
    """Animated ring around player marble showing power."""
    if power < 0.05:
        return
    ring_r  = int(r + 6 + power * 18)
    alpha   = int(power * 180)
    col     = (80, 220, 80) if power < 0.5 else (255, 160, 0) if power < 0.85 else (220, 60, 60)
    s       = pygame.Surface((ring_r*2+4, ring_r*2+4), pygame.SRCALPHA)
    pygame.draw.circle(s, (*col, alpha), (ring_r+2, ring_r+2), ring_r, 3)
    surf.blit(s, (int(cx) - ring_r - 2, int(cy) - ring_r - 2))


def draw_table(surf):
    surf.fill(FELT)
    for i in range(0, GAME_W, 55):
        pygame.draw.line(surf, FELT_LINE, (i, 0), (i, GAME_H), 1)
    for j in range(0, GAME_H, 55):
        pygame.draw.line(surf, FELT_LINE, (0, j), (GAME_W, j), 1)
    pygame.draw.rect(surf, GOLD, (0, 0, GAME_W, GAME_H), 7)


def draw_sidebar(surf, game, fonts, sx):
    font, sfont = fonts
    pygame.draw.rect(surf, (17, 17, 28), (sx, 0, SIDEBAR, H))

    y = 14
    def label(txt, col=GRAY):
        nonlocal y
        surf.blit(sfont.render(txt, True, col), (sx+12, y))

    def value(txt, col=YELLOW):
        nonlocal y
        surf.blit(font.render(txt, True, col), (sx+12, y))

    label("SCORE"); y += 19; value(f"{game.score:,}"); y += 32
    label("LEVEL"); y += 19; value(str(game.level + 1)); y += 32
    label("SHOTS LEFT"); y += 19; value(str(game.shots_left)); y += 32
    label(f"TARGETS: {len(game.targets)}", WHITE); y += 28

    # Power bar
    with hand.lock:
        vel = hand.velocity;  detected = hand.detected;  cam = hand.cam_frame

    pct = min(vel / 0.9, 1.0)
    label("POWER", GRAY); y += 19
    bar_w = SIDEBAR - 28
    pygame.draw.rect(surf, (38, 38, 50), (sx+12, y, bar_w, 18), 0, 5)
    if pct > 0:
        fill_col = (80,220,80) if pct < 0.55 else (255,160,0) if pct < 0.85 else (220,60,60)
        pygame.draw.rect(surf, fill_col, (sx+12, y, int(pct*bar_w), 18), 0, 5)
    pygame.draw.rect(surf, GRAY, (sx+12, y, bar_w, 18), 1, 5)
    y += 28

    # Hand status
    col = (80, 220, 80) if detected else (200, 80, 80)
    txt = "Hand detected" if detected else "No hand"
    surf.blit(sfont.render(txt, True, col), (sx+12, y)); y += 26

    # Aim angle indicator
    aim_rad = hand.aim_angle if detected else -math.pi/2
    aix, aiy = sx + 40, y + 38
    pygame.draw.circle(surf, (38, 38, 50), (aix, aiy), 30, 2)
    ex = int(aix + 26 * math.cos(aim_rad))
    ey = int(aiy + 26 * math.sin(aim_rad))
    col_arrow = (80,220,80) if detected else (80,80,100)
    pygame.draw.line(surf, col_arrow, (aix, aiy), (ex, ey), 3)
    pygame.draw.circle(surf, col_arrow, (aix, aiy), 4)
    surf.blit(sfont.render("AIM", True, GRAY), (sx + 75, y + 30))
    y += 80

    # Webcam
    if cam is not None:
        cam_surf = pygame.surfarray.make_surface(cam.swapaxes(0, 1))
        cam_y    = H - CAM_H - 30
        surf.blit(cam_surf, (sx + 5, cam_y))
        pygame.draw.rect(surf, GOLD, (sx + 5, cam_y, CAM_W, CAM_H), 1)
        surf.blit(sfont.render("WEBCAM", True, GRAY), (sx + 10, cam_y - 17))

    # Hint
    for i, txt in enumerate(["Point finger to aim", "Flick to shoot!"]):
        surf.blit(sfont.render(txt, True, (48, 48, 65)), (sx + 10, H - 32 + i * 16))


# ─── Game ─────────────────────────────────────────────────────
class Game:
    SHOTS_PER_LEVEL = 10

    def __init__(self):
        self.score      = 0
        self.level      = 0
        self.shots_left = self.SHOTS_PER_LEVEL
        self.state      = "waiting"   # waiting | shooting | level_clear | failed
        self.marbles    = []
        self.particles  = []
        self.player     = None
        self._load_level()

    def _load_level(self):
        self.marbles    = []
        self.particles  = []
        self.shots_left = self.SHOTS_PER_LEVEL
        colors = MARBLE_PALETTE[:]
        random.shuffle(colors)
        pos_list = LEVELS[self.level % len(LEVELS)]
        for i, (x, y) in enumerate(pos_list):
            self.marbles.append(Marble(x, y, TARGET_R, colors[i % len(colors)]))
        self.player = Marble(GAME_W // 2, GAME_H - 55, PLAYER_R, YELLOW, is_player=True)
        self.marbles.append(self.player)
        self.state = "waiting"

    def _burst(self, x, y, color, n=12, speed=1.0):
        for _ in range(n):
            self.particles.append(Particle(x, y, color, speed))

    def _shoot(self, angle, power):
        spd = power * MAX_SPEED
        self.player.vx = math.cos(angle) * spd
        self.player.vy = math.sin(angle) * spd
        self._burst(self.player.x, self.player.y, YELLOW, 10, 1.2)
        self.shots_left -= 1
        self.state = "shooting"

    @property
    def targets(self):
        return [m for m in self.marbles if not m.is_player]

    def update(self):
        # Particles always update
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.life > 0]

        if self.state in ("level_clear", "failed"):
            return

        # Input
        with hand.lock:
            flick   = hand.flick_detected
            f_ang   = hand.flick_angle
            f_pow   = hand.flick_power
            hand.flick_detected = False

        if flick and self.state == "waiting" and self.shots_left > 0:
            self._shoot(f_ang, f_pow)

        # Physics
        for m in self.marbles:
            m.update()

        # Collision
        for i in range(len(self.marbles)):
            for j in range(i + 1, len(self.marbles)):
                mi, mj = self.marbles[i], self.marbles[j]
                if collide(mi, mj):
                    mx = (mi.x + mj.x) / 2;  my = (mi.y + mj.y) / 2
                    self._burst(mx, my, mj.color, 7)

        # Remove out-of-bounds targets (no wall bounce → slide off)
        for m in list(self.marbles):
            if m.is_player:
                continue
            if m.x < -m.r or m.x > GAME_W + m.r or m.y < -m.r or m.y > GAME_H + m.r:
                self._burst(m.x, m.y, m.color, 20, 1.5)
                self.score += 10
                m.alive = False
        self.marbles = [m for m in self.marbles if m.is_player or m.alive]

        # State transitions
        if self.state == "shooting" and not any(m.moving for m in self.marbles):
            if not self.targets:
                self.score += 30 * (self.level + 1)
                self._burst(GAME_W//2, GAME_H//2, YELLOW, 40, 2.0)
                self.state = "level_clear"
            elif self.shots_left <= 0:
                self.state = "failed"
            else:
                self.state = "waiting"

    def next_level(self):
        self.level += 1
        self._load_level()

    def retry(self):
        self._load_level()


# ─── Main ─────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("알까기 – Motion Marble")
    clock  = pygame.time.Clock()

    def mkfont(names, size, bold=False):
        for n in names:
            f = pygame.font.SysFont(n, size, bold=bold)
            if f: return f
        return pygame.font.Font(None, size)

    font  = mkfont(["applegothic", "malgunGothic", "arial"], 24, bold=True)
    sfont = mkfont(["applegothic", "malgunGothic", "arial"], 15)
    bfont = mkfont(["applegothic", "malgunGothic", "arial"], 44, bold=True)
    fonts = (font, sfont)

    # Camera – must init on main thread (macOS)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = threading.Thread(target=tracking_worker, args=(cap,), daemon=True)
    tracker.start()

    game    = Game()
    started = False
    table   = pygame.Surface((GAME_W, GAME_H))

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

                if not started:
                    started = True; continue

                if game.state == "level_clear" and event.key == pygame.K_RETURN:
                    game.next_level()
                if game.state == "failed" and event.key == pygame.K_r:
                    game.retry()
                if game.state == "failed" and event.key == pygame.K_n:
                    game.next_level()

                # Keyboard fallback shoot
                if game.state == "waiting" and event.key == pygame.K_SPACE:
                    with hand.lock:
                        ang = hand.aim_angle
                        vel = hand.velocity
                    game._shoot(ang, max(0.35, min(vel, 1.0)))

                # Arrow keys to adjust aim (keyboard fallback)
                if game.state == "waiting":
                    with hand.lock:
                        if not hand.detected:
                            step = 0.08
                            if event.key == pygame.K_LEFT:
                                hand.aim_angle -= step
                            if event.key == pygame.K_RIGHT:
                                hand.aim_angle += step

        if started:
            game.update()

        # ── Render ──────────────────────────────────────────
        draw_table(table)

        # Particles
        for p in game.particles:
            if 0 <= int(p.x) < GAME_W and 0 <= int(p.y) < GAME_H:
                s = pygame.Surface((int(p.r)*2+2, int(p.r)*2+2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p.color, int(p.life * 200)),
                                   (int(p.r)+1, int(p.r)+1), int(p.r))
                table.blit(s, (int(p.x - p.r), int(p.y - p.r)))

        # Aim ray + power ring
        if game.state == "waiting" and started:
            with hand.lock:
                aim = hand.aim_angle
                vel = hand.velocity
            pct = min(vel / 0.9, 1.0)
            draw_aim_ray(table, game.player.x, game.player.y, aim, pct)
            draw_power_ring(table, game.player.x, game.player.y, PLAYER_R, pct)

        # Marbles
        for m in game.marbles:
            draw_marble(table, m)

        screen.fill(BG)
        screen.blit(table, (0, 0))
        draw_sidebar(screen, game, fonts, GAME_W)

        # ── Overlay screens ─────────────────────────────────
        if not started:
            ov = pygame.Surface((GAME_W, H), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 155))
            screen.blit(ov, (0, 0))
            t = bfont.render("알까기", True, YELLOW)
            screen.blit(t, t.get_rect(centerx=GAME_W//2, centery=H//2 - 95))
            for i, line in enumerate([
                "검지로 목표를 가리켜 조준",
                "손을 빠르게 튕겨서 발사!",
                "속도가 빠를수록 강하게 발사됩니다",
                "",
                "Press any key to start",
            ]):
                col = WHITE if "key" in line else GRAY
                s = sfont.render(line, True, col)
                screen.blit(s, s.get_rect(centerx=GAME_W//2, centery=H//2 - 20 + i*30))

        elif game.state == "level_clear":
            ov = pygame.Surface((GAME_W, H), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 145))
            screen.blit(ov, (0, 0))
            for txt, col, dy in [
                (f"Level {game.level} Clear!", YELLOW, -55),
                (f"Score: {game.score:,}",     WHITE,  5),
                ("Enter – Next Level",          GRAY,   50),
            ]:
                s = (bfont if dy < 0 else font).render(txt, True, col)
                screen.blit(s, s.get_rect(centerx=GAME_W//2, centery=H//2 + dy))

        elif game.state == "failed":
            ov = pygame.Surface((GAME_W, H), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 155))
            screen.blit(ov, (0, 0))
            for txt, col, dy in [
                ("Shot Limit Reached", RED,   -55),
                (f"Score: {game.score:,}", WHITE, 5),
                ("R – Retry  /  N – Next Level", GRAY, 50),
            ]:
                s = (bfont if dy < 0 else font).render(txt, True, col)
                screen.blit(s, s.get_rect(centerx=GAME_W//2, centery=H//2 + dy))

        pygame.display.flip()


if __name__ == "__main__":
    main()
