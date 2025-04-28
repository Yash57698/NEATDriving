import pygame
import math

# Initialize pygame
pygame.init()
font = pygame.font.SysFont(None, 36)

# Line checkpoints: list of line segments [(start), (end)]
checkpoints = [[(30, 300), (119, 300)], [(34, 167), (113, 163)], [(37, 82), (116, 111)], [(151, 26), (166, 104)], [(265, 30), (224, 97)], [(311, 127), (260, 171)], [(338, 209), (291, 242)], [(351, 247), (351, 316)], [(430, 249), (436, 314)], [(480, 242), (529, 313)], [(506, 186), (575, 217)], [(529, 104), (593, 171)], [(641, 97), (650, 163)], [(737, 98), (678, 165)], [(750, 193), (678, 203)], [(751, 259), (676, 260)], [(753, 319), (675, 315)], [(750, 391), (674, 378)], [(708, 468), (637, 426)], [(653, 515), (593, 467)], [(629, 539), (533, 537)], [(664, 582), (573, 609)], [(724, 649), (580, 630)], [(529, 637), (536, 701)], [(465, 635), (466, 703)], [(366, 630), (373, 703)], [(299, 631), (296, 704)], [(233, 630), (232, 713)], [(146, 631), (139, 718)], [(118, 631), (27, 637)], [(119, 555), (26, 540)], [(118, 458), (24, 449)], [(116, 379), (25, 368)]]
current_checkpoint = 0
points = 0

def lines_intersect(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def car_hits_track_edges(car_pos, angle_deg):
    cx, cy = car_pos
    w, h = car_width, car_height
    angle_rad = -math.radians(angle_deg)

    # Offset to each corner from center
    half_w, half_h = w / 2, h / 2
    corners_local = [(-half_w, -half_h), (half_w, -half_h),
                     (half_w, half_h), (-half_w, half_h)]

    # Rotate and translate corners to world position
    corners_world = []
    for dx, dy in corners_local:
        x = cx + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        y = cy + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        corners_world.append((x, y))

    # Build car edges from corners
    edges = [(corners_world[i], corners_world[(i + 1) % 4]) for i in range(4)]

    # Check each edge against track lines
    for edge in edges:
        for track in track_points:
            for i in range(len(track) - 1):
                if get_line_intersection(edge[0], edge[1], track[i], track[i + 1]):
                    return True
    return False

# Screen dimensions
WIDTH, HEIGHT = 800, 750
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drivable Rectangle")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Car properties
car_width, car_height = 50, 30
car_x, car_y = 70, 450
car_angle = 90
car_speed = 0
max_speed = 5
acceleration = 0.1
deceleration = 0.05
turn_speed = 3
track_color = (0, 0, 0)
track_width = 10
track_points = [[(22, 700), (32, 80), (98, 21), (271, 26), (354, 246), (480, 243), (531, 97), (752, 95),(752, 429), (627, 542), (756, 689), (31, 729), (25, 700)],[ (122, 634), (117, 112), (224, 100), (323, 319), (532, 317), (594, 174), (674, 162), (673, 386), (531, 535), (583, 629), (125, 631)]]

def draw_track():
    for track in track_points:
        if(len(track) > 1):
            pygame.draw.lines(screen, track_color, False, track, track_width)

def reset_game():
    global car_x, car_y, car_angle, car_speed, current_checkpoint, points
    car_x, car_y = 70, 450
    car_angle = 90
    car_speed = 0
    current_checkpoint = 0
    points = 0
        
# Main loop
running = True
def cast_ray(origin, angle_deg, max_distance=200):
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = -math.sin(angle_rad)  # pygame y-axis is downward
    end = (origin[0] + dx * max_distance, origin[1] + dy * max_distance)

    closest_intersection = None
    min_dist = float('inf')

    # Check intersection with each segment in track
    for track in track_points:
        for i in range(len(track) - 1):
            pt1 = track[i]
            pt2 = track[i + 1]
            intersection = get_line_intersection(origin, end, pt1, pt2)
            if intersection:
                dist = math.hypot(intersection[0] - origin[0], intersection[1] - origin[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_intersection = intersection

    return closest_intersection if closest_intersection else end

def get_line_intersection(p1, p2, p3, p4):
    # Compute the intersection point of lines (p1 to p2) and (p3 to p4)
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Parallel lines

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    if (min(x1, x2) <= px <= max(x1, x2) and
        min(y1, y2) <= py <= max(y1, y2) and
        min(x3, x4) <= px <= max(x3, x4) and
        min(y3, y4) <= py <= max(y3, y4)):
        return px, py
    return None


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if car_hits_track_edges((car_x, car_y), car_angle):
        reset_game()

    if keys[pygame.K_UP]:
        car_speed = min(car_speed + acceleration, max_speed)
    elif keys[pygame.K_DOWN]:
        car_speed = max(car_speed - acceleration, -max_speed / 2)
    else:
        if car_speed > 0:
            car_speed = max(car_speed - deceleration, 0)
        elif car_speed < 0:
            car_speed = min(car_speed + deceleration, 0)

    if keys[pygame.K_LEFT]:
        car_angle += turn_speed * (car_speed / max_speed)
    if keys[pygame.K_RIGHT]:
        car_angle -= turn_speed * (car_speed / max_speed)

    # Update car position
    car_x += car_speed * math.cos(math.radians(car_angle))
    car_y -= car_speed * math.sin(math.radians(car_angle))

    # Checkpoint crossing detection using previous and current position
    if current_checkpoint < len(checkpoints):
        cp_start, cp_end = checkpoints[current_checkpoint]

        # Get car corners
        angle_rad = -math.radians(car_angle)
        half_w, half_h = car_width / 2, car_height / 2
        corners_local = [(-half_w, -half_h), (half_w, -half_h),
                         (half_w, half_h), (-half_w, half_h)]

        corners_world = []
        for dx, dy in corners_local:
            x = car_x + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            y = car_y + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            corners_world.append((x, y))

        car_edges = [(corners_world[i], corners_world[(i + 1) % 4]) for i in range(4)]

        # Check all car edges for crossing the current checkpoint line
        for edge_start, edge_end in car_edges:
            if get_line_intersection(edge_start, edge_end, cp_start, cp_end):
                points += 1
                current_checkpoint += 1
                break  # Avoid counting the same checkpoint multiple times


    screen.fill(WHITE)
    draw_track()
    num_rays = 16
    ray_length = 200
    ray_color = (0, 255, 0)
    for i in range(num_rays):
        angle = car_angle + i * 360 / num_rays
        origin = (car_x, car_y)
        end_point = cast_ray(origin, angle, ray_length)
        pygame.draw.line(screen, ray_color, origin, end_point,2)
        pygame.draw.circle(screen, (0, 0, 255), (int(end_point[0]), int(end_point[1])), 3)

    car_image = pygame.image.load("car_red_1.png")
    car_image = pygame.transform.scale(car_image, (car_width, car_height))

    rotated_car = pygame.transform.rotate(car_image, car_angle)
    rotated_rect = rotated_car.get_rect(center=(car_x, car_y))
    # Draw car hitbox
    screen.blit(rotated_car, rotated_rect.topleft)
    for i in range(4):
        pygame.draw.line(screen, (255, 0, 0), corners_world[i], corners_world[(i + 1) % 4], 2)

    # Draw line checkpoints
    for idx, (start, end) in enumerate(checkpoints):
        color = (0, 200, 255) if idx == current_checkpoint else (180, 180, 180)
        pygame.draw.line(screen, color, start, end, 4)

    # Display checkpoint progress
    progress_text = font.render(f"Checkpoints: {points}/{len(checkpoints)}", True, (0, 0, 0))
    screen.blit(progress_text, (500, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()