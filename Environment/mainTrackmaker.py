import pygame
import math

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 750
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drivable Rectangle")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Clock for controlling frame rate
clock = pygame.time.Clock()

MAKETRACK = True
if MAKETRACK:
    pygame.display.set_caption("Drivable Rectangle - Track Maker")

# Car properties
car_width, car_height = 50, 30
car_x, car_y = WIDTH // 2, HEIGHT // 2
car_angle = 0
car_speed = 0
max_speed = 5
acceleration = 0.1
deceleration = 0.05
turn_speed = 3
track_color = (0, 0, 0)
track_width = 10
track_points = [(134, 555), (135, 143), (193, 95), (576, 100), (644, 159), (624, 546), (567, 599), (190, 599), (135, 556), (45, 560), (50, 140), (181, 5), (567, 10), (752, 139), (718, 545), (587, 693), (188, 687), (46, 563), (132, 492), (46, 499), (128, 436), (45, 431), (134, 378), (46, 381), (135, 317), (51, 317), (135, 256), (50, 253), (133, 199), (52, 200), (136, 146), (51, 138), (168, 113), (109, 79), (190, 91), (183, 7), (242, 90), (237, 6), (285, 93), (283, 8), (327, 97), (330, 8), (372, 97), (378, 11), (415, 99), (421, 9), (462, 96), (471, 14), (518, 97), (528, 13), (572, 101), (592, 31), (606, 124), (643, 69), (640, 156), (694, 105), (648, 185), (743, 181), (646, 220), (739, 230), (643, 279), (738, 295), (638, 332), (729, 351), (633, 399), (728, 417), (630, 476), (720, 485), (623, 531), (712, 542), (607, 569), (664, 606), (574, 595), (589, 688), (518, 604), (522, 691), (446, 599), (454, 690), (390, 602), (393, 691), (326, 598), (323, 689), (247, 598), (243, 690), (190, 598), (180, 686), (159, 580), (113, 623), (131, 554), (48, 555)]

def draw_track():
    if(len(track_points) > 1):
        pygame.draw.lines(screen, track_color, False, track_points, track_width)
        
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
    for i in range(len(track_points) - 1):
        pt1 = track_points[i]
        pt2 = track_points[i + 1]
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
        if event.type == pygame.MOUSEBUTTONDOWN and MAKETRACK:
            if event.button == 1:  # Left mouse button
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if len(track_points) == 0 or (mouse_x, mouse_y) != track_points[-1]:
                    track_points.append((mouse_x, mouse_y))
            elif event.button == 3:  # Right mouse button
                if track_points:
                    track_points.pop()

    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_s] and MAKETRACK:
        print(track_points)
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
    screen.blit(rotated_car, rotated_rect.topleft)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()