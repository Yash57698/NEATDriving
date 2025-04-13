import pygame
import math

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
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
track_points = []

def draw_track():
    if(len(track_points) > 1):
        pygame.draw.lines(screen, track_color, False, track_points, track_width)
        
# Main loop

running = True
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

    car_image = pygame.image.load("car_red_1.png")
    car_image = pygame.transform.scale(car_image, (car_width, car_height))

    rotated_car = pygame.transform.rotate(car_image, car_angle)
    rotated_rect = rotated_car.get_rect(center=(car_x, car_y))
    screen.fill(WHITE)
    screen.blit(rotated_car, rotated_rect.topleft)
    draw_track()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()