import pygame
import math
import torch
import numpy as np
import sys
import streamlit as st
import os

sys.path.append('..')
from Framework.NN import NN
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
placeholder = st.empty()
st.title("Car Simulator")

track2 = [[(134, 555), (135, 143), (193, 95), (576, 100), (644, 159), (624, 546), (567, 599), (190, 599), (135, 556)],[(45, 560), (50, 140), (181, 5), (567, 10), (752, 139), (718, 545), (587, 693), (188, 687), (46, 563)]]
checkpoints2 = [[(134, 378), (46, 381)], [(135, 317), (51, 317)], [(135, 256), (50, 253)], [(133, 199), (52, 200)], [(136, 146), (51, 138)], [(168, 113), (109, 79)], [(190, 91), (183, 7)], [(242, 90), (237, 6)], [(285, 93), (283, 8)], [(327, 97), (330, 8)], [(372, 97), (378, 11)], [(415, 99), (421, 9)], [(462, 96), (471, 14)], [(518, 97), (528, 13)], [(572, 101), (592, 31)], [(606, 124), (643, 69)], [(640, 156), (694, 105)], [(648, 185), (743, 181)], [(646, 220), (739, 230)], [(643, 279), (738, 295)], [(638, 332), (729, 351)], [(633, 399), (728, 417)], [(630, 476), (720, 485)], [(623, 531), (712, 542)], [(607, 569), (664, 606)], [(574, 595), (589, 688)], [(518, 604), (522, 691)], [(446, 599), (454, 690)], [(390, 602), (393, 691)], [(326, 598), (323, 689)], [(247, 598), (243, 690)], [(190, 598), (180, 686)], [(159, 580), (113, 623)], [(131, 554), (48, 555)],[(132, 492), (46, 499)], [(128, 436), (45, 431)]]
carresetpostion2 = (92,526)

track1 = [[(22, 700), (32, 80), (98, 21), (271, 26), (354, 246), (480, 243), (531, 97), (752, 95),(752, 429), (627, 542), (756, 689), (31, 729), (25, 700)],[ (122, 634), (117, 112), (224, 100), (323, 319), (532, 317), (594, 174), (674, 162), (673, 386), (531, 535), (583, 629), (125, 631)]]
checkpoints1 = [[(30, 300), (119, 300)], [(34, 167), (113, 163)], [(37, 82), (116, 111)], [(151, 26), (166, 104)], [(265, 30), (224, 97)], [(311, 127), (260, 171)], [(338, 209), (291, 242)], [(351, 247), (351, 316)], [(430, 249), (436, 314)], [(480, 242), (529, 313)], [(506, 186), (575, 217)], [(529, 104), (593, 171)], [(641, 97), (650, 163)], [(737, 98), (678, 165)], [(750, 193), (678, 203)], [(751, 259), (676, 260)], [(753, 319), (675, 315)], [(750, 391), (674, 378)], [(708, 468), (637, 426)], [(653, 515), (593, 467)], [(629, 539), (533, 537)], [(664, 582), (573, 609)], [(724, 649), (580, 630)], [(529, 637), (536, 701)], [(465, 635), (466, 703)], [(366, 630), (373, 703)], [(299, 631), (296, 704)], [(233, 630), (232, 713)], [(146, 631), (139, 718)], [(118, 631), (27, 637)], [(119, 555), (26, 540)], [(118, 458), (24, 449)], [(116, 379), (25, 368)]]
carresetpostion1 = (70,450)


class CarSimulator:
    WIDTH, HEIGHT = 1200, 850
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    track_color = (0, 0, 0)
    track_width = 10
    checkpoints = checkpoints2
    track_points = track2
    resetpos = carresetpostion2
    car_x, car_y = resetpos
    car_width, car_height = 50, 30
    car_angle = 90
    car_speed = 0
    max_speed = 5
    acceleration = 0.1
    deceleration = 0.05
    turn_speed = 3

    def __init__(self, draw=False, slow=False,Display = False):
        if(Display):
            self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Drivable Rectangle")
            self.clock = pygame.time.Clock()
            self.car_image = pygame.image.load("../Environment/car_red_1.png")
            self.car_image = pygame.transform.scale(self.car_image, (self.car_width, self.car_height))
            self.font = pygame.font.SysFont(None, 36)
        self.draw = draw
        self.slow = slow

        self.Display = Display
        self.reset()

    def reset(self):
        self.car_x, self.car_y = self.resetpos
        self.car_angle = 90 + np.random.randint(-30, 30)
        self.car_speed = 0
        self.current_checkpoint = 0
        self.points = 0

        self.car_width, self.car_height = 50, 30
        self.max_speed = 5
        self.acceleration = 0.1
        self.deceleration = 0.05
        self.turn_speed = 3

    def cast_ray(self, origin, angle_deg, max_distance=200):
        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad)
        dy = -math.sin(angle_rad)
        end = (origin[0] + dx * max_distance, origin[1] + dy * max_distance)

        closest = None
        min_dist = float('inf')

        for track in self.track_points:
            for i in range(len(track) - 1):
                pt1, pt2 = track[i], track[i+1]
                inter = self.get_line_intersection(origin, end, pt1, pt2)
                if inter:
                    dist = math.hypot(inter[0] - origin[0], inter[1] - origin[1])
                    if dist < min_dist:
                        closest, min_dist = inter, dist
        return closest if closest else end

    @staticmethod
    def get_line_intersection(p1, p2, p3, p4):
        x1, y1 = p1; x2, y2 = p2
        x3, y3 = p3; x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0: return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        if (min(x1,x2)<=px<=max(x1,x2) and min(y1,y2)<=py<=max(y1,y2) and
            min(x3,x4)<=px<=max(x3,x4) and min(y3,y4)<=py<=max(y3,y4)):
            return (px, py)
        return None

    def run(self, creatureNN: NN, generation=0, framecap=250, id=-1, genome=None, speciesid=-1, species_infotext=None):
        self.framecap = framecap
        self.reset()
        frames = 0
        checkpoints_crossed = 0

        while True:
            frames += 1
            if(self.Display):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit(0)
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_s:
                            self.slow = not self.slow
                        elif event.key == pygame.K_d:
                            self.draw = not self.draw

            ray_distances = []
            for i in range(8):
                angle = self.car_angle + i * 360 / 8
                end = self.cast_ray((self.car_x, self.car_y), angle)
                dist = math.hypot(end[0] - self.car_x, end[1] - self.car_y)
                ray_distances.append(dist / 200)

            ray_distances.append(self.car_speed / self.max_speed)
            ray_distances.append(self.car_angle / 360)
            ray_tensor = torch.tensor(ray_distances)
            output = torch.tensor(creatureNN(ray_tensor)).numpy()
            outputvert = output[:3]
            outputhori = output[3:]
            # output = torch.nn.functional.softmax(torch.tensor(creatureNN(ray_tensor)), dim=0).detach().numpy()

            # dir = np.argmax(output)
            vert = np.argmax(outputvert) - 1
            hori = np.argmax(outputhori) - 1

            if self.car_hits_track_edges() or frames > framecap:
                return checkpoints_crossed * 1000 + 1

            # Movement logic
            if vert == -1:
                self.car_speed = min(self.car_speed + self.acceleration, self.max_speed)
            elif vert == 1:
                self.car_speed = max(self.car_speed - self.acceleration, -self.max_speed / 2)
            else:
                self.car_speed -= np.sign(self.car_speed) * self.deceleration

            self.car_angle += self.turn_speed * hori * (self.car_speed / self.max_speed)
            self.car_x += self.car_speed * math.cos(math.radians(self.car_angle))
            self.car_y -= self.car_speed * math.sin(math.radians(self.car_angle))

            # Checkpoint crossing detection using previous and current position
            if self.current_checkpoint < len(self.checkpoints):
                cp_start, cp_end = self.checkpoints[self.current_checkpoint]

                # Get car corners
                angle_rad = -math.radians(self.car_angle)
                half_w, half_h = self.car_width / 2, self.car_height / 2
                corners_local = [(-half_w, -half_h), (half_w, -half_h),
                                (half_w, half_h), (-half_w, half_h)]

                corners_world = []
                for dx, dy in corners_local:
                    x = self.car_x + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
                    y = self.car_y + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
                    corners_world.append((x, y))

                car_edges = [(corners_world[i], corners_world[(i + 1) % 4]) for i in range(4)]

                # Check all car edges for crossing the current checkpoint line
                for edge_start, edge_end in car_edges:
                    if self.get_line_intersection(edge_start, edge_end, cp_start, cp_end):
                        self.points += 1
                        self.current_checkpoint += 1
                        checkpoints_crossed += 1
                        self.current_checkpoint %= len(self.checkpoints)
                        break  # Avoid counting the same checkpoint multiple times

            if self.Display:
                self.clock.tick(60 if self.slow else 1000)
            if self.draw and self.Display:
                self.render(frames, output, generation, id, speciesid, species_infotext, genome)
            else:
                if(self.Display):
                    print(f"fps: {self.clock.get_fps():.2f} Generation: {generation}",end = '\r')

    def render(self, frames, output, generation, id, speciesid, species_infotext, genome):
        self.screen.fill(self.WHITE)
        self.draw_track()
        self.draw_checkpoints()
        self.draw_car()
        self.draw_rays()
        self.draw_text(frames, output, generation, id, speciesid, species_infotext, genome)
        arr = pygame.surfarray.array3d(self.screen)
        arr = np.rot90(arr, 1)
        arr = np.flip(arr, 0)
        placeholder.image(arr, channels="RGB", use_container_width=True)

    def draw_track(self):
        for track in self.track_points:
            if len(track) > 1:
                pygame.draw.lines(self.screen, self.track_color, False, track, self.track_width)

    def car_hits_track_edges(self):
        cx, cy = self.car_x, self.car_y
        angle_rad = -math.radians(self.car_angle)
        half_w, half_h = self.car_width / 2, self.car_height / 2
        corners_local = [(-half_w, -half_h), (half_w, -half_h),
                         (half_w, half_h), (-half_w, half_h)]
        corners_world = [
            (cx + dx * math.cos(angle_rad) - dy * math.sin(angle_rad),
             cy + dx * math.sin(angle_rad) + dy * math.cos(angle_rad))
            for dx, dy in corners_local
        ]
        edges = [(corners_world[i], corners_world[(i+1) % 4]) for i in range(4)]
        for edge in edges:
            for track in self.track_points:
                for i in range(len(track) - 1):
                    if self.get_line_intersection(edge[0], edge[1], track[i], track[i+1]):
                        return True
        return False

    def draw_car(self):
        rotated_car = pygame.transform.rotate(self.car_image, self.car_angle)
        rotated_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated_car, rotated_rect.topleft)

    def draw_rays(self):
        for i in range(8):
            angle = self.car_angle + i * 360 / 8
            end = self.cast_ray((self.car_x, self.car_y), angle)
            pygame.draw.line(self.screen, (0, 255, 0), (self.car_x, self.car_y), end, 2)
            pygame.draw.circle(self.screen, (255, 0, 0), (int(end[0]), int(end[1])), 5)

    def draw_text(self, frames, output, generation, id, speciesid, species_infotext, genome):
        progress_text = self.font.render(f"Checkpoints: {self.points}/{len(self.checkpoints)}", True, (0, 0, 0))
        self.screen.blit(progress_text, (400, 10))
        progress_text = self.font.render(f"Frame: {frames:03d}/{self.framecap}, FrameRate: {self.clock.get_fps():.2f}", True, (0, 0, 0))
        self.screen.blit(progress_text, (400, 50))

        output_text = self.font.render(f"Outputs: {[round(float(o),2) for o in output]}", True, (0, 0, 0))
        self.screen.blit(output_text, (10, 800))

        if(species_infotext != None):
            texts = species_infotext.split('\n')
            for i, text in enumerate(texts):
                species_text = self.font.render(text, True, (0, 0, 0))
                self.screen.blit(species_text, (800, 600 + i * 30))

        if(generation != 0):
            generation_text = self.font.render(f"{id} in Species: {speciesid} of Generation: {generation}", True, (0, 0, 0))
            self.screen.blit(generation_text, (400, 750))

        if(genome != None):
            genome.draw_network(screen = self.screen,offset = (800,50))

    def draw_rays(self):
        for i in range(8):
            angle = self.car_angle + i * 360 / 8
            end = self.cast_ray((self.car_x, self.car_y), angle)
            pygame.draw.line(self.screen, (0, 255, 0), (self.car_x, self.car_y), end, 2)
            pygame.draw.circle(self.screen, (0, 0, 255), (int(end[0]), int(end[1])), 3)

    def draw_checkpoints(self):
        for idx, (start, end) in enumerate(self.checkpoints):
                color = (0, 200, 255) if idx == self.current_checkpoint else (180, 180, 180)
                pygame.draw.line(self.screen, color, start, end, 4)

