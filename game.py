import pygame
from pygame.locals import *
import pymunk
from pymunk.pygame_util import *
from pymunk.vec2d import Vec2d
import math
import random





# pygame.init()
# size = 600, 600
# display = pygame.display.set_mode(size)
# # draw_options = pymunk.pygame_util.DrawOptions(screen)

# clock = pygame.time.Clock()
# FPS = 60
# space1 = pymunk.Space()
# space.gravity = 0, -900




def random_pos_circumference(radius, center_x, center_y):
    theta = random.random()*2*math.pi
    # r = radius*math.sqrt(random.random())
    x = radius * math.cos(theta) + center_x
    y = radius * math.sin(theta) + center_y
    return x, y


# def convert_coordinates(point):
#     return int(point[0]), int(size[0]-point[1])

class Ball():
    def __init__(self, space, x, y, radius=10):
        self.body = pymunk.Body()
        self.body.position = x, y
        self.body.mass = 1
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.density = 1
        self.shape.elasticity = 1
        self.shape.friction = 0.5
        # self.shape.filter = pymunk.ShapeFilter(mask= pymunk.ShapeFilter.ALL_MASKS ^ 1)
        space.add(self.body, self.shape)

    # def draw(self):
    #     pygame.draw.circle(display, (55, 100, 173), convert_coordinates(self.body.position), 10)


def static_body(space, x, y):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = x, y
    space.add
    return body

class Box():
    def __init__(self, space, x, y, width, height):
        self.width = width
        self.height = height
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = x - self.width/2, y - self.height/2
        self.shape = pymunk.Poly.create_box(self.body, (self.width, self.height))
        self.shape.density = 1
        self.shape.elasticity = 1
        # self.shape.filter = pymunk.ShapeFilter(categories=1)
        space.add(self.body)

    # def draw(self):
    #     pos = convert_coordinates(self.body.position)
    #     # x2 = covnert_coordinates(self.mody.)
    #     pygame.draw.rect(display, (255, 255, 255), (*pos, self.width, self.height))

class String():
    def __init__(self, space, body1, attachment, identifier = "body"):
        self.body1 = body1
        if identifier == "body":
            self.body2 = attachment
        else:
            self.body2 = pymunk.Body(body_type=pymunk.Body.STATIC)
            self.body2.position = attachment
        joint = pymunk.PinJoint(self.body1, self.body2)
        space.add(joint)
    # def draw(self):
    #     pos1 = convert_coordinates(self.body1.position)
    #     pos2 = convert_coordinates(self.body2.position)
    #     pygame.draw.line(display, (255, 255, 255), pos1, pos2, 3)



class App:
    def __init__(self, space):
        pygame.init()
        
        self.space = space
        self.running = True
        self.clock = pygame.time.Clock()
        self.fps = 100
        self.width = 600
        self.height = 600
        self.screen_color = (28,28,28)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.draw_options = DrawOptions(self.screen)
        space.gravity = 0, 900
        space.damping = 0.95
        self.ball_color = (55, 100, 173)
        
    def run(self):
        while self.running:
            self.clock.tick(self.fps)
            self.events()
            self.draw()
            self.space.step(1/self.fps)
            # self.update()
        pygame.quit()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                
                # key inputs
                match event.key:
                    case pygame.K_LEFT:
                        cart.body.position += Vec2d(-1, 0) * 20
                    case pygame.K_RIGHT:
                        cart.body.position += Vec2d(1, 0) * 20

    
    def draw(self):
        self.screen.fill(self.screen_color)
        self.space.debug_draw(self.draw_options)
        pygame.display.update()




def create_pendulum(space, seg_len, seg_count):
    ball_list = []
    string_list = []
    cart = Box(space, 300, 300, 50, 20)
    ball_list.append(Ball(space, *random_pos_circumference(seg_len, *cart.body.position), 5))
    string_list.append(String(space, ball_list[0].body, cart.body))

    for i in range(seg_count):
        ball_list.append(Ball(space, *random_pos_circumference(seg_len, ball_list[i].body.position.x, ball_list[i].body.position.y), 5))
        string_list.append(String(space, ball_list[i].body, ball_list[i+1].body))
    
    return cart, ball_list, string_list


if __name__ == "__main__":

    space = pymunk.Space()
    
    cart, ball_list, string_list = create_pendulum(space, 50, 5)



    App(space).run()

