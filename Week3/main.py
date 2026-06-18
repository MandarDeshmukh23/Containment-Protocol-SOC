import pygame
import random
import math

# --- Configuration ---
WIDTH, HEIGHT = 1000, 700
FPS = 60

# --- Global Parameters ---
PARAMS = {
    'radius': 50.0,
    'separation': 1.5,
    'alignment': 1.0,
    'cohesion': 1.0,
    'avoidance': 4.0,
    'chase': 3.0,
    'flee': 5.0,
    'transmission': 30.0,
    'evacuation': 4.0,
    'max_speed': 4.0,
    'max_force': 0.1,
    'pred_speed': 4.5,
    'pred_force': 0.15
}

# --- Environment Objects ---
OBSTACLES = [
    {'type': 'circle', 'pos': pygame.Vector2(WIDTH//2, HEIGHT//2), 'radius': 60},
    {'type': 'rect', 'rect': pygame.Rect(200, 200, 30, 300)},
    {'type': 'rect', 'rect': pygame.Rect(WIDTH - 230, 200, 30, 300)}
]
SAFE_ZONE = {'pos': pygame.Vector2(WIDTH - 100, 100), 'radius': 50}

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

class Agent:
    def __init__(self, is_predator=False):
        self.pos = pygame.Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        speed = PARAMS['pred_speed'] if is_predator else PARAMS['max_speed']
        self.vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        if self.vel.length() > 0: self.vel.scale_to_length(speed)
        self.acc = pygame.Vector2(0, 0)
        
        self.is_predator = is_predator
        self.informed = False

    def edges(self):
        # Wrap around edges
        margin = 20
        if self.pos.x > WIDTH + margin: self.pos.x = -margin
        elif self.pos.x < -margin: self.pos.x = WIDTH + margin
        if self.pos.y > HEIGHT + margin: self.pos.y = -margin
        elif self.pos.y < -margin: self.pos.y = HEIGHT + margin

    def apply_behaviors(self, agents, mode):
        if self.is_predator:
            self.hunt(agents)
        else:
            self.flock(agents, mode)

    def hunt(self, agents):
        chase_force = pygame.Vector2(0, 0)
        closest_dist = float('inf')
        closest_prey = None

        for other in agents:
            if not other.is_predator:
                d = self.pos.distance_to(other.pos)
                if d < PARAMS['radius'] * 3 and d < closest_dist:
                    closest_dist = d
                    closest_prey = other

        if closest_prey:
            chase_force = closest_prey.pos - self.pos
            if chase_force.length() > 0:
                chase_force.scale_to_length(PARAMS['pred_speed'])
                chase_force -= self.vel
                if chase_force.length() > PARAMS['pred_force']:
                    chase_force.scale_to_length(PARAMS['pred_force'])

        self.acc += chase_force * PARAMS['chase']

    def flock(self, agents, mode):
        alignment = pygame.Vector2(0, 0)
        cohesion = pygame.Vector2(0, 0)
        separation = pygame.Vector2(0, 0)
        flee = pygame.Vector2(0, 0)
        evac = pygame.Vector2(0, 0)
        avoid = pygame.Vector2(0, 0)
        total = 0

        # 1. Agent Interactions
        for other in agents:
            if other != self:
                d = self.pos.distance_to(other.pos)
                
                # Mode 4: Predator/Prey logic
                if mode == 4 and other.is_predator:
                    if d < PARAMS['radius'] * 2:
                        diff = self.pos - other.pos
                        if d > 0: diff /= d
                        flee += diff
                
                # Standard Flocking
                elif not other.is_predator and d < PARAMS['radius']:
                    # Mode 5: Information Spread logic
                    if mode == 5 and self.informed and not other.informed and d < PARAMS['transmission']:
                        other.informed = True

                    alignment += other.vel
                    cohesion += other.pos
                    diff = self.pos - other.pos
                    if d > 0: diff /= d
                    separation += diff
                    total += 1

        if total > 0:
            alignment /= total
            if alignment.length() > 0:
                alignment.scale_to_length(PARAMS['max_speed'])
                alignment -= self.vel
                if alignment.length() > PARAMS['max_force']: alignment.scale_to_length(PARAMS['max_force'])

            cohesion /= total
            cohesion -= self.pos
            if cohesion.length() > 0:
                cohesion.scale_to_length(PARAMS['max_speed'])
                cohesion -= self.vel
                if cohesion.length() > PARAMS['max_force']: cohesion.scale_to_length(PARAMS['max_force'])

            separation /= total
            if separation.length() > 0:
                separation.scale_to_length(PARAMS['max_speed'])
                separation -= self.vel
                if separation.length() > PARAMS['max_force']: separation.scale_to_length(PARAMS['max_force'])

        if flee.length() > 0:
            flee.scale_to_length(PARAMS['max_speed'])
            flee -= self.vel
            if flee.length() > PARAMS['max_force'] * 2: flee.scale_to_length(PARAMS['max_force'] * 2)

        # 2. Obstacle Avoidance (Mode 3)
        if mode == 3:
            for obs in OBSTACLES:
                closest = pygame.Vector2(0, 0)
                obs_r = 0
                if obs['type'] == 'circle':
                    closest = obs['pos']
                    obs_r = obs['radius']
                elif obs['type'] == 'rect':
                    closest.x = clamp(self.pos.x, obs['rect'].left, obs['rect'].right)
                    closest.y = clamp(self.pos.y, obs['rect'].top, obs['rect'].bottom)
                    obs_r = 15 # Padding

                d = self.pos.distance_to(closest)
                if 0 < d < obs_r + PARAMS['radius']:
                    diff = self.pos - closest
                    diff.normalize_ip()
                    diff /= d # Stronger closer
                    avoid += diff
            
            if avoid.length() > 0:
                avoid.scale_to_length(PARAMS['max_speed'])
                avoid -= self.vel
                if avoid.length() > PARAMS['max_force'] * 2: avoid.scale_to_length(PARAMS['max_force'] * 2)

        # 3. Evacuation (Mode 5)
        if mode == 5 and self.informed:
            d_to_safe = self.pos.distance_to(SAFE_ZONE['pos'])
            if d_to_safe > SAFE_ZONE['radius']:
                diff = SAFE_ZONE['pos'] - self.pos
                if diff.length() > 0:
                    diff.scale_to_length(PARAMS['max_speed'])
                    evac = diff - self.vel
                    if evac.length() > PARAMS['max_force'] * 1.5: evac.scale_to_length(PARAMS['max_force'] * 1.5)
            else:
                self.vel *= 0.95 # Slow down inside safe zone

        # 4. Apply weighted forces
        self.acc += separation * PARAMS['separation']
        self.acc += alignment * PARAMS['alignment']
        self.acc += cohesion * PARAMS['cohesion']
        self.acc += avoid * PARAMS['avoidance']
        self.acc += flee * PARAMS['flee']
        self.acc += evac * PARAMS['evacuation']

    def update(self):
        max_s = PARAMS['pred_speed'] if self.is_predator else PARAMS['max_speed']
        self.pos += self.vel
        self.vel += self.acc
        if self.vel.length() > max_s:
            self.vel.scale_to_length(max_s)
        self.acc *= 0 # Reset

    def draw(self, surface):
        angle = math.atan2(self.vel.y, self.vel.x)
        
        if self.is_predator:
            color = (255, 50, 50) # Red
            size = 12
        elif self.informed:
            color = (255, 204, 0) # Yellow
            size = 8
        else:
            color = (0, 255, 204) # Cyan
            size = 8

        # Calculate triangle vertices based on velocity heading
        p1 = self.pos + pygame.Vector2(size, 0).rotate_rad(angle)
        p2 = self.pos + pygame.Vector2(-size/2, size/2).rotate_rad(angle)
        p3 = self.pos + pygame.Vector2(-size/2, -size/2).rotate_rad(angle)
        
        pygame.draw.polygon(surface, color, [p1, p2, p3])


def spawn_population(mode):
    population = []
    # Standard 150 agents
    for _ in range(150):
        population.append(Agent(is_predator=False))
    
    if mode == 4:
        for _ in range(3):
            population.append(Agent(is_predator=True))
            
    if mode == 5:
        # Pick one random agent to be informed
        population[random.randint(0, 149)].informed = True
        
    return population

def draw_environment(surface, mode):
    if mode == 3:
        for obs in OBSTACLES:
            if obs['type'] == 'circle':
                pygame.draw.circle(surface, (100, 30, 30), (int(obs['pos'].x), int(obs['pos'].y)), obs['radius'])
                pygame.draw.circle(surface, (200, 50, 50), (int(obs['pos'].x), int(obs['pos'].y)), obs['radius'], 2)
            elif obs['type'] == 'rect':
                pygame.draw.rect(surface, (100, 30, 30), obs['rect'])
                pygame.draw.rect(surface, (200, 50, 50), obs['rect'], 2)
                
    if mode == 5:
        pygame.draw.circle(surface, (30, 100, 50), (int(SAFE_ZONE['pos'].x), int(SAFE_ZONE['pos'].y)), SAFE_ZONE['radius'])
        pygame.draw.circle(surface, (50, 200, 100), (int(SAFE_ZONE['pos'].x), int(SAFE_ZONE['pos'].y)), SAFE_ZONE['radius'], 2)

def draw_ui(surface, mode, font):
    mode_text = {
        1: "Task 1 & 2: Standard Flocking",
        3: "Task 3: Complex Obstacles",
        4: "Task 4: Predator & Prey",
        5: "Task 5: Information Spread"
    }
    
    instructions = [
        mode_text[mode],
        "-------------------",
        "Press 1: Standard",
        "Press 3: Obstacles",
        "Press 4: Predators",
        "Press 5: Evacuation",
        "Press R: Reset Swarm"
    ]
    
    for i, text in enumerate(instructions):
        color = (255, 255, 255) if i == 0 else (150, 150, 150)
        img = font.render(text, True, color)
        surface.blit(img, (20, 20 + (i * 25)))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Multi-Agent Systems: Boids Testbed")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 18)

    current_mode = 1
    population = spawn_population(current_mode)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    current_mode = 1
                    population = spawn_population(current_mode)
                elif event.key == pygame.K_3:
                    current_mode = 3
                    population = spawn_population(current_mode)
                elif event.key == pygame.K_4:
                    current_mode = 4
                    population = spawn_population(current_mode)
                elif event.key == pygame.K_5:
                    current_mode = 5
                    population = spawn_population(current_mode)
                elif event.key == pygame.K_r:
                    population = spawn_population(current_mode)

        # Update physics
        for agent in population:
            agent.apply_behaviors(population, current_mode)

        for agent in population:
            agent.update()
            agent.edges()

        # Render
        screen.fill((30, 30, 30))
        draw_environment(screen, current_mode)
        
        for agent in population:
            agent.draw(screen)
            
        draw_ui(screen, current_mode, font)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()