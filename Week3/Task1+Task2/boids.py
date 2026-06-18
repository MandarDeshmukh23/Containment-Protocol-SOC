import pygame
import random

# --- Configuration & Parameters ---
WIDTH, HEIGHT = 800, 600
NUM_BOIDS = 100 # Minimum requirement
FPS = 60

# Adjustable Parameters
PERCEPTION_RADIUS = 50.0
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.0
SEPARATION_WEIGHT = 1.5
MAX_SPEED = 4.0
MAX_FORCE = 0.1

class Boid:
    def __init__(self):
        self.position = pygame.Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        self.velocity = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        if self.velocity.length() > 0:
            self.velocity.scale_to_length(MAX_SPEED)
        self.acceleration = pygame.Vector2(0, 0)

    def edges(self):
        """Wraps the boid around the screen edges."""
        if self.position.x > WIDTH: self.position.x = 0
        elif self.position.x < 0: self.position.x = WIDTH
        if self.position.y > HEIGHT: self.position.y = 0
        elif self.position.y < 0: self.position.y = HEIGHT

    def apply_behaviors(self, boids):
        alignment = pygame.Vector2(0, 0)
        cohesion = pygame.Vector2(0, 0)
        separation = pygame.Vector2(0, 0)
        total = 0

        for other in boids:
            if other != self:
                distance = self.position.distance_to(other.position)
                if distance < PERCEPTION_RADIUS:
                    alignment += other.velocity
                    cohesion += other.position
                    
                    # Weight separation by inverse distance
                    diff = self.position - other.position
                    if distance > 0:
                        diff /= distance 
                    separation += diff
                    total += 1

        if total > 0:
            # Alignment
            alignment /= total
            alignment.scale_to_length(MAX_SPEED)
            alignment -= self.velocity
            if alignment.length() > MAX_FORCE:
                alignment.scale_to_length(MAX_FORCE)

            # Cohesion
            cohesion /= total
            cohesion -= self.position
            cohesion.scale_to_length(MAX_SPEED)
            cohesion -= self.velocity
            if cohesion.length() > MAX_FORCE:
                cohesion.scale_to_length(MAX_FORCE)

            # Separation
            separation /= total
            separation.scale_to_length(MAX_SPEED)
            separation -= self.velocity
            if separation.length() > MAX_FORCE:
                separation.scale_to_length(MAX_FORCE)

        self.acceleration += alignment * ALIGNMENT_WEIGHT
        self.acceleration += cohesion * COHESION_WEIGHT
        self.acceleration += separation * SEPARATION_WEIGHT

    def update(self):
        self.position += self.velocity
        self.velocity += self.acceleration
        if self.velocity.length() > MAX_SPEED:
            self.velocity.scale_to_length(MAX_SPEED)
        self.acceleration *= 0 # Reset acceleration each frame

    def draw(self, surface):
        # Draw the boid as a simple circle (can be upgraded to a triangle reflecting heading)
        pygame.draw.circle(surface, (200, 200, 200), (int(self.position.x), int(self.position.y)), 3)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids Simulation")
    clock = pygame.time.Clock()

    flock = [Boid() for _ in range(NUM_BOIDS)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Compute behaviors
        for boid in flock:
            boid.apply_behaviors(flock)

        # Update physics
        for boid in flock:
            boid.update()
            boid.edges()

        # Render
        screen.fill((30, 30, 30))
        for boid in flock:
            boid.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()