import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- HYPERPARAMETERS ---
GRID_SIZE = 40          # Height and Width of the CA grid
CHANNEL_N = 16          # Total number of channels (4 visible + 12 hidden)
BATCH_SIZE = 8
POOL_SIZE = 1024
MIN_STEPS = 64
MAX_STEPS = 96
LR = 2e-3
BETAS = (0.5, 0.5)
TRAIN_ITERATIONS = 5000

# --- TARGET CREATION ---
# Create a simple target image (a filled square in the center)
# In practice, you can load an RGBA image here using PIL/Image.
def generate_target_image():
    target = torch.zeros((4, GRID_SIZE, GRID_SIZE), device=device)
    # Draw a 14x14 filled white square in the center
    start, end = GRID_SIZE // 2 - 7, GRID_SIZE // 2 + 7
    target[:, start:end, start:end] = 1.0 
    return target

TARGET_RGBA = generate_target_image()

# --- INITIALIZATION HELPER ---
def make_seed(batch_size=1):
    """Creates a blank grid tensor with a single life seed in the center."""
    grid = torch.zeros((batch_size, CHANNEL_N, GRID_SIZE, GRID_SIZE), device=device)
    mid = GRID_SIZE // 2
    # Set the seed cell's RGBA and Hidden channels
    grid[:, :4, mid, mid] = 1.0  # RGBA = 1.0 (White, opaque)
    return grid

# --- MODEL ARCHITECTURE ---
class NeuralCA(nn.Module):
    def __init__(self, channel_n=16, hidden_n=128):
        super(NeuralCA, self).__init__()
        self.channel_n = channel_n
        
        # Hardcoded Sobel filters for perception
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], 
                                [-2.0, 0.0, 2.0], 
                                [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.T
        
        # Reshape for depthwise convolution (per-channel processing)
        self.register_buffer('w_x', sobel_x.unsqueeze(0).unsqueeze(0).repeat(channel_n, 1, 1, 1))
        self.register_buffer('w_y', sobel_y.unsqueeze(0).unsqueeze(0).repeat(channel_n, 1, 1, 1))
        
        # The update network (1x1 Convolutions are mathematically equivalent to Dense/Linear per cell)
        self.update_net = nn.Sequential(
            nn.Conv2d(channel_n * 3, hidden_n, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_n, channel_n, kernel_size=1, bias=False)
        )
        
        # Crucial Initialization Trick: Zero out the weights of the final layer
        # This gives a "do-nothing" behavior at the start of training.
        nn.init.zeros_(self.update_net[-1].weight)

    def perceive(self, x):
        """Phase A: Convolve with Sobel filters and stack with identity state."""
        # Depthwise padding to preserve spatial dimensions
        grad_x = nn.functional.conv2d(x, self.w_x, padding=1, groups=self.channel_n)
        grad_y = nn.functional.conv2d(x, self.w_y, padding=1, groups=self.channel_n)
        # Concatenate identity state, x-gradients, and y-gradients along the channel axis (dim=1)
        return torch.cat([x, grad_x, grad_y], dim=1)

    def get_living_mask(self, x):
        """Identifies cells that are currently alive or adjacent to a living cell."""
        # A cell is alive if its alpha channel > 0.1
        alpha = x[:, 3:4, :, :]
        living = nn.functional.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1
        return living.float()

    def forward(self, x, steps):
        for _ in range(steps):
            pre_life_mask = self.get_living_mask(x)
            
            # 1. Perceive neighbors
            perception = self.perceive(x)
            
            # 2. Compute update vector via shared network
            dx = self.update_net(perception)
            
            # 3. Stochastic update mask (50% asynchronous execution)
            stochastic_mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) > 0.5).float()
            x = x + dx * stochastic_mask
            
            # 4. Post-life masking to zero out truly dead cells
            post_life_mask = self.get_living_mask(x)
            life_mask = pre_life_mask * post_life_mask
            x = x * life_mask
            
        return x

# --- SAMPLE POOL CLASS ---
class SamplePool:
    def __init__(self, pool_size, channel_n, grid_size):
        self.pool_size = pool_size
        # Initialize pool completely with pristine seeds
        self.slots = make_seed(pool_size).cpu()

    def sample(self, batch_size):
        idx = np.random.choice(self.pool_size, batch_size, replace=False)
        batch = self.slots[idx].to(device)
        return batch, idx

    def commit(self, batch, idx):
        self.slots[idx] = batch.detach().cpu()

# --- DAMAGE HELPER (For Task 4) ---
def inflict_damage(batch):
    """Blasts a circular hole in the first grid of the batch to train regeneration."""
    # Only damage a single entry in the batch to keep gradients stable
    r = 5  # Damage radius
    cx, cy = np.random.randint(GRID_SIZE // 4, 3 * GRID_SIZE // 4, 2)
    
    y, x = torch.meshgrid(torch.arange(GRID_SIZE), torch.arange(GRID_SIZE), indexing='ij')
    dist = (x - cx)**2 + (y - cy)**2
    mask = (dist > r**2).to(device).float()
    
    batch[0] = batch[0] * mask
    return batch

# --- TRAINING LOOP ---
def train():
    model = NeuralCA(channel_n=CHANNEL_N).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=BETAS)
    scheduler = optim.lr_scheduler.MultiStepBreakpoints = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)
    
    pool = SamplePool(POOL_SIZE, CHANNEL_N, GRID_SIZE)
    
    print("Starting Training Loop...")
    for i in range(TRAIN_ITERATIONS):
        # 1. Sample from pool
        batch, idx = pool.sample(BATCH_SIZE)
        
        # Sort by loss to find the highest error slot to replace with a seed (1 in 8 reset strategy)
        # For simplicity, we systematically replace the last index of the batch with a true seed
        batch[-1] = make_seed(1)[0]
        
        # 2. Inject damage to force self-repair properties
        batch = inflict_damage(batch)
        
        # 3. Select random step depth
        steps = np.random.randint(MIN_STEPS, MAX_STEPS)
        
        # 4. Forward Pass
        optimizer.zero_grad()
        output_grid = model(batch, steps)
        
        # 5. L2 Loss calculated ONLY on the visible RGBA channels (first 4 channels)
        outputs_rgba = output_grid[:, :4, :, :]
        loss = nn.functional.mse_loss(outputs_rgba, TARGET_RGBA.unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1))
        
        # 6. Backward pass through time
        loss.backward()
        
        # Gradient clipping to prevent exploding updates
        for p in model.parameters():
            if p.grad is not None:
                p.grad.clamp_(-0.1, 0.1)
                
        optimizer.step()
        scheduler.step()
        
        # 7. Commit updated grids back to memory pool
        pool.commit(output_grid, idx)
        
        if i % 100 == 0:
            print(f"Iteration {i:04d} | Loss: {loss.item():.6f}")

if __name__ == '__main__':
    train()