import os
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from io import BytesIO

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- HYPERPARAMETERS ---
GRID_SIZE = 56          # Upscaled slightly to capture emoji detail nicely
CHANNEL_N = 16          # 4 visible (RGBA) + 12 hidden
BATCH_SIZE = 8
POOL_SIZE = 1024
MIN_STEPS = 64
MAX_STEPS = 96
LR = 2e-3
BETAS = (0.5, 0.5)
TRAIN_ITERATIONS = 4000  # Adjust as needed for convergence

# --- STEP 1: LOAD & PREPROCESS TARGET EMOJI ---
def load_emoji_target(emoji_char="🦎"):
    """
    Downloads an open-source Noto Color Emoji from GitHub, 
    resizes it, and converts it into an RGBA PyTorch tensor.
    """
    # Convert emoji character to its hex unicode string (e.g., '🦎' -> '1f98e')
    hex_code = "-".join([f"{ord(c):x}" for c in emoji_char])
    url = f"https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/128/emoji_u{hex_code}.png"
    
    print(f"Fetching emoji '{emoji_char}' from: {url}")
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGBA")
    except Exception as e:
        print(f"Failed to fetch custom emoji ({e}). Falling back to a fallback URL.")
        # Fallback to a reliable standard lizard emoji asset if request fails
        url = "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/128/emoji_u1f98e.png"
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGBA")

    # Resize to our grid dimensions using high-quality resampling
    img = img.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.LANCZOS)
    img_np = np.asarray(img, dtype=np.float32) / 255.0
    
    # Standardize Alpha pre-multiplied colors (matching the Distill implementation)
    # This cleans up dark edges around transparent boundaries
    alpha = img_np[..., 3:4]
    img_np[..., :3] = img_np[..., :3] * alpha
    
    # Convert from HWC to BCHW tensor format
    target_tensor = torch.tensor(img_np, device=device).permute(2, 0, 1)
    return target_tensor

# Load your target (Change the character to try different emojis like "🦎", "👁️", "🌳", "🐠")
TARGET_RGBA = load_emoji_target("🦎")

# --- STEP 2: MODEL INITIALIZATION HELPERS ---
def make_seed(batch_size=1):
    """Initializes a grid with all zeros except a 1-pixel seed in the center."""
    grid = torch.zeros((batch_size, CHANNEL_N, GRID_SIZE, GRID_SIZE), device=device)
    mid = GRID_SIZE // 2
    # The seed cell has visible channels set to black/opaque, and alpha activated
    grid[:, 3, mid, mid] = 1.0  # Alpha = 1.0
    grid[:, :3, mid, mid] = 1.0 # White core to initiate RGB color propagation
    return grid

# --- STEP 3: NCA MODEL ARCHITECTURE ---
class NeuralCA(nn.Module):
    def __init__(self, channel_n=16, hidden_n=128):
        super(NeuralCA, self).__init__()
        self.channel_n = channel_n
        
        # Hardcoded Sobel filters for spatial derivative gradients
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], 
                                [-2.0, 0.0, 2.0], 
                                [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.T
        
        self.register_buffer('w_x', sobel_x.unsqueeze(0).unsqueeze(0).repeat(channel_n, 1, 1, 1))
        self.register_buffer('w_y', sobel_y.unsqueeze(0).unsqueeze(0).repeat(channel_n, 1, 1, 1))
        
        # 1x1 convolutions act as our shared per-cell dense neural network layers
        self.update_net = nn.Sequential(
            nn.Conv2d(channel_n * 3, hidden_n, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_n, channel_n, kernel_size=1, bias=False)
        )
        # Initialization trick: Set final weights to 0 for a stable 'do-nothing' initial state
        nn.init.zeros_(self.update_net[-1].weight)

    def perceive(self, x):
        """Phase A: Convolve with Sobel filters and stack with the identity state."""
        grad_x = nn.functional.conv2d(x, self.w_x, padding=1, groups=self.channel_n)
        grad_y = nn.functional.conv2d(x, self.w_y, padding=1, groups=self.channel_n)
        return torch.cat([x, grad_x, grad_y], dim=1)

    def get_living_mask(self, x):
        """Identifies active tissue cells based on an Alpha threshold > 0.1."""
        alpha = x[:, 3:4, :, :]
        living = nn.functional.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1
        return living.float()

    def forward(self, x, steps):
        for _ in range(steps):
            pre_life_mask = self.get_living_mask(x)
            
            perception = self.perceive(x)
            dx = self.update_net(perception)
            
            # Stochastic Update Strategy (50% random execution probability per cell)
            stochastic_mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) > 0.5).float()
            x = x + dx * stochastic_mask
            
            # Post-life Masking Strategy (Kills off dead or completely detached cells)
            post_life_mask = self.get_living_mask(x)
            x = x * (pre_life_mask * post_life_mask)
        return x

# --- STEP 4: MEMORY POOL & REGENERATION STRATEGY ---
class SamplePool:
    def __init__(self, pool_size, channel_n, grid_size):
        self.pool_size = pool_size
        self.slots = make_seed(pool_size).cpu()

    def sample(self, batch_size):
        idx = np.random.choice(self.pool_size, batch_size, replace=False)
        batch = self.slots[idx].to(device)
        return batch, idx

    def commit(self, batch, idx):
        self.slots[idx] = batch.detach().cpu()

def inflict_damage(batch):
    """Regeneration Strategy: Slices a chunk out of a grid to force healing gradients."""
    # We only damage the first element in the batch to keep optimization metrics stable
    r = np.random.randint(7, 11)  # Variable damage size
    cx, cy = np.random.randint(GRID_SIZE // 3, 2 * GRID_SIZE // 3, 2)
    
    y, x = torch.meshgrid(torch.arange(GRID_SIZE), torch.arange(GRID_SIZE), indexing='ij')
    dist = (x - cx)**2 + (y - cy)**2
    mask = (dist > r**2).to(device).float()
    
    batch[0] = batch[0] * mask
    return batch

# --- STEP 5: VISUALIZATION FUNCTIONS (TESTING) ---
def save_nca_animation(frames, filename):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')
    
    # Convert from BCHW tensor back to conventional HWC image
    first_frame = frames[0].detach().cpu().permute(1, 2, 0).numpy()
    # Un-multiply alpha for cleaner image rendering in matplotlib
    alpha = np.clip(first_frame[..., 3:4], 1e-8, 1.0)
    first_frame[..., :3] = np.clip(first_frame[..., :3] / alpha, 0.0, 1.0)
    first_frame = np.clip(first_frame, 0.0, 1.0)
    
    im = ax.imshow(first_frame)
    plt.tight_layout()

    def update(i):
        img_data = frames[i].detach().cpu().permute(1, 2, 0).numpy()
        a = np.clip(img_data[..., 3:4], 1e-8, 1.0)
        img_data[..., :3] = np.clip(img_data[..., :3] / a, 0.0, 1.0)
        img_data = np.clip(img_data, 0.0, 1.0)
        im.set_array(img_data)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=40, blit=True)
    ani.save(filename, writer='pillow', fps=25)
    plt.close()
    print(f"Saved: {filename}")

# --- STEP 6: CORE EXECUTION PIPELINE ---
def main():
    model = NeuralCA(channel_n=CHANNEL_N).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=BETAS)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)
    
    pool = SamplePool(POOL_SIZE, CHANNEL_N, GRID_SIZE)
    
    print("\n================== TRAINING PHASE ==================")
    for i in range(TRAIN_ITERATIONS):
        batch, idx = pool.sample(BATCH_SIZE)
        
        # Stabilization Strategy (1-in-8 reset): Wipes out one old grid and replaces it 
        # with a true seed to keep the network active at generating structures from scratch.
        batch[-1] = make_seed(1)[0]
        
        # Regeneration Strategy: Inflict damage on an already grown structure
        if i > 200:  # Start damaging once the pool has developed some basic structure
            batch = inflict_damage(batch)
            
        steps = np.random.randint(MIN_STEPS, MAX_STEPS)
        
        optimizer.zero_grad()
        output_grid = model(batch, steps)
        
        # Evaluate loss strictly against the visible 4 channels (RGBA)
        outputs_rgba = output_grid[:, :4, :, :]
        loss = nn.functional.mse_loss(outputs_rgba, TARGET_RGBA.unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1))
        
        loss.backward()
        
        # Gradient clipping to counter vanishing/exploding gradients over multi-step unrolls
        for p in model.parameters():
            if p.grad is not None:
                p.grad.clamp_(-0.1, 0.1)
                
        optimizer.step()
        scheduler.step()
        
        pool.commit(output_grid, idx)
        
        if i % 100 == 0:
            print(f"Iteration {i:04d} | Current Objective Loss: {loss.item():.6f}")

    print("\n================== TESTING PHASE ==================")
    model.eval()
    
    # Test 1: Growth Process
    print("Testing Growth from single seed...")
    x = make_seed(batch_size=1)
    growth_frames = []
    with torch.no_grad():
        for _ in range(120): # Record 120 steps of structural development
            growth_frames.append(x[0, :4].clone())
            x = model(x, steps=1)
    save_nca_animation(growth_frames, "emoji_growth.gif")

    # Test 2: Damage Recovery
    print("Testing Regeneration from dynamic damage...")
    # Slicing the right half of the emoji completely away
    r = 10
    cx, cy = GRID_SIZE // 2 + 4, GRID_SIZE // 2
    y, x_idx = torch.meshgrid(torch.arange(GRID_SIZE), torch.arange(GRID_SIZE), indexing='ij')
    dist = (x_idx - cx)**2 + (y - cy)**2
    mask = (dist > r**2).to(device).float()
    
    # Inflict damage across all 16 internal channels
    x[0] = x[0] * mask
    
    recovery_frames = []
    with torch.no_grad():
        for _ in range(100):
            recovery_frames.append(x[0, :4].clone())
            x = model(x, steps=1)
    save_nca_animation(recovery_frames, "emoji_regeneration.gif")
    print("Inference Testing Complete! Check your local folder for .gif files.")

if __name__ == '__main__':
    main()