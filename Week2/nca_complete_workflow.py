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
GRID_SIZE = 56          # Height and Width of the CA grid
CHANNEL_N = 16          # 4 visible (RGBA) + 12 hidden channels
BATCH_SIZE = 8
POOL_SIZE = 1024
MIN_STEPS = 64
MAX_STEPS = 96
LR = 2e-3
BETAS = (0.5, 0.5)
TRAIN_ITERATIONS = 4000  # Adjust as needed for convergence

# --- STEP 1: LOAD TARGET IMAGE ---
def load_emoji_target(emoji_char="🦎"):
    """Downloads a Noto Color Emoji and converts it to an RGBA tensor."""
    hex_code = "-".join([f"{ord(c):x}" for c in emoji_char])
    url = f"https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/128/emoji_u{hex_code}.png"
    
    print(f"Fetching target emoji '{emoji_char}'...")
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGBA")
    except Exception:
        url = "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/128/emoji_u1f98e.png"
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGBA")

    img = img.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.LANCZOS)
    img_np = np.asarray(img, dtype=np.float32) / 255.0
    
    # Pre-multiply alpha channel to avoid dark border artifacts
    alpha = img_np[..., 3:4]
    img_np[..., :3] = img_np[..., :3] * alpha
    
    return torch.tensor(img_np, device=device).permute(2, 0, 1)

TARGET_RGBA = load_emoji_target("🦎")

# --- STEP 2: SEED INITIALIZATION ---
def make_seed(batch_size=1):
    """Generates a blank grid tensor with a single center pixel seed."""
    grid = torch.zeros((batch_size, CHANNEL_N, GRID_SIZE, GRID_SIZE), device=device)
    mid = GRID_SIZE // 2
    grid[:, 3, mid, mid] = 1.0   # Set Alpha = 1.0
    grid[:, :3, mid, mid] = 1.0  # Set RGB = 1.0 (White core)
    return grid

# --- STEP 3: MODEL ARCHITECTURE (Task 2 Analysis Alignment) ---
class NeuralCA(nn.Module):
    def __init__(self, channel_n=16, hidden_n=128):
        super(NeuralCA, self).__init__()
        self.channel_n = channel_n
        
        # Hardcoded Sobel Kernels for local gradient perception
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], 
                                [-2.0, 0.0, 2.0], 
                                [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.T
        
        # Register buffers for depthwise per-channel convolutions
        self.register_buffer('w_x', sobel_x.unsqueeze(0).unsqueeze(0).repeat(channel_n, 1, 1, 1))
        self.register_buffer('w_y', sobel_y.unsqueeze(0).unsqueeze(0).repeat(channel_n, 1, 1, 1))
        
        # Shared Update Network (1x1 Convolutions act as dense layers for every single cell)
        self.update_net = nn.Sequential(
            nn.Conv2d(channel_n * 3, hidden_n, kernel_size=1), # Input: 48 channels (16 identity + 32 gradients)
            nn.ReLU(),
            nn.Conv2d(hidden_n, channel_n, kernel_size=1, bias=False) # Output: 16 delta update channels
        )
        nn.init.zeros_(self.update_net[-1].weight) # Zero-initialize final layer for initial stability

    def perceive(self, x):
        grad_x = nn.functional.conv2d(x, self.w_x, padding=1, groups=self.channel_n)
        grad_y = nn.functional.conv2d(x, self.w_y, padding=1, groups=self.channel_n)
        return torch.cat([x, grad_x, grad_y], dim=1) # Shape: [B, 48, H, W]

    def get_living_mask(self, x):
        """Cells are alive if their alpha channel or any neighbor's alpha channel > 0.1."""
        alpha = x[:, 3:4, :, :]
        living = nn.functional.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1
        return living.float()

    def forward(self, x, steps=1):
        for _ in range(steps):
            pre_life_mask = self.get_living_mask(x)
            
            perception = self.perceive(x)
            dx = self.update_net(perception)
            
            # Stochastic Update Mask (asynchronous firing probability of 50%)
            stochastic_mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) > 0.5).float()
            x = x + dx * stochastic_mask
            
            # Post-life Masking to clean un-anchored cells
            post_life_mask = self.get_living_mask(x)
            x = x * (pre_life_mask * post_life_mask)
        return x

# --- STEP 4: SAMPLE POOL MANAGEMENT ---
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

def inflict_training_damage(batch):
    """Regeneration Strategy: Erase circular patches from random areas."""
    r = np.random.randint(6, 10)
    cx, cy = np.random.randint(GRID_SIZE // 4, 3 * GRID_SIZE // 4, 2)
    y, x = torch.meshgrid(torch.arange(GRID_SIZE), torch.arange(GRID_SIZE), indexing='ij')
    dist = (x - cx)**2 + (y - cy)**2
    mask = (dist > r**2).to(device).float()
    batch[0] = batch[0] * mask
    return batch

# --- STEP 5: VISUALIZATION FORMATTING HELPERS ---
def tensor_to_rgb_image(tensor_chw):
    """Converts a [4, H, W] RGBA tensor to a standard normalized [H, W, 3] RGB image."""
    np_img = tensor_chw.detach().cpu().permute(1, 2, 0).numpy()
    alpha = np.clip(np_img[..., 3:4], 1e-8, 1.0)
    # Straighten colors from pre-multiplied alpha state for matplotlib
    rgb = np.clip(np_img[..., :3] / alpha, 0.0, 1.0)
    return rgb

def save_gif(frames, filename):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')
    im = ax.imshow(tensor_to_rgb_image(frames[0]))
    plt.tight_layout()

    def update(i):
        im.set_array(tensor_to_rgb_image(frames[i]))
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=40, blit=True)
    ani.save(filename, writer='pillow', fps=25)
    plt.close()
    print(f"Saved animation to: {filename}")

# --- STEP 6: PIPELINE RUNNER ---
def main():
    model = NeuralCA(channel_n=CHANNEL_N).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=BETAS)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)
    pool = SamplePool(POOL_SIZE, CHANNEL_N, GRID_SIZE)
    
    print("\n================== TASK 1 & 2: TRAINING MODEL ==================")
    for i in range(TRAIN_ITERATIONS + 1):
        batch, idx = pool.sample(BATCH_SIZE)
        
        # Stabilization Strategy: Replace one sample with a pristine seed
        batch[-1] = make_seed(1)[0]
        
        # Regeneration Strategy: Apply cell erasure after the first 200 epochs
        if i > 200:
            batch = inflict_training_damage(batch)
            
        steps = np.random.randint(MIN_STEPS, MAX_STEPS)
        
        optimizer.zero_grad()
        output_grid = model(batch, steps)
        
        # Training Objective Function: L2 Loss on visible RGBA channels
        outputs_rgba = output_grid[:, :4, :, :]
        loss = nn.functional.mse_loss(outputs_rgba, TARGET_RGBA.unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1))
        
        loss.backward()
        
        # Gradient clipping to prevent exploding updates across temporal unrolls
        for p in model.parameters():
            if p.grad is not None:
                p.grad.clamp_(-0.1, 0.1)
                
        optimizer.step()
        scheduler.step()
        pool.commit(output_grid, idx)
        
        if i % 200 == 0:
            print(f"Iteration {i:04d} | Current Objective Loss: {loss.item():.6f}")

    # Switch to Evaluation Mode
    model.eval()

    print("\n================== TASK 3: GROWTH VISUALIZATION ==================")
    x = make_seed(batch_size=1)
    growth_frames = []
    
    # Track discrete intervals for a clean milestone overview plot
    milestones = {0: None, 20: None, 50: None, 100: None}
    
    with torch.no_grad():
        for step in range(101):
            growth_frames.append(x[0, :4].clone())
            if step in milestones:
                milestones[step] = tensor_to_rgb_image(x[0, :4])
            x = model(x, steps=1)
            
    # Save the continuous growth process as an animation
    save_gif(growth_history := growth_frames, "task3_growth_process.gif")
    
    # Render static structural milestones
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, (step, img) in zip(axes, milestones.items()):
        ax.imshow(img)
        ax.set_title(f"Step {step}")
        ax.axis('off')
    plt.suptitle("Task 3: Morphogenesis Timeline (Initial Seed -> Intermediate -> Final)", fontsize=14)
    plt.savefig("task3_milestones.png", bbox_inches='tight')
    plt.close()
    print("Saved static snapshot grid to: task3_milestones.png")

    print("\n================== TASK 4: DAMAGE & RECOVERY EXPERIMENT ==================")
    # Inflict damage to the fully grown cell state array 'x'
    print("Applying structural laceration mask...")
    r_damage = 9
    cx, cy = GRID_SIZE // 2 + 5, GRID_SIZE // 2  # Off-center cut across borders
    y_grid, x_grid = torch.meshgrid(torch.arange(GRID_SIZE), torch.arange(GRID_SIZE), indexing='ij')
    dist_grid = (x_grid - cx)**2 + (y_grid - cy)**2
    damage_mask = (dist_grid > r_damage**2).to(device).float()
    
    # Completely drop out information across all 16 channels in the chosen radius
    x[0] = x[0] * damage_mask
    
    recovery_frames = []
    recovery_milestones = {0: "Immediately Damaged", 15: "Regenerating Edges", 40: "Forming Contours", 80: "Fully Recovered"}
    recovery_snapshots = {}
    
    with torch.no_grad():
        for step in range(81):
            recovery_frames.append(x[0, :4].clone())
            if step in recovery_milestones:
                recovery_snapshots[step] = tensor_to_rgb_image(x[0, :4])
            x = model(x, steps=1)
            
    # Save the continuous healing process as an animation
    save_gif(recovery_frames, "task4_recovery_process.gif")
    
    # Render static healing timeline
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, (step, label) in zip(axes, recovery_milestones.items()):
        ax.imshow(recovery_snapshots[step])
        ax.set_title(f"Step {step}\n{label}")
        ax.axis('off')
    plt.suptitle("Task 4: Structural Damage and Autonomous Recovery", fontsize=14)
    plt.savefig("task4_recovery_milestones.png", bbox_inches='tight')
    plt.close()
    print("Saved static snapshot grid to: task4_recovery_milestones.png")
    print("\nAll Tasks Completed Successfully.")

if __name__ == '__main__':
    main()