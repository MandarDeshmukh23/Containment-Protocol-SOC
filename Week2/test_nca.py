import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from NCA import NeuralCA, make_seed, GRID_SIZE, CHANNEL_N  # Imports from your training script

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_animation(frames, filename="nca_video.gif"):
    """Saves an array of RGBA frames into a smooth animated GIF."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')
    
    # Initialize plot with the first frame
    # Convert from BCHW tensor format to a standard numpy HWC image format
    first_frame = frames[0].detach().cpu().permute(1, 2, 0).numpy()
    # Clamp to ensure valid RGB boundaries [0, 1]
    first_frame = np.clip(first_frame, 0.0, 1.0)
    
    im = ax.imshow(first_frame)
    plt.tight_layout()

    def update(frame_idx):
        img_data = frames[frame_idx].detach().cpu().permute(1, 2, 0).numpy()
        img_data = np.clip(img_data, 0.0, 1.0)
        im.set_array(img_data)
        return [im]

    print(f"Rendering and saving {filename}...")
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
    ani.save(filename, writer='pillow', fps=20)
    plt.close()
    print(f"Successfully saved {filename}!")

def run_inference():
    # 1. Instantiate the model architecture
    model = NeuralCA(channel_n=CHANNEL_N).to(device)
    
    # --- IF YOU SAVED WEIGHTS ---
    # If you saved your weights to a file (e.g., nca_model.pth), uncomment the line below:
    # model.load_state_dict(torch.load("nca_model.pth", map_location=device))
    # -----------------------------
    model.eval() # Put model in evaluation mode (turns off gradient tracking)

    # --- TASK 3: THE GROWTH PROCESS ---
    print("\n--- Running Task 3: Growth Visualization ---")
    x = make_seed(batch_size=1) # Start with a pristine single-cell seed
    growth_history = []
    
    # Record the grid state at every single time-step during growth
    total_growth_steps = 100
    with torch.no_grad():
        for step in range(total_growth_steps):
            # Extract the visible 4 channels (RGBA) of the first batch element
            growth_history.append(x[0, :4].clone())
            x = model(x, steps=1) # Evolve the cell state by 1 step
            
    create_animation(growth_history, filename="task3_growth.gif")

    # --- TASK 4: DAMAGE AND RECOVERY ---
    print("\n--- Running Task 4: Damage and Recovery Experiment ---")
    # We take the fully grown grid 'x' from the end of the previous loop
    
    # Inflict severe damage: Slice a large circular chunk out of the shape
    print("Inflicting damage to the structure...")
    r = 6  # Damage radius
    cx, cy = GRID_SIZE // 2 + 3, GRID_SIZE // 2 + 2 # Slightly off-center slice
    
    y, x_indices = torch.meshgrid(torch.arange(GRID_SIZE), torch.arange(GRID_SIZE), indexing='ij')
    dist = (x_indices - cx)**2 + (y - cy)**2
    damage_mask = (dist > r**2).to(device).float()
    
    # Apply damage across all 16 channels simultaneously
    x[0] = x[0] * damage_mask
    
    # Record the recovery process
    recovery_history = []
    total_recovery_steps = 80
    
    with torch.no_grad():
        for step in range(total_recovery_steps):
            recovery_history.append(x[0, :4].clone())
            x = model(x, steps=1) # Let the cells communicate and repair locally
            
    create_animation(recovery_history, filename="task4_recovery.gif")

if __name__ == '__main__':
    run_inference()