import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time

def init_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    
    print("Warming up camera...")
    for _ in range(15):
        pipeline.wait_for_frames()
    return pipeline, align

def get_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        return None, None, None
        
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return color_image, depth_image, intrinsics

def process_3d_data(depth_image, intrinsics, step=30, grid_size=12):
    h, w = depth_image.shape
    
    # 1. Smooth the raw depth data to reduce chaotic normal arrows
    smoothed_depth = gaussian_filter(depth_image, sigma=2.0)
    
    # Downsample
    y, x = np.mgrid[0:h:step, 0:w:step]
    z = smoothed_depth[y, x] * 0.001 
    
    # 2. Deproject to 3D Space (Structured Grid)
    X = (x - intrinsics.ppx) * z / intrinsics.fx
    Y = (y - intrinsics.ppy) * z / intrinsics.fy
    Z = z
    
    # Mask out invalid (zero-depth) areas
    valid = Z > 0
    
    # 3. Calculate True 3D Surface Normals (Cross Product of Gradients)
    # Compute gradients along the X and Y axes of the grid
    dXdy, dXdx = np.gradient(X)
    dYdy, dYdx = np.gradient(Y)
    dZdy, dZdx = np.gradient(Z)
    
    # Cross product of the tangent vectors
    Nx = dYdx * dZdy - dZdx * dYdy
    Ny = dZdx * dXdy - dXdx * dZdy
    Nz = dXdx * dYdy - dYdx * dXdy
    
    # Normalize the vectors
    magnitude = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    magnitude[magnitude == 0] = 1 # Avoid division by zero
    Nx, Ny, Nz = Nx/magnitude, Ny/magnitude, Nz/magnitude
    
    # Flip normals so they point "out" towards the camera (negative Z direction)
    flip_mask = Nz > 0
    Nx[flip_mask] *= -1
    Ny[flip_mask] *= -1
    Nz[flip_mask] *= -1
    
    # 4. Voxels
    X_v, Y_v, Z_v = X[valid], Y[valid], Z[valid]
    if len(X_v) > 0:
        H, _ = np.histogramdd(np.vstack([X_v, Y_v, Z_v]).T, bins=grid_size)
        voxels = H > 0
    else:
        voxels = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
        
    return X, Y, Z, Nx, Ny, Nz, valid, voxels

def main():
    try:
        pipeline, align = init_camera()
    except Exception as e:
        print(f"Failed to connect to camera: {e}")
        return

    plt.ion()
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10)) # Adjusted for 2x2 grid
    fig.canvas.manager.set_window_title('Live 4-Window RealSense Visualizer')

    # Create 2x2 Grid Layout
    ax1 = fig.add_subplot(221)                     # Top Left: Overlap
    ax2 = fig.add_subplot(222, projection='3d')    # Top Right: Continuous Surface
    ax3 = fig.add_subplot(223, projection='3d')    # Bottom Left: Voxels
    ax4 = fig.add_subplot(224, projection='3d')    # Bottom Right: Normals & Points

    im_color, im_depth = None, None
    print("\nLive view starting. Note: 4x 3D plots will cause low frame rates in Matplotlib.")

    try:
        while plt.fignum_exists(fig.number):
            color_img, depth_img, intrinsics = get_frames(pipeline, align)
            if color_img is None: continue

            # Step defines grid resolution. Lower = denser (slower). Higher = sparser (faster).
            X, Y, Z, Nx, Ny, Nz, valid, voxels = process_3d_data(depth_img, intrinsics, step=25)

            # --- Window 1: RGB-D Overlap ---
            depth_norm = depth_img / (np.max(depth_img) + 1e-6)
            if im_color is None:
                ax1.set_title("1. RGB-D Overlap")
                ax1.axis('off')
                im_color = ax1.imshow(color_img)
                im_depth = ax1.imshow(depth_norm, cmap=plt.get_cmap('jet'), alpha=0.4)
            else:
                im_color.set_data(color_img)
                im_depth.set_data(depth_norm)

            # Clear 3D axes for redrawing
            ax2.cla()
            ax3.cla()
            ax4.cla()

            # --- Window 2: Smooth 3D Surface Curve ---
            ax2.set_title("2. Surface Curve (Smoothed)")
            # We set Z to NaN where it's invalid so the surface doesn't draw walls to zero
            Z_surf = np.where(valid, Z, np.nan)
            ax2.plot_surface(X, Y, Z_surf, cmap='viridis', edgecolor='none', alpha=0.8)
            ax2.view_init(elev=-70, azim=-90) # Angle to see the curve

            # --- Window 3: Voxels ---
            ax3.set_title("3. Voxel Grid")
            colors = np.empty(voxels.shape, dtype=object)
            colors[voxels] = 'cyan'
            ax3.voxels(voxels, facecolors=colors, edgecolor='k', alpha=0.5)
            ax3.set_axis_off()

            # --- Window 4: Point Cloud + Corrected Normals ---
            ax4.set_title("4. Surface Normals (Arrows)")
            # Filter arrays to only plot valid points
            X_v, Y_v, Z_v = X[valid], Y[valid], Z[valid]
            Nx_v, Ny_v, Nz_v = Nx[valid], Ny[valid], Nz[valid]
            
            # Subsample for quiver speed/clarity
            q_skip = 3
            if len(X_v) > 0:
                ax4.scatter(X_v[::q_skip], Y_v[::q_skip], Z_v[::q_skip], c=Z_v[::q_skip], cmap='viridis', s=2)
                ax4.quiver(X_v[::q_skip], Y_v[::q_skip], Z_v[::q_skip], 
                           Nx_v[::q_skip], Ny_v[::q_skip], Nz_v[::q_skip], 
                           length=0.1, normalize=True, color='red', alpha=0.6)
            
            ax4.view_init(elev=-70, azim=-90)

            # Sync axis limits for 3D plots
            for ax in [ax2, ax4]:
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Depth Z (m)')

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        pipeline.stop()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()