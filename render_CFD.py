import os
import pickle
from matplotlib import tri as mtri
import matplotlib.pyplot as plt
from matplotlib import animation


def save_frames(rollout_path, output_dir, skip=10, mode='pred'):
    # Load the rollout data
    with open(rollout_path, 'rb') as fp:
        rollout_data = pickle.load(fp)

    # skip = 10  # skip frames
    num_steps = rollout_data[0]['gt_velocity'].shape[0]  # 600
    num_frames = len(rollout_data) * num_steps // skip

    # Compute bounds
    bounds = []
    for trajectory in rollout_data:
        bb_min = trajectory['gt_velocity'].min(axis=(0, 1))
        bb_max = trajectory['gt_velocity'].max(axis=(0, 1))
        bounds.append((bb_min, bb_max))

    attribute = mode + '_velocity'
    # Loop over frames and save them as images
    for i in range(num_frames):
        step = (i * skip) % num_steps
        traj = (i * skip) // num_steps
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_aspect('equal')
        ax.set_axis_off()
        vmin, vmax = bounds[traj]
        pos = rollout_data[traj]['mesh_pos'][step]
        faces = rollout_data[traj]['faces'][step]
        velocity = rollout_data[traj][attribute][step]
        triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
        cax = ax.tripcolor(triang, velocity[:, 0], vmin=vmin[0], vmax=vmax[0])
        ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
        # Add a colorbar next to the plot
        # cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=0.6)
        # cbar.ax.tick_params(labelsize=12)
        ax.set_title('{} Trajectory {} Step {}'.format(mode, traj, step))
        # fig.savefig(os.path.join(output_dir, mode+f"_frame_{i:04d}.pdf"))
        fig.savefig(os.path.join(output_dir, mode + f"_frame_{i:04d}.png"), dpi=100)
        plt.close(fig)


def save_video(rollout_path, output_dir, skip=10, name=''):
    # 设置matplotlib的后端为Agg，以支持无图形界面环境下的保存
    plt.rcParams['animation.ffmpeg_path'] = '/home/ljz/anaconda3/envs/DiffNP/bin/ffmpeg'
    plt.switch_backend('Agg')

    # 读取数据
    with open(rollout_path, 'rb') as fp:
        rollout_data = pickle.load(fp)

    # 计算一些变量
    # skip = 10
    num_steps = rollout_data[0]['gt_velocity'].shape[0]
    num_frames = len(rollout_data) * num_steps // skip

    # 计算所有轨迹的边界
    bounds = []
    for trajectory in rollout_data:
        bb_min = trajectory['gt_velocity'].min(axis=(0, 1))
        bb_max = trajectory['gt_velocity'].max(axis=(0, 1))
        bounds.append((bb_min, bb_max))

    # 创建动画对象
    fig, ax = plt.subplots(figsize=(12, 8))

    def warpper_animate(mode='pred'):
        def animate(num):
            attribute = mode + '_velocity'
            step = (num * skip) % num_steps
            traj = (num * skip) // num_steps
            ax.cla()
            ax.set_aspect('equal')
            ax.set_axis_off()
            vmin, vmax = bounds[traj]
            pos = rollout_data[traj]['mesh_pos'][step]
            faces = rollout_data[traj]['faces'][step]
            if mode == 'diff':
                velocity = rollout_data[traj]['pred_velocity'][step] - rollout_data[traj]['gt_velocity'][step]
            else:
                velocity = rollout_data[traj][attribute][step]
            triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
            ax.tripcolor(triang, velocity[:, 0], vmin=vmin[0], vmax=vmax[0])
            ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
            ax.set_title('{} Trajectory {} Step {}'.format(mode, traj, step))
            return fig,

        return animate

    pred_ani = animation.FuncAnimation(fig, warpper_animate('pred'), frames=num_frames, interval=100)
    gt_ani = animation.FuncAnimation(fig, warpper_animate('gt'), frames=num_frames, interval=100)
    seen_ani = animation.FuncAnimation(fig, warpper_animate('seen'), frames=num_frames, interval=100)
    diff_ani = animation.FuncAnimation(fig, warpper_animate('diff'), frames=num_frames, interval=100)

    # 保存为视频
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    pred_ani.save(os.path.join(output_dir, 'pred_cfd_trajectory_{}.mp4'.format(name)), writer=writer)
    gt_ani.save(os.path.join(output_dir, 'gt_cfd_trajectory_{}.mp4'.format(name)), writer=writer)
    seen_ani.save(os.path.join(output_dir, 'seen_cfd_trajectory_{}.mp4'.format(name)), writer=writer)
    diff_ani.save(os.path.join(output_dir, 'diff_cfd_trajectory_{}.mp4'.format(name)), writer=writer)


if __name__ == '__main__':
    # Set up the parameters
    timestep = 30
    rollout_path = "./results/cylinder_flow_t{}/autoregressive_s120_t30_ema.pkl".format(timestep)
    save_path = "./results/cylinder_flow_t{}".format(timestep)
    # save_video(rollout_path, save_path, skip=2, name='e48_m95_t30')
    save_frames(rollout_path, save_path, skip=100, mode='pred')
    save_frames(rollout_path, save_path, skip=100, mode='gt')
    save_frames(rollout_path, save_path, skip=100, mode='seen')
