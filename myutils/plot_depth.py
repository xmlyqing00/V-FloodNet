import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


class Visualizer:

    def __init__(self, img, output_dir, img_name):
        self.img = img
        self.size = img.shape[:2]
        self.water_depth = None

        self.output_dir = output_dir
        self.img_name = img_name

    def plot_seg(self, viz_dict):

        out_path = os.path.join(self.output_dir, self.img_name + '_seg.png')
        cv2.imwrite(out_path, viz_dict['viz_img'])
        # fig, axs = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1]})
        #
        # axs[0].imshow(self.img[:, :, ::-1])
        # axs[0].get_xaxis().set_visible(False)
        # axs[0].get_yaxis().set_visible(False)
        #
        # axs[1].imshow(viz_dict['viz_img'][:, :, ::-1])
        # axs[1].get_xaxis().set_visible(False)
        # axs[1].get_yaxis().set_visible(False)
        #
        # fig.tight_layout()
        # fig.savefig(os.path.join(self.output_dir, self.img_name + '_seg.png'))
        # plt.close()
        # plt.show()

    def get_depth(self, x, y):
        return self.water_depth[y, x]

    def plot_depth(self, water_depth, vlist, water_mask, suffix=None):
        self.water_depth = water_depth
        self.water_depth[water_mask == 0] = np.NaN

        y, x = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]))

        fig, axs = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1]})

        axs[0].imshow(self.img[:, :, ::-1])
        axs[0].contourf(x, y, self.get_depth(x, y), 8)
        contours = axs[0].contour(x, y, self.get_depth(x, y), 8, colors='black')
        axs[0].clabel(contours, inline=True, fontsize=10, fmt='%.0f')

        # ax.set_ylim(ax.get_ylim()[::-1])
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)

        pcm = axs[1].contourf(x, y, self.get_depth(x, y), 8)
        if len(vlist) > 1:
            fig.colorbar(pcm, orientation='vertical', ax=axs[1], format='%d')
        else:
            axs[1].text(900, 40, f'water depth: {vlist[0]:.0f} cm', fontsize=14)
        contours = axs[1].contour(x, y, self.get_depth(x, y), 8, colors='black')
        axs[1].clabel(contours, inline=True, fontsize=10, fmt='%.0f')

        axs[1].set_ylim(axs[1].get_ylim()[::-1])
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)

        fig.tight_layout()

        if suffix:
            fig_name = self.img_name + f'_depth_by_{suffix}.png'
        else:
            fig_name = self.img_name + '_depth.png'
        fig.savefig(os.path.join(self.output_dir, fig_name))
        plt.close()
        # plt.show()
