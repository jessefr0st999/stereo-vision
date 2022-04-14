import numpy as np
import pandas as pd
from correlation import cross_correlate_2d
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches
from utils import max_pos

def main():
    # Read images from file as grayscale
    template_image = Image.open('images/template2.png').convert('L')
    region_image = Image.open('images/region2.png').convert('L')
    region_edit_image = Image.open('images/region2_edit.png').convert('L')

    template = np.asarray(template_image)
    region = np.asarray(region_image)
    region_edit = np.asarray(region_edit_image)

    step = 5
    region_cor = cross_correlate_2d(template, region, step_x=step, step_y=step)
    region_edit_cor = cross_correlate_2d(template, region_edit, step_x=step, step_y=step)

    region_cor_max = max_pos(region_cor, step, step)
    region_edit_cor_max = max_pos(region_edit_cor, step, step)

    rect_kwargs = {
        'linewidth': 2,
        'edgecolor': 'r',
        'facecolor': 'none',
    }
    region_cor_rect = patches.Rectangle(step * region_cor_max,
        *template.shape, **rect_kwargs)
    region_cor_edit_rect = patches.Rectangle(step * region_edit_cor_max,
        *template.shape, **rect_kwargs)

    figure = plt.figure(figsize=(2, 2))

    figure.add_subplot(2, 2, 1)
    plt.gca().add_patch(region_cor_rect)
    plt.imshow(region)
    figure.add_subplot(2, 2, 2)
    plt.imshow(region_cor)
    figure.add_subplot(2, 2, 3)
    plt.gca().add_patch(region_cor_edit_rect)
    plt.imshow(region_edit)
    figure.add_subplot(2, 2, 4)
    plt.imshow(region_edit_cor)

    plt.show()

if __name__ == '__main__':
    main()