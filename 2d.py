import numpy as np
from correlation import cross_correlate_2d
from correlation_spectral import cross_correlate_2d_spectral
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches
from utils import greyscale_with_nan, max_pos
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectral', action='store_true', default=False)
    parser.add_argument('--template', default='rm_template.png')
    parser.add_argument('--region', default='rm_region.png')
    parser.add_argument('--step', type=int, default=5)
    args = parser.parse_args()

    # convert('L') converts image to grayscale
    region_image = Image.open(f'images/{args.region}').convert('L')
    region = np.asarray(region_image)

    if args.spectral:
        template_image = Image.open(f'images/{args.template}').convert('L')
        template = np.asarray(template_image)
        step = 1
        start = datetime.now()
        region_cor = cross_correlate_2d_spectral(template, region)
    else:
        # Replace transparent values with NaNs for non-spectral
        template_image = greyscale_with_nan(f'images/{args.template}')
        template = np.asarray(template_image)
        step = args.step
        start = datetime.now()
        region_cor = cross_correlate_2d(template, region, step_x=step, step_y=step)
    print(f'Time elapsed : {datetime.now() - start}')

    region_cor_max = max_pos(region_cor, step, step)
    print(f'Maximum cross-correlation at: {region_cor_max}')

    # Draw a box around the maximum
    rect_kwargs = {
        'linewidth': 2,
        'edgecolor': 'r',
        'facecolor': 'none',
    }
    region_cor_rect = patches.Rectangle(step * region_cor_max,
        *template.T.shape, **rect_kwargs)

    figure = plt.figure(figsize=(3, 1))

    figure.add_subplot(3, 1, 1)
    plt.imshow(template)
    figure.add_subplot(3, 1, 2)
    plt.gca().add_patch(region_cor_rect)
    plt.imshow(region)
    figure.add_subplot(3, 1, 3)
    plt.imshow(region_cor)

    plt.show()

if __name__ == '__main__':
    main()